"""Translate one retrieved sentence. The chat looks the sentence up; this does not read a PDF."""

from __future__ import annotations

import threading
from typing import Any, Protocol

from we_care_mt.constants import NLLB_SRC_LANG, NLLB_TGT_LANG
from we_care_mt.glossary.manager import Glossary
from we_care_mt.glossary.protect import apply_protection, restore_protection
from we_care_mt.translation.cache import TranslationCache

_DIRECTIONS = {
    "en-am": (NLLB_SRC_LANG, NLLB_TGT_LANG),
    "am-en": (NLLB_TGT_LANG, NLLB_SRC_LANG),
}


class LangTranslator(Protocol):
    def translate_langs(self, texts: list[str], *, src_lang: str, tgt_lang: str, **kwargs: Any) -> list[str]:
        ...


class SpanTranslator:
    """One Facebook NLLB model, both directions, glossary applied only on English→Amharic."""

    def __init__(
        self,
        translator: LangTranslator,
        glossary: Glossary | None = None,
        cache: TranslationCache | None = None,
        model_id: str = "nllb",
        num_beams: int = 1,
    ):
        self.translator = translator
        self.glossary = glossary
        self.cache = cache
        self.model_id = model_id
        self.num_beams = max(1, num_beams)
        self._lock = threading.Lock()

    def translate(self, text: str, direction: str) -> dict[str, Any]:
        pair = _DIRECTIONS.get(direction)
        if pair is None:
            raise ValueError("direction must be en-am or am-en")
        source = text or ""
        if not source.strip():
            return {"text": "", "direction": direction, "qa_notes": []}
        src_lang, tgt_lang = pair
        signature = f"span|v2|{direction}|beams={self.num_beams}"
        notes: list[str] = []
        with self._lock:
            if self.cache is not None:
                cached = self.cache.get(self.model_id, source, signature)
                if cached is not None:
                    return self._finish(source, cached, direction, notes=["cache_hit"], mapping={})
            protected = source
            mapping: dict[str, str] = {}
            if direction == "en-am":
                protected, mapping = apply_protection(source, self.glossary)
            # Decoding stops at the end-of-sentence token. The cap only bounds a runaway.
            max_new = min(192, max(32, len(source.split()) * 2 + 8))
            translated = self.translator.translate_langs(
                [protected],
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                max_input_tokens=256,
                max_new_tokens=max_new,
                num_beams=self.num_beams,
            )[0]
        return self._finish(source, translated, direction, notes=[], mapping=mapping)

    def _glossary_notes(self, source: str, translated: str) -> tuple[str, list[str]]:
        notes: list[str] = []
        if self.glossary is None:
            return translated, notes
        translated, enforced = self.glossary.enforce_target(source, translated)
        notes.extend(enforced)
        for entry in self.glossary.find_hits(source):
            if entry.preserve_source or not entry.amharic:
                continue
            if entry.confidence not in {"verified", "standard"}:
                continue
            if entry.amharic not in translated:
                notes.append(f"term_missing:{entry.english}")
        return translated, notes

    def _finish(
        self,
        source: str,
        translated: str,
        direction: str,
        *,
        notes: list[str],
        mapping: dict[str, str],
    ) -> dict[str, Any]:
        if direction == "en-am":
            if mapping:
                translated = restore_protection(translated, mapping)
            translated, glossary_notes = self._glossary_notes(source, translated)
            notes.extend(glossary_notes)
        if "WCMT" in translated:
            notes.append("placeholder_left")
        translated = translated.strip()
        if (
            self.cache is not None
            and "cache_hit" not in notes
            and "placeholder_left" not in notes
            and not any(note.startswith("term_missing") for note in notes)
        ):
            signature = f"span|v2|{direction}|beams={self.num_beams}"
            self.cache.put(self.model_id, source, signature, translated)
        return {"text": translated, "direction": direction, "qa_notes": notes}
