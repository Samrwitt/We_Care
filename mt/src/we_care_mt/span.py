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
        signature = f"span|{direction}|beams={self.num_beams}"
        notes: list[str] = []
        with self._lock:
            if self.cache is not None:
                cached = self.cache.get(self.model_id, source, signature)
                if cached is not None:
                    return {"text": cached, "direction": direction, "qa_notes": ["cache_hit"]}
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
        if direction == "en-am":
            translated = restore_protection(translated, mapping)
            if self.glossary is not None:
                translated, enforced = self.glossary.enforce_target(source, translated)
                notes.extend(enforced)
        if "WCMT" in translated:
            notes.append("placeholder_left")
        translated = translated.strip()
        if self.cache is not None and "placeholder_left" not in notes:
            self.cache.put(self.model_id, source, signature, translated)
        return {"text": translated, "direction": direction, "qa_notes": notes}
