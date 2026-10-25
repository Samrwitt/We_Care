"""Glossary-aware mock translator for tests. Not a medical MT system."""

from __future__ import annotations

import re

from we_care_mt.glossary.manager import Glossary


_FUNCTION_WORDS = {
    "the": "",
    "a": "",
    "an": "",
    "and": "እና",
    "or": "ወይም",
    "of": "የ",
    "to": "ወደ",
    "for": "ለ",
    "in": "ውስጥ",
    "with": "ከ",
    "by": "በ",
    "is": "ነው",
    "are": "ናቸው",
    "this": "ይህ",
    "that": "ያ",
}


class MockTranslator:
    """Deterministic, model-free backend used by the smoke test.

    Known glossary phrases are replaced. Remaining alphabetic tokens stay in
    English so QA can flag untranslated remainder. Numbers and units are kept.
    """

    model_id = "mock"

    def __init__(self, glossary: Glossary | None = None):
        self.glossary = glossary

    def translate_batch(self, texts: list[str], **_kwargs) -> list[str]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> str:
        out = text
        if self.glossary:
            for entry in self.glossary.find_hits(text):
                if not entry.amharic or entry.preserve_source:
                    continue
                out = re.sub(
                    rf"(?i)(?<![A-Za-z]){re.escape(entry.english)}(?![A-Za-z])",
                    entry.amharic,
                    out,
                )
        tokens = re.split(r"(\s+)", out)
        rebuilt: list[str] = []
        for tok in tokens:
            low = tok.lower()
            if low in _FUNCTION_WORDS:
                rebuilt.append(_FUNCTION_WORDS[low] if _FUNCTION_WORDS[low] else tok)
            else:
                rebuilt.append(tok)
        result = "".join(rebuilt)
        result = re.sub(r" {2,}", " ", result).strip()
        if not any("\u1200" <= ch <= "\u137F" for ch in result):
            result = f"‹ትርጉም› {result}"
        return result
