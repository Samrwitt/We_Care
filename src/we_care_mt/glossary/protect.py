"""Placeholder protection for numbers, units, drug names, and freeze terms."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from we_care_mt.constants import PLACEHOLDER_PREFIX

if TYPE_CHECKING:
    from we_care_mt.glossary.manager import Glossary

# Numbers with optional units / percents / ratios. Keep source glyphs intact.
_PROTECT_RES = [
    re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|kg|ml|mL|L|iu|IU|mmHg|mmol|mEq|%|/100,?000|/1,?000)\b"),
    re.compile(r"\b\d+\s*/\s*\d+(?:,\d{3})*\b"),
    re.compile(r"\b\d+(?:\.\d+)?%\b"),
]


def apply_protection(text: str, glossary: Glossary | None = None) -> tuple[str, dict[str, str]]:
    mapping: dict[str, str] = {}
    counter = 0

    def _sub(match: re.Match[str]) -> str:
        nonlocal counter
        original = match.group(0)
        # De-duplicate identical protected spans.
        for key, val in mapping.items():
            if val == original:
                return key
        counter += 1
        key = f"{PLACEHOLDER_PREFIX}{counter:04d}"
        mapping[key] = original
        return key

    out = text
    if glossary:
        for entry in glossary.find_hits(text):
            if entry.preserve_source or entry.confidence == "do_not_translate":
                pattern = re.compile(rf"(?i)(?<![A-Za-z]){re.escape(entry.english)}(?![A-Za-z])")
                out = pattern.sub(_sub, out)
        for rx in glossary.freeze_patterns:
            out = rx.sub(_sub, out)
    for rx in _PROTECT_RES:
        out = rx.sub(_sub, out)
    return out, mapping


def restore_protection(text: str, mapping: dict[str, str]) -> str:
    out = text
    # Restore longest keys first in case of accidental prefix overlap.
    for key in sorted(mapping, key=len, reverse=True):
        out = out.replace(key, mapping[key])
        # Models sometimes insert spaces into placeholders: WCMT 0001
        spaced = re.sub(r"(WCMT)(\d{4})", r"\1 \2", key)
        out = out.replace(spaced, mapping[key])
    return out
