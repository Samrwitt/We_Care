"""Freeze numbers, units, and acronyms across a translation hop."""

from __future__ import annotations

import re

PREFIX = "WCMT"

_PROTECT = [
    re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"),
    re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|kg|ml|mL|L|iu|IU|mmHg|mmol|mEq|%)\b",
        re.I,
    ),
    re.compile(r"\b\d+\s*/\s*\d+(?:,\d{3})*\b"),
    re.compile(r"\b\d+(?:\.\d+)?%\b"),
    re.compile(r"\b(?:IUCD|IUD|DMPA|HIV|AIDS|PMTCT|STI|FP|ANC|PNC|PPH)\b"),
]


def protect(text: str) -> tuple[str, dict[str, str]]:
    mapping: dict[str, str] = {}
    counter = 0

    def _sub(match: re.Match[str]) -> str:
        nonlocal counter
        original = match.group(0)
        for key, val in mapping.items():
            if val == original:
                return key
        counter += 1
        key = f"{PREFIX}{counter:04d}"
        mapping[key] = original
        return key

    out = text
    for pattern in _PROTECT:
        out = pattern.sub(_sub, out)
    return out, mapping


def restore(text: str, mapping: dict[str, str]) -> str:
    out = text
    for key in sorted(mapping, key=len, reverse=True):
        out = out.replace(key, mapping[key])
        spaced = key.replace(PREFIX, f"{PREFIX} ")
        out = out.replace(spaced, mapping[key])
    return out
