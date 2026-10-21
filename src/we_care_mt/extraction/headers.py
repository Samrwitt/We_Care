"""Repeating header/footer detection."""

from __future__ import annotations

from collections import Counter

from we_care_mt.constants import KNOWN_RUNNING_HEADERS


def normalize_running_text(text: str) -> str:
    return " ".join(text.split())


def detect_repeating_bands(
    page_band_texts: list[list[str]],
    min_pages: int = 3,
) -> list[str]:
    counts: Counter[str] = Counter()
    for texts in page_band_texts:
        for raw in texts:
            t = normalize_running_text(raw)
            if len(t) < 8:
                continue
            counts[t] += 1
            # Also count prefix matches against known MoH running heads.
            for known in KNOWN_RUNNING_HEADERS:
                if known.lower() in t.lower():
                    counts[known] += 1
    return [t for t, n in counts.most_common() if n >= min_pages]


def is_running_element(text: str, known: list[str]) -> bool:
    t = normalize_running_text(text)
    if not t:
        return False
    low = t.lower()
    for item in known:
        if not item:
            continue
        k = item.lower()
        if k in low or low in k:
            return True
        # roman or arabic page numbers sitting on the running footer line
        if k[:40] in low:
            return True
    return False


def looks_like_page_number(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if t.isdigit() and len(t) <= 4:
        return True
    roman = set("ivxlcdmIVXLCDM")
    if 1 <= len(t) <= 6 and all(ch in roman for ch in t):
        return True
    return False
