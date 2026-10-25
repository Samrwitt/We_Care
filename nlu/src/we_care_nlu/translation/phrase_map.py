"""Amharic-to-English phrase map for retrieval when NLLB weights are not loaded.

This covers the lexicon and gazetteer. It does not translate guideline sentences.
"""

from __future__ import annotations

import re
from functools import lru_cache

from we_care_nlu.ner.extract import load_gazetteer
from we_care_nlu.retrieval.expand import synonym_groups

_ETH = re.compile(r"[\u1200-\u137F]")

# Longer frames first. English sides keep the polarity words the reranker looks for.
_FRAMES: list[tuple[str, str]] = [
    ("የአስቸኳይ የእርግዝና መከላከያ", "emergency contraception"),
    ("የእርግዝና መከላከያ ዘዴዎች ምንድን ናቸው", "what contraceptive methods are there"),
    ("የእርግዝና መከላከያ", "contraceptive"),
    ("እርግዝናን ለመከላከል ምን አማራጮች አሉ", "what contraceptive options are there"),
    ("የጎንዮሽ ጉዳቱስ", "what about the side effects"),
    ("የጎንዮሽ ጉዳት", "side effects"),
    ("ጡት እያጠባሁ", "while breastfeeding"),
    ("ጡት በማጥባት ጊዜ", "while breastfeeding"),
    ("ጡት ማጥባት", "breastfeeding"),
    ("መጠቀም ይቻላል", "can be used"),
    ("መጠቀም አይቻልም", "must not be used"),
    ("አይመከርም", "contraindicated"),
    ("የተከለከለ", "contraindicated"),
    ("ተቃራኒ", "contraindicated"),
    ("የት ይገኛል", "where is it available"),
    ("የት እገኛለሁ", "where can I get"),
    ("እንዴት ይሠራል", "how does it work"),
    ("እንዴት ይሰራል", "how does it work"),
    ("ለምን ያህል ጊዜ", "how long"),
    ("ምን ያህል ጊዜ", "how long"),
    ("ከተረሳ", "missed dose"),
    ("ረሳሁ", "missed dose"),
    ("ምንድን ናቸው", "what are they"),
    ("ምንድን ነው", "what is"),
    ("ከባድ መድማት", "heavy bleeding"),
    ("መተንፈስ አልችልም", "cannot breathe"),
]


@lru_cache(maxsize=1)
def _pairs() -> list[tuple[str, str]]:
    pairs = list(_FRAMES)
    gazetteer = load_gazetteer()
    for values in gazetteer.values():
        if not isinstance(values, dict):
            continue
        for canonical, surfaces in values.items():
            if not isinstance(surfaces, list):
                continue
            latin = [str(item) for item in surfaces if not _ETH.search(str(item))]
            amharic = [str(item) for item in surfaces if _ETH.search(str(item))]
            target = latin[0] if latin else str(canonical).replace("_", " ")
            for surface in amharic:
                pairs.append((surface, target))
    for group in synonym_groups():
        latin = [term for term in group if not _ETH.search(term)]
        amharic = [term for term in group if _ETH.search(term)]
        if not latin or not amharic:
            continue
        for surface in amharic:
            pairs.append((surface, latin[0]))
    unique: dict[str, str] = {}
    for source, target in sorted(pairs, key=lambda item: len(item[0]), reverse=True):
        key = source.casefold()
        unique.setdefault(key, target)
    ordered = [(source, unique[source.casefold()]) for source, _ in pairs]
    # Longest surface wins, and the first English gloss for that surface is kept.
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for source, target in sorted(ordered, key=lambda item: len(item[0]), reverse=True):
        key = source.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append((source, target))
    return result


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(not (end <= left or start >= right) for left, right in spans)


def map_to_english(text: str) -> str:
    original = text or ""
    occupied: list[tuple[int, int]] = []
    hits: list[tuple[int, int, str]] = []
    folded = original.casefold()
    for source, target in _pairs():
        start = 0
        needle = source.casefold()
        while True:
            idx = folded.find(needle, start)
            if idx < 0:
                break
            end = idx + len(source)
            if not _overlaps(idx, end, occupied):
                hits.append((idx, end, target))
                occupied.append((idx, end))
            start = end
    hits.sort()
    parts: list[str] = []
    cursor = 0
    for start, end, target in hits:
        parts.append(original[cursor:start])
        parts.append(target)
        cursor = end
    parts.append(original[cursor:])
    return " ".join("".join(parts).split())


class PhraseMapTranslator:
    """Query translator only. Reply sentences stay in English until NLLB is enabled."""

    can_translate_reply = False
    backend = "map"

    def to_english(self, text: str) -> str:
        return map_to_english(text)

    def to_amharic(self, text: str) -> str:
        raise RuntimeError("Phrase map does not translate guideline sentences.")
