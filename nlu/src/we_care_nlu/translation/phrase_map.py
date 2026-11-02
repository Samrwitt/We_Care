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
    ("የቤተሰብ ምጣኔ ዘዴዎች", "family planning methods"),
    ("የእርግዝና መከላከያ", "contraceptive"),
    ("እርግዝናን ለመከላከል ምን አማራጮች አሉ", "what contraceptive options are there"),
    ("የወር አበባ መዛባት", "irregular menstrual cycle"),
    ("የወር አበባ ተዛባ", "irregular period"),
    ("የወር አበባ ሲያመኝ", "period pain"),
    ("የወር አበባ ህመም", "period pain"),
    ("የወር አበባ ሕመም", "period pain"),
    ("ክሪፕቲክ እርግዝና", "cryptic pregnancy"),
    ("ስውር እርግዝና", "cryptic pregnancy"),
    ("ኤች.አይ.ቪ ሕክምና", "HIV treatment"),
    ("ኤች.አይ.ቪ ህክምና", "HIV treatment"),
    ("ኤችአይቪ ሕክምና", "HIV treatment"),
    ("ከማሕፀን ውጪ እርግዝና", "ectopic pregnancy"),
    ("ከማህፀን ውጭ እርግዝና", "ectopic pregnancy"),
    ("የጎንዮሽ ጉዳቱስ", "what about the side effects"),
    ("የጎንዮሽ ጉዳት", "side effects"),
    ("ጡት እያጠባሁ", "while breastfeeding"),
    ("ጡት በማጥባት ጊዜ", "while breastfeeding"),
    ("ጡት ማጥባት", "breastfeeding"),
    ("መጠቀም ይቻላል", "can be used"),
    ("መጠቀም እችላለሁ", "can I use"),
    ("መጠቀም አይቻልም", "must not be used"),
    ("አይመከርም", "contraindicated"),
    ("የተከለከለ", "contraindicated"),
    ("ተቃራኒ", "contraindicated"),
    ("የት ይገኛል", "where is it available"),
    ("የት እገኛለሁ", "where can I get"),
    ("የት ልሂድ", "where should I go"),
    ("እንዴት ይሠራል", "how does it work"),
    ("እንዴት ይሰራል", "how does it work"),
    ("እንዴት ይታከማል", "how is it treated"),
    ("ምን ላድርግ", "what should I do"),
    ("ምን ላደርግ", "what should I do"),
    ("ለምን ያህል ጊዜ", "how long"),
    ("ምን ያህል ጊዜ", "how long"),
    ("ስንት አመት ይቆያል", "how many years does it last"),
    ("ከተረሳ", "missed dose"),
    ("ረሳሁ", "missed dose"),
    ("ክኒን ከተረሳ", "missed pill"),
    ("ምንድን ናቸው", "what are they"),
    ("ምንድን ነው", "what is"),
    ("ምንድር ነው", "what is"),
    ("ከባድ መድማት", "heavy bleeding"),
    ("መተንፈስ አልችልም", "cannot breathe"),
    ("የብልት ፈሳሽ", "vaginal discharge"),
    ("የብልት ቁስለት", "genital ulcer"),
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


def covered_question(text: str) -> str | None:
    """English query when the phrase list covers the whole question.

    Known questions stay on this list so retrieval keeps its polarity words
    and the model is only asked to translate the guideline sentence.
    """
    mapped = map_to_english(text)
    if not mapped.strip() or _ETH.search(mapped):
        return None
    return mapped


def map_to_amharic_terms(text: str, *, limit: int = 16) -> list[str]:
    """Amharic surfaces whose English gloss appears in an English ask (for BM25)."""
    low = (text or "").casefold()
    if not low or _ETH.search(low):
        return []
    out: list[str] = []
    seen: set[str] = set()
    # Longer English glosses first so "emergency contraception" beats "contraception".
    pairs = sorted(_pairs(), key=lambda item: len(item[1]), reverse=True)
    for amharic, english in pairs:
        gloss = (english or "").casefold().strip()
        if len(gloss) < 3 or gloss not in low:
            continue
        key = amharic.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(amharic)
        if len(out) >= limit:
            break
    return out


class PhraseMapTranslator:
    """Query translator only. Reply sentences stay in English until NLLB is enabled."""

    can_translate_reply = False
    backend = "map"

    def to_english(self, text: str) -> str:
        return map_to_english(text)

    def to_amharic(self, text: str) -> str:
        raise RuntimeError("Phrase map does not translate guideline sentences.")
