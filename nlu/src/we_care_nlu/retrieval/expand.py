"""Additive query expansion. Does not replace polarity checks."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import yaml

from we_care_nlu.config import package_root

METHOD_AM = {
    "injectable_contraceptive": "መርፌ",
    "implant": "ኢምፕላንት",
    "iud": "አይዩሲዲ",
    "condom": "ኮንዶም",
    "oral_pill": "ክኒን",
    "emergency_contraception": "የአስቸኳይ መከላከያ",
    "sterilization": "ቋሚ ዘዴ",
    "lam": "ጡት ማጥባት መከላከያ",
}

INTENT_NEEDLES = {
    "side_effects": ("የጎንዮሽ ጉዳት", "side effect", "irregular bleeding", "መደበኛ ያልሆነ"),
    "contraindications": ("የተከለከለ", "ተቃራኒ", "contraindicat", "must not"),
    "eligibility": ("መጠቀም ይቻላል", "can be used", "may be used"),
    "definition": ("is a", "is an"),
    "access": ("ይገኛል", "available", "health facilit"),
    "duration": ("ዓመት", "ሳምንት", "years", "weeks", "every"),
    "family_planning_methods": ("ዘዴዎች", "አማራጮች", "options", "include"),
    "how_it_works": ("ይሠራል", "works by", "stopping ovulation"),
    "missed_dose": ("ከተረሳ", "missed", "as soon as remembered"),
}


@lru_cache(maxsize=1)
def synonym_groups() -> list[list[str]]:
    path: Path = package_root() / "data" / "synonyms" / "amharic_medical.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    groups = []
    for terms in (data.get("groups") or {}).values():
        cleaned = [str(t).strip() for t in (terms or []) if str(t).strip()]
        if cleaned:
            groups.append(cleaned)
    return groups


def unique_join(parts: Iterable[str]) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        token = (part or "").strip()
        key = token.lower()
        if not token or key in seen:
            continue
        seen.add(key)
        out.append(token)
    return " ".join(out)


_QUERY_FILLER = {
    "ምንድን",
    "ምንድር",
    "ነው",
    "ናቸው",
    "ለምን",
    "እንዴት",
    "መቼ",
    "የት",
    "what",
    "is",
    "are",
    "how",
    "why",
}


def _normalize_am_acronyms(text: str) -> str:
    """Collapse dotted Ethiopic/Latin acronyms: አይ.ዩ.ዲ → አይዩዲ, I.U.D → IUD."""
    import re

    text = re.sub(r"(?<=[\u1200-\u137FA-Za-z])\.(?=[\u1200-\u137FA-Za-z])", "", text or "")
    return text


def expand_query(text: str, extra: Iterable[str] | None = None) -> str:
    text = _normalize_am_acronyms(text)
    # Drop bare question fillers so BM25 does not rank every FAQ title first.
    kept = [tok for tok in (text or "").split() if tok.strip("?.!፡።").lower() not in _QUERY_FILLER]
    core = " ".join(kept) if kept else text
    bits = [text, core]
    low = core.lower()
    compact = _normalize_am_acronyms(low.replace(" ", ""))
    for terms in synonym_groups():
        if any(term.lower() in low or _normalize_am_acronyms(term.lower()) in compact for term in terms):
            bits.extend(terms)
    if extra:
        bits.extend(extra)
    return unique_join(bits)


def method_terms(entities: Iterable[object]) -> list[str]:
    terms: list[str] = []
    for ent in entities or []:
        typ = getattr(ent, "type", None) or (ent.get("type") if isinstance(ent, dict) else None)
        if typ != "contraceptive_method":
            continue
        value = getattr(ent, "value", None) or (ent.get("value") if isinstance(ent, dict) else "")
        surface = getattr(ent, "surface", None) or (ent.get("surface") if isinstance(ent, dict) else "")
        if surface:
            terms.append(str(surface))
        if value:
            terms.append(str(value))
            terms.append(METHOD_AM.get(str(value), ""))
    return [t for t in terms if t]
