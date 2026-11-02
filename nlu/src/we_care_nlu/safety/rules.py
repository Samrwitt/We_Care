from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from we_care_nlu.config import package_root
from we_care_nlu.contracts.schemas import RetrievalHit
from we_care_nlu.preprocess.normalize import has_negation


def load_emergency(path: Path | None = None) -> dict[str, Any]:
    path = path or package_root() / "data" / "gazetteers" / "emergency.yaml"
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# Family-planning phrases that contain an emergency cue but are not emergencies.
_BENIGN_EMERGENCY_PHRASES = (
    "የአስቸኳይ የእርግዝና መከላከያ",
    "የአስቸኳይ መከላከያ",
    "የአስቸኳይ ክኒን",
    "emergency contraception",
    "emergency contraceptive",
)


def emergency_flags(text: str) -> dict[str, Any]:
    data = load_emergency()
    low = text.lower()
    for phrase in _BENIGN_EMERGENCY_PHRASES:
        low = low.replace(phrase.lower(), " ")
    hits = []
    for cue in (data.get("amharic") or []) + (data.get("english") or []):
        if cue.lower() in low:
            hits.append(cue)
    return {
        "is_emergency": bool(hits),
        "cues": hits,
        "source": "rule_based",
        "action": data.get("action") if hits else None,
    }


def evidence_conflicts(hits: list[RetrievalHit], query: str) -> dict[str, Any]:
    if len(hits) < 2:
        return {"conflict": False}
    q = query.lower()
    query_is_polar = has_negation(query) or any(
        c in q
        for c in (
            "ተቃራኒ",
            "የተከለከለ",
            "አይመከርም",
            "ይቻላል",
            "ጡት",
            "contraindicat",
            "breastfeed",
            "can i",
            "allowed",
        )
    )
    polar = [h for h in hits if h.polarity_conflict]
    contra = sum(
        1
        for h in hits
        if "ተቃራኒ" in (h.amharic_text + h.english_text)
        or "contraindicat" in (h.english_text or "").lower()
        or "የተከለከለ" in (h.amharic_text or "")
    )
    allow = sum(
        1
        for h in hits
        if "ይቻላል" in (h.amharic_text or "") or "can be used" in (h.english_text or "").lower()
    )
    conflict = bool(polar) or (query_is_polar and contra and allow)
    return {
        "conflict": conflict,
        "polarity_mismatched_hits": [h.evidence_id for h in polar],
        "note": "Opposing eligibility vs contraindication statements were retrieved." if conflict else "",
    }


def review_warnings(hits: list[RetrievalHit]) -> list[str]:
    notes = []
    if any(h.review_status == "unreviewed" for h in hits):
        notes.append("One or more hits are unreviewed machine translations, not clinically verified.")
    if hits and all(h.review_status == "unreviewed" for h in hits):
        notes.append("No human-reviewed evidence was retrieved.")
    return notes
