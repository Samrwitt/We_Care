"""Cue lists and helpers for multi-turn context resolution.

Used by resolve.resolve_context(). Clinical cues (e.g. ሲያመኝ) prevent sticky
FP-method carry-over from earlier turns.
"""

from __future__ import annotations

from we_care_nlu.contracts.schemas import ContextEntity, ContextState, Entity
from we_care_nlu.preprocess.normalize import preprocess

FOLLOW_CUES = (
    "የጎንዮሽ ጉዳቱስ",
    "ጉዳቱስ",
    "እንዴትስ",
    "ለምንስ",
    "እና ከዚያ",
    "what about",
    "and the",
    "its side",
)

CONDITION_TYPES = {"condition", "breastfeeding_status", "pregnancy_status"}
# Intents that are complete topical asks — never force "which FP method?".
STANDALONE_INTENTS = {
    "definition",
    "family_planning_methods",
    "access",
    "how_it_works",
    "duration",
    "emergency",
    "greeting",
    "conversation_end",
    "out_of_domain",
}
CARRY_INTENTS = {
    "eligibility",
    "contraindications",
    "side_effects",
    "pregnancy_breastfeeding",
    "follow_up",
    "treatment",
    "symptoms",
    "prevention",
    "causes",
    "screening",
    "medication",
    "dosage",
}

# Clinical asks that stand on their own — do NOT attach a prior FP method / BF status.
# Example bug: after injectable+breastfeeding, "ሲያመኝ ምን ላድርግ?" must not answer Depo+BF.
_METHOD_INDEPENDENT_CUES = (
    "ሲያመኝ",
    "ያመኛል",
    "መዛባት",
    "ተዛባ",
    "ወር አበባ",
    "የወር አበባ",
    "ቁርጠት",
    "period pain",
    "irregular period",
    "irregular cycle",
    "dysmenorrhea",
    "menstrual",
    "cryptic",
    "ክሪፕቲክ",
    "ስውር እርግዝና",
    "ቁስለት",
    "ፈሳሽ",
    "ሽንት ማቃጠል",
    "endometriosis",
    "ኢንዶሜትሪዮሲስ",
    "pcos",
    "ፒሲኦኤስ",
)
_METHOD_SURFACES = (
    "መርፌ",
    "ክኒን",
    "ኪኒን",
    "ኢምፕላንት",
    "በትር",
    "ኮንዶም",
    "አይዩ",
    "አይ.ዩ",
    "ሉፕ",
    "ፖስት",
    "pill",
    "implant",
    "injectable",
    "depo",
    "dmpa",
    "condom",
    "iud",
    "iucd",
    "jadelle",
)


def _mentions_fp_method(text: str) -> bool:
    low = (text or "").casefold()
    return any(surface.casefold() in low for surface in _METHOD_SURFACES)


def _method_independent_clinical(text: str) -> bool:
    """True when the ask is about a clinical symptom/condition, not a prior FP method."""
    blob = text or ""
    if not any(cue in blob for cue in _METHOD_INDEPENDENT_CUES):
        return False
    return not _mentions_fp_method(blob)


def looks_like_followup(text: str) -> bool:
    pre = preprocess(text)
    if any(c in pre.normalized for c in FOLLOW_CUES):
        return True
    # Elliptical follow-ups only (e.g. "ጎንዮሽ?" / "ለምን?"). Full questions like
    # "ክሪፕቲክ እርግዝና ምንድን ነው?" must not inherit "which method?" from prior turns.
    body = pre.normalized.strip().rstrip("?؟።")
    toks = [t for t in body.split() if t]
    return len(toks) <= 2 and pre.normalized.endswith(("?", "؟"))


def _as_entity(item: ContextEntity | Entity | dict) -> Entity:
    if isinstance(item, Entity):
        return item
    if isinstance(item, ContextEntity):
        return Entity(
            type=item.type,
            value=item.value,
            surface=item.surface or item.value,
            start=-1,
            end=-1,
            confidence=0.7,
            source="context",
        )
    return Entity(
        type=str(item.get("type") or ""),
        value=str(item.get("value") or ""),
        surface=str(item.get("surface") or item.get("value") or ""),
        start=-1,
        end=-1,
        confidence=0.7,
        source="context",
    )


