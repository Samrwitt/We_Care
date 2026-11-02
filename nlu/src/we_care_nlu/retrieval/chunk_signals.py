"""Chunk / entity signals for medical retrieval (methods, BF, HIV, curated IDs)."""

from __future__ import annotations

import re
from typing import Any, Iterable

from we_care_nlu.retrieval.expand import INTENT_NEEDLES, method_terms
from we_care_nlu.retrieval.query_facets import (
    distinctive_coverage,
    distinctive_overlap,
    distinctive_query_tokens,
    soft_topic_groups,
    soft_topic_hit,
    unspecified_distractor,
)

BF_CUES = (
    "ጡት ማጥባት",
    "ጡት በማጥባት",
    "ጡት እያጠባ",
    "እያጠባሁ",
    "ማጥባት",
    "breastfeed",
    "breastfeeding",
)

LAM_CUES = ("ሊሊ", " lam", "lam ", "lactational amenorrhea", "lactational")

METHOD_LIST_CUES = (
    "ኮንዶም", "ክኒን", "መርፌ", "ኢምፕላንት", "አይዩ", "አይ.ዩ", "አስቸኳይ", "ቋሚ",
    "condom", "pill", "injectable", "implant", "iud", "iucd", "steriliz",
)

INTENT_CHUNK_HINTS: dict[str, tuple[str, ...]] = {
    "definition": ("_def",),
    "side_effects": ("_se",),
    "missed_dose": ("_missed",),
    "access": ("_access", "chunk_where"),
    "duration": ("_def", "chunk_implant_def", "chunk_injectable_def", "chunk_iucd_def"),
    "how_it_works": ("_how",),
    "eligibility": ("chunk_bf_allowed", "chunk_bf_allowed_pop", "chunk_hiv_fp", "chunk_adolescents"),
    "contraindications": ("chunk_bf_contra", "chunk_emergency_contra", "_contra"),
    "family_planning_methods": ("chunk_methods",),
    "pregnancy_breastfeeding": ("chunk_bf_", "chunk_lam"),
    "emergency": ("chunk_ec", "_ec"),
}

METHOD_CHUNK_PREFIX: dict[str, str] = {
    "injectable_contraceptive": "chunk_injectable",
    "implant": "chunk_implant",
    "iud": "chunk_iucd",
    "condom": "chunk_condom",
    "oral_pill": "chunk_pill",
    "emergency_contraception": "chunk_ec",
    "sterilization": "chunk_permanent",
    "lam": "chunk_lam",
}

CONDITION_CHUNK: dict[str, tuple[str, ...]] = {
    # First id is the curated *FP eligibility* slice — only for FP-scoped intents.
    "hiv": ("chunk_hiv_fp", "ኤች.አይ.ቪ", "ኤችአይቪ", "hiv", "aids", "ኤድስ"),
}

FP_SCOPED_INTENTS = {
    "eligibility",
    "family_planning_methods",
    "contraindications",
    "family_planning",
    "pregnancy_breastfeeding",
}

CLINICAL_CONDITION_INTENTS = {
    "treatment",
    "medication",
    "symptoms",
    "causes",
    "prevention",
    "screening",
    "definition",
    "diagnosis_info",
}

HIV_THERAPY_CUES = (
    "ኤ.አር.ቲ",
    "ኤአርቲ",
    "ኤ.አር.ቲ.",
    "antiretroviral",
    "ፀረ-ሬትሮቫይረስ",
    "ፀረ ሬትሮቫይረስ",
    "የኤች.አይ.ቪ ሕክምና",
    "የኤችአይቪ ሕክምና",
    "የኤች.አይ.ቪ መድኃኒት",
    "የኤች.አይ.ቪ መድሃኒት",
    "የኤችአይቪ መድኃኒት",
    "ፀረ ኤች",
    "ፀረ-ኤች",
    "viral load",
    "የቫይረስ መጠን",
    "ኤች.አይ.ቪ መድኃኒት",
    "ኤች.አይ.ቪ መድሃኒት",
)

_HIV_ART_TOKEN = re.compile(r"(?<![A-Za-z])ART(?![A-Za-z])")

_OFF_TOPIC_COINFECTION = (
    "ቂጥኝ",
    "syphilis",
    "ጨብጥ",
    "gonorrh",
    "ክላሚዲያ",
    "chlamydia",
    "ኤችኤስቪ",
    "hsv",
    "herpes",
    "ጂኤንኤስ",
)

ACCESS_SERVICE_CUES = (
    "አገልግሎት",
    "ጤና ኬላ",
    "ጤና ጣቢያ",
    "ሆስፒታል",
    "health post",
    "health centre",
    "health center",
    "hospital",
    "family planning service",
)

def is_demo_chunk(chunk: Any) -> bool:
    eid = str(getattr(chunk, "evidence_id", "") or "")
    ver = str(getattr(chunk, "document_version", "") or "")
    return ver == "demo" or eid.startswith("chunk_")

def _ent_type_value(ent: Any) -> tuple[str, str]:
    typ = getattr(ent, "type", None) or (ent.get("type") if isinstance(ent, dict) else None)
    value = getattr(ent, "value", None) or (ent.get("value") if isinstance(ent, dict) else "")
    return str(typ or ""), str(value or "")

def entity_method_values(entities: Iterable[Any] | None) -> list[str]:
    out: list[str] = []
    for ent in entities or []:
        typ, value = _ent_type_value(ent)
        if typ != "contraceptive_method" or not value:
            continue
        out.append(value)
    return out

def entity_condition_values(entities: Iterable[Any] | None) -> list[str]:
    out: list[str] = []
    for ent in entities or []:
        typ, value = _ent_type_value(ent)
        if typ != "condition" or not value:
            continue
        out.append(value.lower())
    return out

def _condition_lexical_hit(blob: str, cond: str) -> bool:
    hints = CONDITION_CHUNK.get(cond) or ()
    # Skip curated evidence-id token at index 0.
    return any(cue.lower() in blob for cue in hints[1:] if cue)

def _hiv_therapy_hit(blob: str) -> bool:
    text = blob or ""
    low = text.lower()
    if any(c.lower() in low for c in HIV_THERAPY_CUES):
        return True
    # Whole-token ART only — avoid matching inside "FP የሚጠቀሙ…" / random Latin.
    return bool(_HIV_ART_TOKEN.search(text))

def has_breastfeeding(query: str, entities: Iterable[Any] | None = None) -> bool:
    q = (query or "").lower()
    if any(c.lower() in q for c in BF_CUES):
        return True
    for ent in entities or []:
        typ = getattr(ent, "type", None) or (ent.get("type") if isinstance(ent, dict) else None)
        if typ == "breastfeeding_status":
            return True
    return False

def blob_of(chunk: Any) -> str:
    return f"{getattr(chunk, 'amharic_text', '') or ''} {getattr(chunk, 'english_text', '') or ''}".lower()

def curated_aligned_with_query(
    *,
    query: str,
    blob: str,
    entities: Iterable[Any] | None,
) -> bool:
    """Whether a curated demo chunk is on-topic for this ask."""
    conditions = entity_condition_values(entities)
    methods = entity_method_values(entities)
    low = (blob or "").lower()
    # Condition-scoped asks (HIV): require condition cues; don't accept BF-only slices.
    if conditions:
        for cond in conditions:
            hints = CONDITION_CHUNK.get(cond) or ()
            if any(cue.lower() in low for cue in hints):
                return True
        return False
    if methods:
        return method_hit(blob, entities)
    if has_breastfeeding(query, entities) and bf_hit(blob):
        return True
    dist = distinctive_query_tokens(query)
    if dist:
        return distinctive_overlap(blob, dist) > 0
    return True

def method_hit(blob: str, entities: Iterable[Any] | None) -> bool:
    terms = method_terms(entities or [])
    if not terms:
        return True
    low = blob.lower()
    return any(t.lower() in low for t in terms if t)

def bf_hit(blob: str) -> bool:
    low = blob.lower()
    return any(c.lower() in low for c in BF_CUES)

def intent_needle_hit(blob: str, intent: str | None) -> bool:
    if not intent:
        return False
    low = blob.lower()
    return any(n.lower() in low for n in INTENT_NEEDLES.get(intent) or ())

def curated_intent_match(
    evidence_id: str,
    intent: str | None,
    method_values: list[str],
    *,
    conditions: list[str] | None = None,
    query: str = "",
) -> float:
    """How well a curated chunk id matches the ask (0–1+)."""
    eid = evidence_id or ""
    score = 0.0
    q = (query or "").lower()
    general_access = intent == "access" and not method_values and (
        "አገልግሎት" in q or "ቤተሰብ" in q or "family planning" in q or "የት" in q
    )
    if intent:
        for hint in INTENT_CHUNK_HINTS.get(intent) or ():
            if hint in eid:
                # General "where is FP service?" → chunk_where over condom_access.
                if general_access and hint == "chunk_where" and eid == "chunk_where":
                    score += 2.4
                elif general_access and hint == "_access" and eid.endswith("_access"):
                    score += 0.4
                else:
                    score += 1.6
                break
    for value in method_values:
        prefix = METHOD_CHUNK_PREFIX.get(value)
        if prefix and eid.startswith(prefix):
            score += 1.2
            aligned = any(h in eid for h in (INTENT_CHUNK_HINTS.get(intent) or ())) if intent else True
            if intent and not aligned:
                score -= 0.35
            break
    for cond in conditions or []:
        hints = CONDITION_CHUNK.get(cond) or ()
        if hints and eid == hints[0]:
            score += 2.2
    if eid == "chunk_methods":
        list_ask = intent == "family_planning_methods" or any(
            c in (query or "").lower()
            for c in ("ዘዴዎች", "አማራጮች", "what methods", "method options", "ምንድን ናቸው")
        )
        if list_ask and not (conditions or []) and intent not in {
            "eligibility",
            "access",
            "definition",
            "side_effects",
            "missed_dose",
        }:
            score += 1.5
        elif conditions or intent in {"eligibility", "access", "definition", "side_effects"} or method_values:
            # Condition/eligibility/access/named-method asks should not collapse to catalogue.
            score -= 2.5
    return score

