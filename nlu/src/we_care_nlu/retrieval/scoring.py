"""Rank and filter retrieval candidates for answering.

Public API: relevance_boost, should_force_include, answer_quality, relevant_enough.
"""

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
from we_care_nlu.retrieval.chunk_signals import (
    ACCESS_SERVICE_CUES,
    BF_CUES,
    CLINICAL_CONDITION_INTENTS,
    CONDITION_CHUNK,
    FP_SCOPED_INTENTS,
    HIV_THERAPY_CUES,
    INTENT_CHUNK_HINTS,
    LAM_CUES,
    METHOD_CHUNK_PREFIX,
    METHOD_LIST_CUES,
    _OFF_TOPIC_COINFECTION,
    bf_hit,
    blob_of,
    curated_aligned_with_query,
    curated_intent_match,
    entity_condition_values,
    entity_method_values,
    has_breastfeeding,
    intent_needle_hit,
    is_demo_chunk,
    method_hit,
    _hiv_therapy_hit,
    _condition_lexical_hit,
)

_LOW_QUALITY = re.compile(
    r"("
    r"https?://|doi\.org|\[\d+\]|"
    r"የተቀናጀ የሆርሞን|CHC፡|COC፡|Cu-IUD፡|"  # glossary / acronym dumps
    r"ጨምሮ።?\s*$"  # TOC teaser ending
    r")",
    re.I,
)

_SPACED_LATIN = re.compile(r"(?:[A-Za-z]\s+){3,}[A-Za-z]")

def low_quality_penalty(text: str) -> float:
    """Negative score for bibliography, glossary dumps, TOC teasers, OCR junk."""
    body = (text or "").strip()
    if not body:
        return -5.0
    pen = 0.0
    if _LOW_QUALITY.search(body):
        pen -= 1.2
    if _SPACED_LATIN.search(body):
        pen -= 1.5
    # Long citation-like lines with many commas/semicolons and little answer shape.
    if body.count("፤") + body.count(";") >= 4 and len(body) > 180:
        pen -= 0.8
    if body.count("https://") + body.count("doi") >= 1:
        pen -= 1.0
    named = sum(1 for cue in METHOD_LIST_CUES if cue.lower() in body.lower())
    if "ጨምሮ" in body and named < 3:
        pen -= 1.5
    if "፦" in body and named < 3 and len(body) < 220:
        pen -= 1.0
    # Dangling list intros ("the following criteria…") with no criteria attached.
    if any(x in body for x in ("የሚከተሉት", "እንደሚከተለው", "እንደሚከተሉት")) and not re.search(
        r"\d", body
    ):
        pen -= 2.2
    # Click-through / teaser only ("click the link…") — not an answer.
    if any(x in body for x in ("ማስፈንጠሪያ", "ይጫኑ", "click here", "read more")) and len(body) < 160:
        pen -= 2.5
    return pen

def relevance_boost(
    *,
    query: str,
    blob: str,
    intent: str | None,
    entities: Iterable[Any] | None,
    evidence_id: str = "",
    demo: bool = False,
) -> float:
    """Additive fused-score boost used at retrieval time."""
    boost = 0.0
    q_bf = has_breastfeeding(query, entities)
    m_hit = method_hit(blob, entities)
    b_hit = bf_hit(blob)
    methods = entity_method_values(entities)
    conditions = entity_condition_values(entities)
    eid = evidence_id or ""
    dist = distinctive_query_tokens(query)
    overlap = distinctive_overlap(blob, dist)

    # Strong lexical preference for distinctive topic tokens (beats curated overboost).
    if overlap:
        boost += 1.35 * overlap if not methods else 0.45 * overlap
    elif dist and not methods and not q_bf and not conditions:
        # Mild only — do not erase BM25 paraphrase hits.
        boost -= 0.25

    if intent_needle_hit(blob, intent):
        boost += 0.45 if intent == "missed_dose" else 0.15
    if m_hit and methods:
        boost += 0.22
    if q_bf:
        if b_hit and m_hit:
            boost += 0.65
        elif b_hit:
            boost += 0.2
        if methods and any(cue in blob for cue in LAM_CUES) and not m_hit:
            boost -= 0.75
    if demo:
        aligned = curated_aligned_with_query(query=query, blob=blob, entities=entities)
        if aligned:
            boost += 0.2
            boost += curated_intent_match(
                eid, intent, methods, conditions=conditions, query=query
            )
            if q_bf and eid.startswith("chunk_bf"):
                boost += 0.55 if (m_hit or not methods) else 0.25
            if conditions and eid.startswith("chunk_hiv") and "hiv" in conditions:
                if intent in FP_SCOPED_INTENTS or intent is None:
                    boost += 0.9
                elif intent in CLINICAL_CONDITION_INTENTS:
                    # FP+HIV eligibility text is the wrong answer for treatment/symptoms.
                    boost -= 1.6
            if intent == "access" and not methods and eid == "chunk_where":
                boost += 0.85
            if intent == "access" and not methods and eid.endswith("_access"):
                boost -= 0.35
        elif dist:
            # Curated FP defs must not outrank docs that share distinctive query terms.
            boost -= 0.85
            if intent and any(h in eid for h in (INTENT_CHUNK_HINTS.get(intent) or ())):
                boost -= 0.45
    # Condition lexical hit (HIV cues in blob). Stronger when intent-aligned.
    for cond in conditions:
        if _condition_lexical_hit(blob, cond):
            boost += 0.55 if intent not in CLINICAL_CONDITION_INTENTS else 0.9
            break
        elif intent in CLINICAL_CONDITION_INTENTS:
            # "እንዴት ይታከማል?" FAQs about unrelated conditions.
            boost -= 1.1
    if intent in CLINICAL_CONDITION_INTENTS and intent_needle_hit(blob, intent):
        boost += 0.55
    if "hiv" in conditions and intent in {"treatment", "medication"}:
        if _hiv_therapy_hit(blob):
            boost += 1.35
        q_low = (query or "").lower()
        if any(x in blob for x in _OFF_TOPIC_COINFECTION) and not any(
            x in q_low for x in _OFF_TOPIC_COINFECTION
        ):
            boost -= 1.0
    if intent == "family_planning_methods" and not conditions:
        named = sum(1 for cue in METHOD_LIST_CUES if cue.lower() in blob)
        if named >= 5:
            boost += 0.85
        elif named >= 4:
            boost += 0.65
        elif named >= 3:
            boost += 0.4
        if demo and named >= 3:
            boost += 0.35
        if "ጨምሮ" in blob and named < 3:
            boost -= 0.55
        if "፦" in blob and named < 3:
            boost -= 0.35
    boost += low_quality_penalty(blob)
    return boost

def should_force_include(
    *,
    query: str,
    chunk: Any,
    intent: str | None,
    entities: Iterable[Any] | None,
) -> float:
    """Priority > 0 means merge this chunk into candidates even if BM25 missed it."""
    blob = blob_of(chunk)
    eid = str(getattr(chunk, "evidence_id", "") or "")
    demo = is_demo_chunk(chunk)
    methods = entity_method_values(entities)
    q_bf = has_breastfeeding(query, entities)
    pri = 0.0

    list_ask = intent == "family_planning_methods" or any(
        c in (query or "").lower()
        for c in ("ዘዴዎች ምን", "ምንድን ናቸው", "አማራጮች", "what methods", "method options")
    )
    if list_ask:
        named = sum(1 for cue in METHOD_LIST_CUES if cue.lower() in blob)
        if named >= 4 or (demo and named >= 3) or eid == "chunk_methods":
            pri = max(pri, named + (3.0 if eid == "chunk_methods" else 0.0) + (2.0 if demo else 0.0))

    conditions = entity_condition_values(entities)
    if demo and methods:
        for value in methods:
            prefix = METHOD_CHUNK_PREFIX.get(value)
            if prefix and eid.startswith(prefix):
                # Prefer intent-aligned curated slice for that method.
                pri = max(
                    pri,
                    4.0
                    + curated_intent_match(
                        eid, intent, methods, conditions=conditions, query=query
                    ),
                )
                if intent and any(h in eid for h in (INTENT_CHUNK_HINTS.get(intent) or ())):
                    pri += 2.0
                break

    if q_bf:
        if bf_hit(blob):
            m_ok = method_hit(blob, entities)
            if demo and (eid.startswith("chunk_bf") or m_ok):
                pri = max(pri, 5.0 if m_ok else 3.0)
            elif m_ok and intent_needle_hit(blob, intent or "eligibility"):
                pri = max(pri, 2.5)

    # HIV / condition: curated FP+HIV only for FP-scoped intents.
    for cond in conditions:
        hints = CONDITION_CHUNK.get(cond) or ()
        if demo and hints and eid == hints[0]:
            if intent in FP_SCOPED_INTENTS or intent is None:
                pri = max(pri, 6.0)
            # else: do not force-include FP eligibility for treatment/symptoms
        elif _condition_lexical_hit(blob, cond):
            if intent in CLINICAL_CONDITION_INTENTS and (
                intent_needle_hit(blob, intent) or (cond == "hiv" and _hiv_therapy_hit(blob))
            ):
                pri = max(pri, 5.5 if (cond == "hiv" and _hiv_therapy_hit(blob)) else 3.2)
            elif intent not in CLINICAL_CONDITION_INTENTS:
                pri = max(pri, 2.8)

    if intent == "access" and not methods and demo and eid == "chunk_where":
        pri = max(pri, 5.5)

    # Intent-hinted curated chunks only when on-topic (method match or distinctive tokens).
    # Do NOT inject every *_def for a definition ask about cryptic pregnancy, etc.
    # General access asks only use chunk_where (handled above), not every *_access.
    if demo and intent and intent != "access" and any(
        h in eid for h in (INTENT_CHUNK_HINTS.get(intent) or ())
    ):
        if curated_aligned_with_query(query=query, blob=blob, entities=entities):
            pri = max(pri, 3.5)

    # Docs that fully ground distinctive + soft-domain tokens (e.g. period pain → NSAIDs).
    dist = distinctive_query_tokens(query)
    soft = soft_topic_groups(query)
    if (
        dist
        and soft
        and soft_topic_hit(blob, soft, query=query)
        and not unspecified_distractor(query, blob)
        and distinctive_coverage(blob, dist) >= 0.99
    ):
        pri = max(pri, 4.5)

    return pri

def answer_quality(
    text: str,
    *,
    intent: str | None,
    query: str,
    entities: Iterable[Any] | None,
    evidence_id: str = "",
    fused: float = 0.0,
) -> float:
    """Higher is better for final reply selection."""
    body = (text or "").strip()
    if not body:
        return -20.0
    score = min(len(body), 320) / 120.0  # mild length preference, not dominant
    score += fused * 0.35
    score += low_quality_penalty(body)
    if body.endswith(("፣", ",", "እና", "and", "or", "ወይም")):
        score -= 3.0

    blob = body.lower()
    methods = entity_method_values(entities)
    m_hit = method_hit(blob, entities)
    q_bf = has_breastfeeding(query, entities)
    b_hit = bf_hit(blob)

    conditions = entity_condition_values(entities)
    if methods:
        # HIV curated answer covers methods generally (does not list every method name).
        if evidence_id == "chunk_hiv_fp" and "hiv" in conditions and intent in FP_SCOPED_INTENTS:
            score += 3.0
        elif evidence_id == "chunk_hiv_fp" and intent in CLINICAL_CONDITION_INTENTS:
            score -= 10.0
        else:
            score += 5.0 if m_hit else -6.0
    if q_bf:
        if b_hit and m_hit:
            score += 6.0
        elif b_hit:
            score += 1.5
        else:
            score -= 4.0
        if methods and any(cue in blob for cue in LAM_CUES) and not m_hit:
            score -= 8.0
    for cond in conditions:
        hints = CONDITION_CHUNK.get(cond) or ()
        if not hints:
            continue
        curated_fp = evidence_id == hints[0]
        lexical = _condition_lexical_hit(blob, cond)
        if intent in CLINICAL_CONDITION_INTENTS:
            if curated_fp:
                score -= 10.0
            elif lexical:
                score += 4.0
            else:
                score -= 8.0
        elif curated_fp or lexical:
            score += 6.0
        else:
            score -= 3.0

    if intent_needle_hit(blob, intent):
        score += 2.5
    # Treatment/medication asks: prefer clinical therapy wording over FP catalogues.
    if intent in {"treatment", "medication"}:
        fp_named = sum(1 for cue in METHOD_LIST_CUES if cue.lower() in blob)
        if fp_named >= 2 and not intent_needle_hit(blob, intent):
            score -= 5.0
        if evidence_id == "chunk_hiv_fp":
            score -= 8.0
        # Bare FAQ titles ("…እንዴት ይታከማል?") are not answers.
        stripped = body.strip()
        if stripped.endswith(("?", "؟")) and len(stripped) < 80:
            score -= 6.0
        if "hiv" in conditions:
            if _hiv_therapy_hit(blob):
                score += 6.0
            q_low = (query or "").lower()
            if any(x in blob for x in _OFF_TOPIC_COINFECTION) and not any(
                x in q_low for x in _OFF_TOPIC_COINFECTION
            ):
                score -= 7.0
    named = sum(1 for cue in METHOD_LIST_CUES if cue.lower() in blob)
    if intent == "family_planning_methods" and not conditions:
        score += named * 1.35
        if named < 3:
            score -= 4.0
        if "የተለያዩ አማራጮች" in body and named < 4:
            score -= 2.0

    dist = distinctive_query_tokens(query)
    overlap = distinctive_overlap(blob, dist)
    coverage = distinctive_coverage(blob, dist)
    soft_groups = soft_topic_groups(query)
    soft_ok = soft_topic_hit(blob, soft_groups, query=query)
    if soft_groups and not soft_ok:
        score -= 8.0
    if unspecified_distractor(query, blob):
        score -= 10.0
    # When a method/condition entity scopes the ask, entity alignment dominates;
    # otherwise require most distinctive tokens (not just shared broad topic words).
    if dist and not methods and not conditions:
        if soft_ok and coverage >= 0.99:
            score += 5.0 * overlap
        elif soft_ok and coverage >= 0.5:
            score += 2.5 * overlap
        else:
            score -= 9.0
    elif dist and overlap and soft_ok:
        score += 1.2 * overlap

    eid = evidence_id or ""
    curated_ok = curated_aligned_with_query(query=query, blob=blob, entities=entities)
    if eid.startswith("chunk_"):
        if curated_ok:
            score += 3.0 + curated_intent_match(
                eid, intent, methods, conditions=conditions, query=query
            ) * 1.5
        else:
            score -= 4.0
    if eid == "chunk_methods" and methods:
        score -= 5.0
    # Definition / how / SE asks: prefer definition-shaped body, not FAQ titles.
    if intent == "definition":
        if any(c in blob for c in ("ነው።", "is a", "is an", "ማለት", "ይባላል", "የሚታወቅ", "refers to")):
            score += 2.5
        stripped = body.strip()
        if stripped.endswith(("?", "؟")) or stripped.endswith("ይቻላል?") or "እንዴት" in stripped[:40]:
            score -= 5.0
        if len(stripped) < 60:
            score -= 2.0
    if intent == "access":
        if any(c in blob for c in ACCESS_SERVICE_CUES) and not methods:
            score += 3.5 if eid == "chunk_where" else 1.5
        elif any(c in blob for c in ("ይገኛል", "available", "pharmacy", "health facilit")):
            score += 2.0 if methods else 0.5
    if intent == "side_effects" and any(c in blob for c in ("ጎንዮሽ", "side effect", "bleeding", "nausea")):
        score += 2.0
    return score

def relevant_enough(
    text: str,
    *,
    query: str,
    intent: str | None,
    entities: Iterable[Any] | None,
    need_tokens: list[str] | None = None,
) -> bool:
    """Whether a candidate may be considered for the final answer."""
    blob = (text or "").lower()
    if not blob.strip():
        return False
    methods = entity_method_values(entities)
    q_bf = has_breastfeeding(query, entities)
    m_hit = method_hit(blob, entities)
    b_hit = bf_hit(blob)

    conditions = entity_condition_values(entities)
    if q_bf and methods:
        return m_hit and b_hit
    if conditions:
        cond_ok = any(_condition_lexical_hit(blob, cond) for cond in conditions)
        if not cond_ok:
            if methods:
                return m_hit
            # Allow curated FP+HIV id match via evidence path handled elsewhere; blob must name condition.
            return False
        if intent in CLINICAL_CONDITION_INTENTS:
            if intent in {"treatment", "medication"} and "hiv" in conditions:
                # Require ART / HIV-therapy wording — not HSV antivirals or FP eligibility.
                return _hiv_therapy_hit(blob)
            return intent_needle_hit(blob, intent) or intent == "definition"
        # Condition-curated / FP eligibility spans may omit the specific method name.
        return True
    if methods:
        return m_hit
    if intent == "access" and not methods:
        return any(c in blob for c in ACCESS_SERVICE_CUES) or "ይገኛል" in blob or "available" in blob
    if intent == "family_planning_methods":
        return True
    # Definition asks need an answer sentence, not a FAQ title like "…እንዴት … ይቻላል?"
    if intent == "definition":
        stripped = blob.strip()
        if stripped.endswith(("?", "؟")) or stripped.endswith("ይቻላል?"):
            return False
    # Distinctive topic terms (ክሪፕቲክ, ሲያመኝ, …) must be mostly grounded — not just ወር አበባ.
    dist = distinctive_query_tokens(query)
    soft_groups = soft_topic_groups(query)
    if soft_groups and not soft_topic_hit(blob, soft_groups, query=query):
        return False
    if unspecified_distractor(query, blob):
        return False
    if dist:
        need_frac = 1.0 if len(dist) <= 2 else 0.5
        return distinctive_coverage(blob, dist) >= need_frac
    need = need_tokens or []
    if need and any(tok.lower() in blob for tok in need):
        return True
    if intent_needle_hit(blob, intent):
        return True
    # Soft Amharic fallback: shared allow/access stems.
    soft = ("መጠቀም", "ይቻላል", "ይገኛል", "can be used", "available", "side effect", "ጎንዮሽ")
    return any(s in blob for s in soft)

