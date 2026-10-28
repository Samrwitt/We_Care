"""General relevance helpers for retrieval + answer picking.

Goal: any (method, condition, intent) ask should prefer short grounded answers
over long guideline dumps, TOC teasers, or distractors (e.g. LAM for injectable).
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from we_care_nlu.retrieval.expand import INTENT_NEEDLES, method_terms

# Question fillers + broad medical topic words that alone must not pick a curated FP def.
_QUERY_STOP = {
    "what", "are", "there", "this", "with", "from", "that", "have", "been",
    "does", "they", "them", "about", "while", "used", "must", "where", "when",
    "which", "your", "into", "such", "also", "their", "than", "then", "how",
    "long", "many", "much", "some", "more", "most", "only", "just", "the", "and",
    "ለምን", "ምንድን", "ምንድር", "ምንድ", "ምን", "ነው", "ናቸው", "እንዴት", "መቼ", "የት", "ማን",
    "እባክዎ", "ስለ", "ላይ", "ውስጥ", "ጋር", "ወይም", "እና", "ይህ", "ያ", "እንደ",
    "ጠቃሚ", "አስፈላጊ", "መረጃ", "ጥያቄ", "አለብኝ", "እችላለሁ", "ይቻላል", "ይገኛል",
    "የሚባለው", "ሚባለው", "የሚባል", "ሚባል",
}
_GENERIC_TOPIC = {
    "እርግዝና", "እርግዝ", "መከላከያ", "መከላከል", "ዘዴ", "ዘዴዎች", "ሴት", "ሴቶች", "ቤተሰብ", "ምጣኔ",
    "አገልግሎት", "ጤና", "አማራጭ", "አማራጮች", "አሉ", "ይቻላል",
    # Broad body/topic words — not enough alone to pick an article (need ሲያመኝ / ተዛባ / …).
    "ወር", "አበባ", "ዑደት", "የወር", "ጤናማ", "ሰው", "ሰዎች", "ሕመምተኛ",
    "pregnancy", "contraception", "contraceptive", "method",
    "methods", "family", "planning", "woman", "women", "health", "service",
    "options", "option", "period", "menstrual", "menstruation", "cycle",
}

# Light synonym / stem families so distinctive tokens can match paraphrase evidence.
_DISTINCTIVE_ALIASES: tuple[tuple[str, ...], ...] = (
    ("ሲያመኝ", "ያመኛል", "ያመማል", "አመመ", "አመም", "ህመም", "ሕመም", "ቁርጠት", "pain", "cramp", "ache"),
    ("ላድርግ", "ማድረግ", "ማስታገስ", "ያስታግሳል", "መድኃኒት", "መድሃኒት", "ibuprofen", "paracetamol"),
    ("ተዛባ", "መዛባት", "ተዛቡ", "irregular"),
    ("ክሪፕቲክ", "ስውር", "ሚስጥራዊ", "cryptic"),
)

# Soft domain anchors in the ask → answer must share the same domain (not just ሕመም/መድኃኒት).
_SOFT_TOPIC_CUES: tuple[tuple[str, ...], ...] = (
    # Keep cues specific — short "ኢንዶሜትሪ" also matches unrelated "ኢንዶሜትሪተስ" (endometritis).
    (
        "ወር አበባ",
        "የወር አበባ",
        "menstrual",
        "dysmenorrhea",
        "ዲስሜኖሬያ",
        "endometriosis",
        "ኢንዶሜትሪዮሲስ",
    ),
    ("ኤች.አይ.ቪ", "ኤችአይቪ", "hiv", "ኤድስ", "aids"),
    ("እርግዝና", "pregnant", "pregnancy", "እርጉዝ"),
)

# Specific symptoms/conditions that must not hijack a vaguer ask in the same domain.
_UNSPECIFIED_DISTRACTORS: tuple[tuple[str, ...], ...] = (
    ("ራስ ምታት", "migraine", "headache"),
    ("ቂጥኝ", "syphilis"),
    ("ጨብጥ", "gonorrhea", "gonorrh"),
    ("ክላሚዲያ", "chlamydia"),
    ("ኤችኤስቪ", "hsv", "herpes"),
    ("ፒአይዲ", "pid", "pelvic inflammatory", "ኢንዶሜትሪተስ", "endometritis"),
    ("ከማኅፀን ውጪ", "ectopic", "ሜቶትሪክሴት", "methotrexate"),
)
_AM_SUFFIXES = ("ዎች", "ዎት", "ዋ", "ው", "ን", "ም")

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

# Curated chunk_id patterns: intent → preferred evidence_id fragments.
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

_LOW_QUALITY = re.compile(
    r"("
    r"https?://|doi\.org|\[\d+\]|"
    r"የተቀናጀ የሆርሞን|CHC፡|COC፡|Cu-IUD፡|"  # glossary / acronym dumps
    r"ጨምሮ።?\s*$"  # TOC teaser ending
    r")",
    re.I,
)
_SPACED_LATIN = re.compile(r"(?:[A-Za-z]\s+){3,}[A-Za-z]")


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


CONDITION_CHUNK: dict[str, tuple[str, ...]] = {
    # First id is the curated *FP eligibility* slice — only for FP-scoped intents.
    "hiv": ("chunk_hiv_fp", "ኤች.አይ.ቪ", "ኤችአይቪ", "hiv", "aids", "ኤድስ"),
}

# Intents where chunk_hiv_fp (contraception + HIV) is an appropriate answer.
FP_SCOPED_INTENTS = {
    "eligibility",
    "family_planning_methods",
    "contraindications",
    "family_planning",
    "pregnancy_breastfeeding",
}

# Clinical asks about the condition itself — not contraception eligibility.
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

# Therapy wording that marks true HIV treatment content (vs FP eligibility / co-infection asides).
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


_AM_PREFIXES = ("የ", "በ", "ለ", "ከ", "ስለ", "ወደ", "እንደ")


def _strip_am_prefix(tok: str) -> str:
    for pref in _AM_PREFIXES:
        if tok.startswith(pref) and len(tok) - len(pref) >= 3:
            return tok[len(pref) :]
    return tok


def _normalize_am_token(tok: str) -> str:
    """Strip light clitics so እርግዝናን → እርግዝና (generic), not a false distinctive."""
    tok = _strip_am_prefix(tok)
    for suf in _AM_SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            stem = tok[: -len(suf)]
            # Only collapse when the stem is already a known filler/topic — never
            # invent short stems like የሚባለው → ሚባለ (matches every "የሚባለው" sentence).
            if stem in _GENERIC_TOPIC or stem in _QUERY_STOP:
                return stem
            if len(suf) >= 2:
                return stem
    return tok


def distinctive_query_tokens(query: str) -> list[str]:
    """Topic tokens that must appear in a grounded answer when present in the ask.

    Filters question fillers and broad words like እርግዝና/መከላከያ/አበባ so a period-pain
    ask cannot be answered by an irregular-cycle article that only shares 'ወር አበባ'.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in (query or "").lower().split():
        bare = raw.strip("?.!,;:፡።፣፤\"'()[]")
        if len(bare) < 3 or bare in _QUERY_STOP or bare in _GENERIC_TOPIC:
            continue
        tok = _normalize_am_token(bare)
        if len(tok) < 3 or tok in _QUERY_STOP or tok in _GENERIC_TOPIC:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def _distinctive_aliases(tok: str) -> list[str]:
    """Return tok plus stems/synonyms for soft Amharic paraphrase matching."""
    t = (tok or "").strip().lower()
    if not t:
        return []
    aliases = [t]
    for pref in ("ሲ", "ያ", "የ", "ት", "እንዲ", "በ"):
        if t.startswith(pref) and len(t) - len(pref) >= 3:
            aliases.append(t[len(pref) :])
    for group in _DISTINCTIVE_ALIASES:
        gl = tuple(x.lower() for x in group)
        if any(a in gl for a in aliases):
            aliases.extend(gl)
            break
    # unique, stable
    seen: set[str] = set()
    out: list[str] = []
    for a in aliases:
        if a and a not in seen:
            seen.add(a)
            out.append(a)
    return out


def distinctive_overlap(blob: str, tokens: list[str] | None) -> int:
    if not tokens:
        return 0
    low = (blob or "").lower()
    n = 0
    for t in tokens:
        if any(a and a in low for a in _distinctive_aliases(t)):
            n += 1
    return n


def distinctive_coverage(blob: str, tokens: list[str] | None) -> float:
    """Fraction of distinctive query tokens grounded in the candidate (0–1)."""
    if not tokens:
        return 1.0
    return distinctive_overlap(blob, tokens) / float(len(tokens))


def soft_topic_groups(query: str) -> list[tuple[str, ...]]:
    """Domain cue groups mentioned in the query (period, HIV, pregnancy, …)."""
    q = (query or "").lower()
    groups: list[tuple[str, ...]] = []
    for cues in _SOFT_TOPIC_CUES:
        if any(c.lower() in q for c in cues):
            groups.append(cues)
    return groups


def soft_topic_hit(
    blob: str,
    groups: list[tuple[str, ...]] | None,
    *,
    query: str = "",
) -> bool:
    """Require the domain the user actually named (e.g. ወር አበባ), not a cousin disease."""
    if not groups:
        return True
    low = (blob or "").lower()
    q = (query or "").lower()
    for group in groups:
        # Cues the user uttered are required in the answer when present.
        primaries = [c for c in group if c.lower() in q]
        if primaries:
            if any(p.lower() in low for p in primaries):
                return True
            continue
        if any(c.lower() in low for c in group):
            return True
    return False


def unspecified_distractor(query: str, blob: str) -> bool:
    """True when blob focuses on a specific illness the user did not name."""
    q = (query or "").lower()
    b = (blob or "").lower()
    for cues in _UNSPECIFIED_DISTRACTORS:
        if any(c.lower() in b for c in cues) and not any(c.lower() in q for c in cues):
            return True
    return False


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
