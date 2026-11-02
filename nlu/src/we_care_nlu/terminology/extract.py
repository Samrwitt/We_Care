"""Extract *candidate* medical terms from indexed KB chunks.

Candidates stay pending_review until promoted. English–Amharic co-occurrence is a
hint, not a proven translation.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any, Iterable

from we_care_nlu.contracts.schemas import ChunkRecord
from we_care_nlu.terminology.models import MedicalConcept

_ETH = re.compile(r"[\u1200-\u137F]")
_LATIN_ACRONYM = re.compile(r"\b([A-Z][A-Z0-9]{1,7})\b")
_PAREN_EN = re.compile(
    r"([\u1200-\u137F][\u1200-\u137F\s\.\-]{1,40}?)\s*[\(（]\s*([A-Za-z][A-Za-z0-9\s\-\./]{1,60}?)\s*[\)）]"
)
_TITLE_NOISE = re.compile(
    r"(edited|copy of|untitled|✓|፡|።|\(\d+\)|\s+\d+\s*$)",
    re.I,
)

_STOP_AM = {
    "ምንድን", "ነው", "ናቸው", "እንዴት", "ለምን", "መቼ", "የት", "ስለ", "እና", "ወይም",
    "ይህ", "ያ", "ላይ", "ውስጥ", "ጋር", "አለ", "ይችላል", "ይቻላል", "መረጃ",
}

_CATEGORY_CUES: list[tuple[str, tuple[str, ...]]] = [
    ("contraceptive_method", ("መከላከያ", "ኮንዶም", "ኢምፕላንት", "መርፌ", "ክኒን", "አይዩ", "iud", "implant", "condom", "pill")),
    ("medical_condition", ("በሽታ", "ካንሰር", "ኤች", "hiv", "sti", "እርግዝና", "infection", "cancer", "ሲንድረም")),
    ("symptom", ("ምልክት", "ህመም", "ሕመም", "መድማት", "symptom", "pain", "bleeding")),
    ("medicine", ("መድሃኒት", "መድኃኒት", "ክትባት", "drug", "vaccine", "tablet")),
    ("procedure", ("ቀዶ", "ምርመራ", "አሰራር", "surgery", "screening", "ultrasound")),
    ("anatomy", ("ማህጸን", "ማኅጸን", "ጡት", "uterus", "cervix", "ovary")),
    ("service", ("አገልግሎት", "ጤና ኬላ", "clinic", "service", "counselling")),
]


def _cid(*parts: str) -> str:
    raw = "|".join(p.strip().lower() for p in parts if p and p.strip())
    return "concept_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _guess_category(blob: str) -> str:
    low = (blob or "").lower()
    for cat, cues in _CATEGORY_CUES:
        if any(c.lower() in low for c in cues):
            return cat
    return "general"


def _clean_title(title: str) -> str:
    t = _TITLE_NOISE.sub(" ", title or "")
    t = re.sub(r"\s+", " ", t).strip(" -_·.")
    return t


def _is_phrase_candidate(label: str) -> bool:
    """Reject sentence-like headings; keep short concept-like phrases."""
    lab = (label or "").strip().rstrip("?؟።")
    if not lab or lab in _STOP_AM:
        return False
    if len(lab) < 3 or len(lab) > 36:
        return False
    toks = lab.split()
    if len(toks) > 4:
        return False
    # Numbered guideline steps / instructions / FAQ titles.
    if re.match(r"^[\d\.\)\(｜\|]+", lab):
        return False
    bad = (
        "ያጸዳል", "ይፍጠሩ", "መሆን አለበት", "ሪፖርት", "ሰነድ", "ምንድን", "እንዴት",
        "ማረጋገጥ", "መጠቀምን", "እድልን", "STEP", "TASK",
    )
    if any(x in lab for x in bad):
        return False
    if "።" in lab or ";" in lab:
        return False
    return True


def _merge(
    store: dict[str, MedicalConcept],
    *,
    am: str = "",
    en: str = "",
    category: str,
    doc_id: str,
    eid: str,
    status: str = "pending_review",
) -> None:
    am = (am or "").strip()
    en = (en or "").strip()
    if not am and not en:
        return
    if am and am in _STOP_AM:
        return
    if am and len(am) < 2:
        return
    if en and len(en) < 2:
        return
    key = _cid(am or en, en or am, category)
    existing = store.get(key)
    if existing is None:
        store[key] = MedicalConcept(
            concept_id=key,
            canonical_amharic=am,
            canonical_english=en,
            synonyms_amharic=[am] if am else [],
            synonyms_english=[en] if en else [],
            category=category,
            source_document_ids=[doc_id] if doc_id else [],
            source_evidence_ids=[eid] if eid else [],
            review_status=status,  # type: ignore[arg-type]
        )
        return
    if am and am not in existing.synonyms_amharic:
        existing.synonyms_amharic.append(am)
        if not existing.canonical_amharic:
            existing.canonical_amharic = am
    if en and en not in existing.synonyms_english:
        existing.synonyms_english.append(en)
        if not existing.canonical_english:
            existing.canonical_english = en
    if doc_id and doc_id not in existing.source_document_ids:
        existing.source_document_ids.append(doc_id)
    if eid and eid not in existing.source_evidence_ids:
        existing.source_evidence_ids.append(eid)


def extract_candidates_from_chunks(chunks: Iterable[ChunkRecord | Any]) -> list[MedicalConcept]:
    """Scan KB chunks → pending_review concept candidates (deduplicated)."""
    store: dict[str, MedicalConcept] = {}
    title_docs: dict[str, set[str]] = defaultdict(set)

    for chunk in chunks:
        am = str(getattr(chunk, "amharic_text", "") or "")
        en = str(getattr(chunk, "english_text", "") or "")
        title = _clean_title(str(getattr(chunk, "document_title", "") or ""))
        heading = str(getattr(chunk, "heading", "") or getattr(chunk, "section", "") or "")
        doc_id = str(getattr(chunk, "document_id", "") or "")
        eid = str(getattr(chunk, "evidence_id", "") or "")
        subject = str(getattr(chunk, "medical_subject", "") or "")
        blob = f"{title} {heading} {am[:240]} {en[:240]} {subject}"
        cat = _guess_category(blob)

        # Document titles only (headings are often mid-sentence guideline noise).
        if title and _ETH.search(title) and _is_phrase_candidate(title):
            title_docs[title].add(doc_id)
            _merge(store, am=title, category=cat, doc_id=doc_id, eid=eid)

        # Amharic (English) pairs — candidate links only (not proven translations).
        for m in _PAREN_EN.finditer(f"{title} {am[:300]}"):
            am_term, en_term = m.group(1).strip(), m.group(2).strip()
            if not (2 <= len(en_term) <= 48):
                continue
            # Prefer short Amharic left-hand sides (true term labels).
            if _ETH.search(am_term) and not _is_phrase_candidate(am_term):
                # Keep English acronym/name even if Amharic span was too long.
                am_term = ""
            if not am_term and not en_term:
                continue
            _merge(
                store,
                am=am_term if am_term and _ETH.search(am_term) else "",
                en=en_term,
                category=_guess_category(f"{am_term} {en_term} {subject}"),
                doc_id=doc_id,
                eid=eid,
            )

        # Latin medical acronyms from English / mixed text.
        for m in _LATIN_ACRONYM.finditer(f"{en} {am} {title}"):
            acr = m.group(1)
            if acr in {"THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT", "PDF", "DOC", "HTTP", "HTTPS"}:
                continue
            _merge(store, en=acr, category=_guess_category(f"{acr} {subject}"), doc_id=doc_id, eid=eid)

    return sorted(store.values(), key=lambda c: (c.category, c.canonical_amharic or c.canonical_english))


def seed_from_gazetteer(gazetteer: dict[str, Any]) -> list[MedicalConcept]:
    """Promote curated gazetteer entries as seeded (approved) concepts."""
    out: list[MedicalConcept] = []
    for typ, values in (gazetteer or {}).items():
        if not isinstance(values, dict):
            continue
        for canonical, surfaces in values.items():
            surfs = [str(s).strip() for s in (surfaces or []) if str(s).strip()]
            am = [s for s in surfs if _ETH.search(s)]
            en = [s for s in surfs if not _ETH.search(s)]
            out.append(
                MedicalConcept(
                    concept_id=_cid("gaz", typ, str(canonical)),
                    canonical_amharic=am[0] if am else "",
                    canonical_english=en[0] if en else str(canonical),
                    synonyms_amharic=am,
                    synonyms_english=en + ([str(canonical)] if str(canonical) not in en else []),
                    category=str(typ),
                    review_status="seeded",
                    notes="Seeded from data/gazetteers/entities.yaml",
                )
            )
    return out
