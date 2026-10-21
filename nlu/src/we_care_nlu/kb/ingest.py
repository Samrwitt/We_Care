"""Ingest Component 1 alignment JSON into knowledge chunks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from we_care_nlu.contracts.schemas import ChunkRecord
from we_care_nlu.preprocess.normalize import preprocess


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _subject(text: str) -> str:
    low = text.lower()
    mapping = [
        ("contracept", "family_planning"),
        ("የእርግዝና መከላከያ", "family_planning"),
        ("የቤተሰብ ምጣኔ", "family_planning"),
        ("hiv", "hiv"),
        ("ኤች.አይ.ቪ", "hiv"),
        ("pregnan", "maternal"),
        ("እርግዝና", "maternal"),
        ("breast", "maternal"),
        ("ጡት", "maternal"),
        ("sti", "sti"),
        ("contraindicat", "safety"),
        ("ተቃራኒ", "safety"),
        ("side effect", "safety"),
        ("የጎንዮሽ", "safety"),
    ]
    for needle, cat in mapping:
        if needle in low:
            return cat
    return "general"


def ingest_alignment(
    payload: dict[str, Any],
    *,
    document_id: str | None = None,
    document_version: str = "unknown",
    default_review_status: str = "unreviewed",
) -> list[ChunkRecord]:
    doc_name = payload.get("document_name") or "unknown.pdf"
    doc_id = document_id or Path(str(doc_name)).stem
    title = payload.get("title") or doc_id
    chunks: list[ChunkRecord] = []
    seen: set[str] = set()
    current_heading = ""
    for row in payload.get("segments") or []:
        kind = row.get("kind") or "paragraph"
        en = (row.get("source_text") or "").strip()
        am = (row.get("translated_text") or "").strip()
        if kind == "heading":
            current_heading = am or en
        if not en and not am:
            continue
        if kind in {"header", "footer", "page_number"}:
            continue
        key = _hash(f"{en}\n{am}")
        if key in seen:
            continue
        seen.add(key)
        flags = (row.get("flags") or "")
        status = default_review_status
        if "human_reviewed" in flags:
            status = "reviewed"
        page = row.get("page_number")
        try:
            page_i = int(page) if page not in (None, "") else None
        except (TypeError, ValueError):
            page_i = None
        text_for_terms = f"{en} {am}"
        terms = []
        for token in ("IUCD", "HIV", "DMPA", "PMTCT", "VIA", "FP"):
            if token in text_for_terms:
                terms.append(token)
        rec = ChunkRecord(
            evidence_id=row.get("segment_id") or f"{doc_id}:{key}",
            document_id=doc_id,
            document_title=title,
            section=current_heading,
            heading=current_heading,
            page=page_i,
            english_text=en,
            amharic_text=am,
            review_status=status,
            medical_subject=_subject(text_for_terms),
            document_version=document_version,
            source_kind="phase1_alignment",
            terms=terms,
            content_hash=key,
        )
        chunks.append(rec)
    return chunks


def ingest_path(path: str | Path, **kwargs) -> list[ChunkRecord]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ingest_alignment(data, **kwargs)


def tokenize(text: str) -> list[str]:
    pre = preprocess(text)
    raw = [
        t
        for t in pre.normalized.replace(".", " ").replace(",", " ").replace("?", " ").replace("!", " ").split()
        if t
    ]
    stripped: list[str] = []
    for tok in raw:
        if any("\u1200" <= ch <= "\u137F" for ch in tok) and len(tok) >= 5 and tok.endswith("ን"):
            stripped.append(tok[:-1])
    raw = raw + stripped
    extra: list[str] = []
    for tok in raw:
        if any("\u1200" <= ch <= "\u137F" for ch in tok) and len(tok) >= 4:
            extra.append(tok[:4])
            if len(tok) >= 5:
                extra.append(tok[:5])
    return raw + extra
