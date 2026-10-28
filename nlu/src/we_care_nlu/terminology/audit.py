"""KB coverage audit — never silently ignore failed / empty sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from we_care_nlu.config import package_root
from we_care_nlu.kb.ingest import ingest_path
from we_care_nlu.terminology.index import ConceptIndex


def discover_kb_files(kb_dir: Path | None = None) -> list[Path]:
    root = Path(kb_dir) if kb_dir else package_root() / "data" / "kb"
    if not root.exists():
        return []
    return sorted(p for p in root.glob("*.json") if p.is_file())


def audit_kb(
    *,
    kb_dir: Path | None = None,
    concept_index: ConceptIndex | None = None,
) -> dict[str, Any]:
    files = discover_kb_files(kb_dir)
    report: dict[str, Any] = {
        "kb_dir": str(kb_dir or package_root() / "data" / "kb"),
        "documents_discovered": len(files),
        "documents_ingested": 0,
        "documents_failed": [],
        "documents_empty": [],
        "total_chunks": 0,
        "duplicate_evidence_ids": 0,
        "empty_chunks": 0,
        "missing_title": 0,
        "unreviewed_chunks": 0,
        "source_chunks": 0,
        "by_file": [],
        "terminology": {},
    }
    all_eids: set[str] = set()
    dup = 0
    chunks_all: list[Any] = []

    for path in files:
        entry: dict[str, Any] = {"path": str(path), "ok": False, "chunks": 0, "error": None}
        try:
            chunks = ingest_path(str(path), default_review_status="unreviewed")
            entry["ok"] = True
            entry["chunks"] = len(chunks)
            if not chunks:
                report["documents_empty"].append(str(path))
            else:
                report["documents_ingested"] += 1
            for c in chunks:
                chunks_all.append(c)
                eid = c.evidence_id
                if eid in all_eids:
                    dup += 1
                all_eids.add(eid)
                if not (c.amharic_text or c.english_text):
                    report["empty_chunks"] += 1
                if not (c.document_title or "").strip():
                    report["missing_title"] += 1
                if c.review_status == "unreviewed":
                    report["unreviewed_chunks"] += 1
                if c.review_status in {"source", "approved", "reviewed"}:
                    report["source_chunks"] += 1
        except Exception as exc:  # noqa: BLE001 — audit must record failures
            entry["error"] = f"{exc.__class__.__name__}: {exc}"
            report["documents_failed"].append(entry)
        report["by_file"].append(entry)

    report["total_chunks"] = len(chunks_all)
    report["duplicate_evidence_ids"] = dup

    idx = concept_index or ConceptIndex().load()
    if not idx.concepts:
        idx.ensure_seeded_from_gazetteer()
    candidates = idx.rebuild_candidates_from_chunks(chunks_all) if chunks_all else []
    report["terminology"] = {
        **idx.stats(),
        "candidates_extracted_this_run": len(candidates),
    }
    return report


def write_audit_report(path: Path, report: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
