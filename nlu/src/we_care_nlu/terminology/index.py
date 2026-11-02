"""Persistent medical concept index (candidates vs approved/seeded)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from we_care_nlu.config import package_root
from we_care_nlu.terminology.extract import extract_candidates_from_chunks, seed_from_gazetteer
from we_care_nlu.terminology.models import MedicalConcept

# Re-export for callers that import MedicalConcept from index.
__all__ = ["ConceptIndex", "MedicalConcept"]


def default_terminology_dir() -> Path:
    return package_root() / "data" / "terminology"


class ConceptIndex:
    """Approved/seeded concepts used at query time; candidates stay review-gated."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else default_terminology_dir()
        self.candidates_path = self.root / "candidates.jsonl"
        self.approved_path = self.root / "approved.json"
        self.concepts: dict[str, MedicalConcept] = {}
        self._surface_map: list[tuple[str, MedicalConcept]] = []

    def load(self) -> "ConceptIndex":
        self.concepts.clear()
        if self.approved_path.exists():
            raw = json.loads(self.approved_path.read_text(encoding="utf-8"))
            rows = raw.get("concepts") if isinstance(raw, dict) else raw
            for row in rows or []:
                c = MedicalConcept.model_validate(row)
                if c.review_status in {"approved", "seeded"}:
                    self.concepts[c.concept_id] = c
        self._rebuild_surfaces()
        return self

    def _rebuild_surfaces(self) -> None:
        entries: list[tuple[str, MedicalConcept]] = []
        for c in self.concepts.values():
            for s in c.all_surfaces():
                entries.append((s, c))
        entries.sort(key=lambda x: len(x[0]), reverse=True)
        self._surface_map = entries

    def save_approved(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "concepts": [c.model_dump() for c in sorted(self.concepts.values(), key=lambda x: x.concept_id)],
        }
        self.approved_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_candidates(self, candidates: Iterable[MedicalConcept]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.candidates_path.open("w", encoding="utf-8") as fh:
            for c in candidates:
                fh.write(c.model_dump_json() + "\n")
        return self.candidates_path

    def load_candidates(self) -> list[MedicalConcept]:
        if not self.candidates_path.exists():
            return []
        out: list[MedicalConcept] = []
        for line in self.candidates_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(MedicalConcept.model_validate(json.loads(line)))
        return out

    def upsert_seeded(self, concepts: Iterable[MedicalConcept]) -> int:
        n = 0
        for c in concepts:
            if c.review_status not in {"approved", "seeded"}:
                c.review_status = "seeded"
            self.concepts[c.concept_id] = c
            n += 1
        self._rebuild_surfaces()
        return n

    def promote_candidate(self, concept_id: str) -> MedicalConcept | None:
        for c in self.load_candidates():
            if c.concept_id == concept_id:
                c.review_status = "approved"
                self.concepts[c.concept_id] = c
                self._rebuild_surfaces()
                self.save_approved()
                return c
        return None

    def ensure_seeded_from_gazetteer(self, gazetteer: dict[str, Any] | None = None) -> int:
        if gazetteer is None:
            from we_care_nlu.ner.extract import load_gazetteer

            gazetteer = load_gazetteer()
        seeded = seed_from_gazetteer(gazetteer)
        n = self.upsert_seeded(seeded)
        self.save_approved()
        return n

    def rebuild_candidates_from_chunks(self, chunks: Iterable[Any]) -> list[MedicalConcept]:
        candidates = extract_candidates_from_chunks(chunks)
        # Drop candidates that already match an approved surface (avoid noise).
        approved_surfaces = {s.lower() for s, _ in self._surface_map}
        filtered: list[MedicalConcept] = []
        for c in candidates:
            surfs = [x.lower() for x in c.all_surfaces()]
            if any(s in approved_surfaces for s in surfs):
                continue
            filtered.append(c)
        self.save_candidates(filtered)
        return filtered

    def match(self, text: str, *, limit: int = 8) -> list[MedicalConcept]:
        from we_care_nlu.terminology.match import match_concepts_in_text

        return match_concepts_in_text(text, self._surface_map, limit=limit)

    def expansion_terms(self, text: str, *, limit: int = 6) -> list[str]:
        """Approved synonyms only — never pending candidates."""
        terms: list[str] = []
        seen: set[str] = set()
        for c in self.match(text, limit=limit):
            for s in c.all_surfaces():
                key = s.lower()
                if key in seen:
                    continue
                seen.add(key)
                terms.append(s)
                if len(terms) >= 24:
                    return terms
        return terms

    def stats(self) -> dict[str, Any]:
        by_cat: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for c in self.concepts.values():
            by_cat[c.category] = by_cat.get(c.category, 0) + 1
            by_status[c.review_status] = by_status.get(c.review_status, 0) + 1
        cands = self.load_candidates()
        return {
            "approved_or_seeded": len(self.concepts),
            "candidates_pending": len(cands),
            "by_category": by_cat,
            "by_status": by_status,
            "approved_path": str(self.approved_path),
            "candidates_path": str(self.candidates_path),
        }
