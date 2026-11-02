"""Shared concept record shapes (JSON-serializable)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ReviewStatus = Literal["pending_review", "approved", "rejected", "seeded"]


class MedicalConcept(BaseModel):
    concept_id: str
    canonical_amharic: str = ""
    canonical_english: str = ""
    synonyms_amharic: list[str] = Field(default_factory=list)
    synonyms_english: list[str] = Field(default_factory=list)
    category: str = "general"
    source_document_ids: list[str] = Field(default_factory=list)
    source_evidence_ids: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = "pending_review"
    notes: str = ""

    def all_surfaces(self) -> list[str]:
        out: list[str] = []
        for s in (
            [self.canonical_amharic, self.canonical_english]
            + list(self.synonyms_amharic)
            + list(self.synonyms_english)
        ):
            t = (s or "").strip()
            if t and t not in out:
                out.append(t)
        return out

    def to_public(self) -> dict[str, Any]:
        return self.model_dump()
