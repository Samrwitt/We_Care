"""Stable data contracts for Components 1, 3, 4, and 5."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ContextEntity(BaseModel):
    type: str
    value: str
    surface: str | None = None


class ContextState(BaseModel):
    """Supplied by Component 3. This engine does not invent missing slots."""

    turn_index: int = 0
    last_intents: list[str] = Field(default_factory=list)
    last_entities: list[ContextEntity] = Field(default_factory=list)
    last_topic: str | None = None
    last_method: str | None = None
    last_condition: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class AnalyzeRequest(BaseModel):
    text: str
    language: Literal["am", "en", "mixed"] | None = None
    conversation_id: str | None = None
    context: ContextState | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int | None = None
    model_key: str | None = None


class Entity(BaseModel):
    type: str
    value: str
    surface: str
    start: int
    end: int
    confidence: float
    source: Literal["gazetteer", "regex", "model", "context"] = "gazetteer"


class RetrievalHit(BaseModel):
    evidence_id: str
    document_id: str
    document_title: str = ""
    section: str = ""
    page: int | None = None
    amharic_text: str = ""
    english_text: str = ""
    review_status: str = "unreviewed"
    medical_subject: str = ""
    document_version: str = ""
    dense_score: float | None = None
    bm25_score: float | None = None
    fused_score: float
    rerank_score: float | None = None
    polarity_conflict: bool = False
    source_kind: Literal["document_evidence"] = "document_evidence"


class AnalyzeResponse(BaseModel):
    language: str
    normalized_text: str
    intent: str
    intent_labels: list[str] = Field(default_factory=list)
    intent_confidence: float
    intent_backend: str
    entities: list[Entity] = Field(default_factory=list)
    context_resolution: dict[str, Any] = Field(default_factory=dict)
    retrieval_results: list[RetrievalHit] = Field(default_factory=list)
    requires_clarification: bool = False
    clarification_question_key: str | None = None
    confidence_status: Literal[
        "high",
        "medium",
        "low",
        "abstain",
        "requires_calibration",
        "requires_training_data",
    ]
    safety: dict[str, Any] = Field(default_factory=dict)
    evidence_policy: str = (
        "Retrieved texts are source evidence. Intent/entity labels are model or lexicon "
        "predictions. Unreviewed translations are not clinically verified."
    )
    component: str = "we-care-nlu"
    generative_answer: None = None


class ReplyResponse(BaseModel):
    """One user turn. Retrieval runs on the English query. The reply is not generated."""

    amharic_reply: str
    english_query: str
    english_reply: str = ""
    translation_backend: str
    translation_status: Literal["translated", "template", "source_only", "abstained", "qa_failed"]
    qa_ok: bool = True
    qa_notes: list[str] = Field(default_factory=list)
    evidence_id: str | None = None
    document_id: str = ""
    document_title: str = ""
    page: int | None = None
    section: str = ""
    review_status: str = ""
    intent: str
    intent_labels: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    requires_clarification: bool = False
    clarification_question_key: str | None = None
    confidence_status: str
    safety: dict[str, Any] = Field(default_factory=dict)
    generative_answer: None = None


class SearchRequest(BaseModel):
    query: str
    language: str | None = "am"
    top_k: int | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    model_key: str | None = None


class ChunkRecord(BaseModel):
    evidence_id: str
    document_id: str
    document_title: str
    section: str
    page: int | None
    english_text: str
    amharic_text: str
    review_status: str
    medical_subject: str
    document_version: str
    source_kind: str
    heading: str = ""
    terms: list[str] = Field(default_factory=list)
    content_hash: str = ""
