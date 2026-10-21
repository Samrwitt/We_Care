"""Canonical data structures passed between pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any


def to_plain(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(x) for x in obj]
    return obj


@dataclass
class ExtractionWarning:
    page: int
    code: str
    message: str
    severity: str = "warning"  # info | warning | error


@dataclass
class TableBlock:
    table_id: str
    page: int
    bbox: list[float]
    n_rows: int
    n_cols: int
    cells: list[list[str]]
    caption: str = ""
    uncertain: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class ContentBlock:
    block_id: str
    page: int
    bbox: list[float]
    text: str
    kind: str  # heading, paragraph, list_item, table, caption, footnote, page_number, header, footer, figure_alt
    font_size: float = 0.0
    font_name: str = ""
    bold: bool = False
    reading_order: int = 0
    heading_level: int | None = None
    list_marker: str = ""
    table: TableBlock | None = None
    source_confidence: float = 1.0
    flags: list[str] = field(default_factory=list)


@dataclass
class DocumentStructure:
    source_path: str
    document_name: str
    title: str
    page_count: int
    blocks: list[ContentBlock] = field(default_factory=list)
    tables: list[TableBlock] = field(default_factory=list)
    repeating_headers: list[str] = field(default_factory=list)
    repeating_footers: list[str] = field(default_factory=list)
    warnings: list[ExtractionWarning] = field(default_factory=list)
    complex_layout_pages: list[int] = field(default_factory=list)
    empty_or_image_pages: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Segment:
    segment_id: str
    document_name: str
    page_number: int
    block_id: str
    kind: str
    source_text: str
    translated_text: str = ""
    model_id: str = ""
    is_translatable: bool = True
    glossary_hits: list[str] = field(default_factory=list)
    protected: dict[str, str] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | translated | failed | skipped
    error: str = ""

    def to_alignment_row(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "block_id": self.block_id,
            "kind": self.kind,
            "source_text": self.source_text,
            "translated_text": self.translated_text,
            "model_id": self.model_id,
            "status": self.status,
            "glossary_hits": "|".join(self.glossary_hits),
            "flags": "|".join(self.flags),
            "error": self.error,
        }


@dataclass
class QAFinding:
    segment_id: str
    page_number: int
    check: str
    severity: str  # critical | major | minor | info
    message: str
    source_excerpt: str = ""
    target_excerpt: str = ""
    recommended_action: str = "Human bilingual medical review required."


@dataclass
class QAReport:
    document_name: str
    model_id: str
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    n_segments: int = 0
    n_findings: int = 0
    n_critical: int = 0
    n_major: int = 0
    findings: list[QAFinding] = field(default_factory=list)
    summary: str = ""
    medically_safe_claim: str = (
        "Automated checks do not establish medical safety. "
        "Qualified human review is required before any clinical use."
    )
