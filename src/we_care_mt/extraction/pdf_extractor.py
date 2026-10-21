"""PyMuPDF extraction with structure preservation and uncertainty flags."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from we_care_mt.extraction.headers import detect_repeating_bands, is_running_element, looks_like_page_number
from we_care_mt.extraction.reading_order import detect_two_column, sort_blocks
from we_care_mt.extraction.structure import body_font_size, classify_block
from we_care_mt.extraction.tables import bbox_overlap, extract_tables
from we_care_mt.schema import ContentBlock, DocumentStructure, ExtractionWarning, TableBlock


def extract_pdf(path: str | Path, config: dict[str, Any] | None = None) -> DocumentStructure:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required. Install with: pip install pymupdf") from exc

    cfg = config or {}
    pdf_path = Path(path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    doc = fitz.open(pdf_path)
    try:
        return _extract_open_document(doc, pdf_path, cfg)
    finally:
        doc.close()


def _extract_open_document(doc, pdf_path: Path, cfg: dict[str, Any]) -> DocumentStructure:
    header_frac = float(cfg.get("header_band_frac", 0.08))
    footer_frac = float(cfg.get("footer_band_frac", 0.08))
    min_repeat = int(cfg.get("header_repeat_min_pages", 3))
    heading_delta = float(cfg.get("min_heading_size_delta", 1.6))
    detect_tbl = bool(cfg.get("detect_tables", True))
    gap = float(cfg.get("two_column_gap_min_pt", 18.0))
    exclude_running = bool(cfg.get("exclude_repeating_headers", True))

    stem = pdf_path.stem
    page_count = doc.page_count
    metadata = dict(doc.metadata or {})

    header_band_texts: list[list[str]] = []
    footer_band_texts: list[list[str]] = []
    all_sizes: list[float] = []
    page_dicts: list[dict[str, Any]] = []

    for page in doc:
        height = float(page.rect.height)
        d = page.get_text("dict")
        page_dicts.append(d)
        tops: list[str] = []
        bots: list[str] = []
        for block in d.get("blocks", []):
            if block.get("type") != 0:
                continue
            bbox = block.get("bbox") or [0, 0, 0, 0]
            text = _block_text(block)
            if not text:
                continue
            y0, y1 = bbox[1], bbox[3]
            if y1 <= height * header_frac:
                tops.append(text)
            if y0 >= height * (1.0 - footer_frac):
                bots.append(text)
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = float(span.get("size") or 0)
                    if size:
                        all_sizes.append(size)
        header_band_texts.append(tops)
        footer_band_texts.append(bots)

    repeating_headers = detect_repeating_bands(header_band_texts, min_pages=min_repeat)
    repeating_footers = detect_repeating_bands(footer_band_texts, min_pages=min_repeat)
    body_size = body_font_size(all_sizes)

    warnings: list[ExtractionWarning] = []
    complex_pages: list[int] = []
    empty_pages: list[int] = []
    blocks: list[ContentBlock] = []
    tables: list[TableBlock] = []
    seq = 0

    for page_i, page in enumerate(doc, start=1):
        width = float(page.rect.width)
        height = float(page.rect.height)
        d = page_dicts[page_i - 1]
        page_tables: list[TableBlock] = []
        if detect_tbl:
            page_tables = extract_tables(page, page_i, stem)
            tables.extend(page_tables)
            if any(t.uncertain for t in page_tables):
                warnings.append(
                    ExtractionWarning(
                        page=page_i,
                        code="uncertain_table",
                        message="One or more tables on this page need human layout review.",
                        severity="warning",
                    )
                )

        text_blocks_raw = [b for b in d.get("blocks", []) if b.get("type") == 0]
        image_blocks = [b for b in d.get("blocks", []) if b.get("type") == 1]
        if image_blocks and not text_blocks_raw:
            empty_pages.append(page_i)
            warnings.append(
                ExtractionWarning(
                    page=page_i,
                    code="image_only_page",
                    message="Page appears image-only; selectable text was not extracted.",
                    severity="error",
                )
            )
        page_text = page.get_text("text").strip()
        if not page_text and page_i not in empty_pages:
            empty_pages.append(page_i)
            warnings.append(
                ExtractionWarning(
                    page=page_i,
                    code="empty_page",
                    message="No selectable text on this page.",
                    severity="info",
                )
            )

        bboxes = [list(b.get("bbox") or [0, 0, 0, 0]) for b in text_blocks_raw]
        two_col = detect_two_column(bboxes, width, gap)
        if two_col:
            complex_pages.append(page_i)
            warnings.append(
                ExtractionWarning(
                    page=page_i,
                    code="two_column",
                    message="Two-column layout detected; reading order was reconstructed, not guaranteed.",
                    severity="info",
                )
            )

        page_blocks: list[ContentBlock] = []
        for raw in text_blocks_raw:
            bbox = [float(x) for x in (raw.get("bbox") or [0, 0, 0, 0])]
            text = _block_text(raw)
            if not text:
                continue
            # Skip text that sits inside a detected table; the table is the source of truth.
            in_table = any(bbox_overlap(bbox, t.bbox) for t in page_tables if t.bbox)
            if in_table:
                continue
            font_name, font_size, bold, flags = _span_style(raw)
            y0 = bbox[1]
            in_header = y0 <= height * header_frac
            in_footer = bbox[3] >= height * (1.0 - footer_frac)
            running = is_running_element(text, repeating_headers + repeating_footers)
            kind, level, marker = classify_block(
                text,
                font_size,
                bold,
                body_size,
                y0,
                height,
                heading_delta,
                in_header,
                in_footer,
                running,
            )
            if looks_like_page_number(text) and in_footer:
                kind = "page_number"
            seq += 1
            flag_list: list[str] = []
            if running:
                flag_list.append("repeating_running_element")
            if abs(bbox[2] - bbox[0]) < 8:
                flag_list.append("narrow_block")
            block = ContentBlock(
                block_id=f"{stem}-p{page_i}-b{seq}",
                page=page_i,
                bbox=bbox,
                text=text,
                kind=kind,
                font_size=font_size,
                font_name=font_name,
                bold=bold,
                heading_level=level,
                list_marker=marker,
                source_confidence=0.7 if flag_list else 1.0,
                flags=flag_list,
            )
            if exclude_running and kind in {"header", "footer", "page_number"}:
                # Keep them in the structure for audit, but mark skipped for translation.
                block.flags.append("excluded_from_translation")
            page_blocks.append(block)

        for table in page_tables:
            seq += 1
            rendered = _table_as_text(table)
            page_blocks.append(
                ContentBlock(
                    block_id=f"{stem}-p{page_i}-tbl-{table.table_id}",
                    page=page_i,
                    bbox=table.bbox,
                    text=rendered,
                    kind="table",
                    table=table,
                    flags=["uncertain_table"] if table.uncertain else [],
                    source_confidence=0.55 if table.uncertain else 0.85,
                )
            )

        page_blocks = sort_blocks(page_blocks, two_col, width)
        blocks.extend(page_blocks)

        # Overlapping text heuristic (InDesign decorative layers).
        if _excessive_overlap(page_blocks):
            complex_pages.append(page_i)
            warnings.append(
                ExtractionWarning(
                    page=page_i,
                    code="overlapping_blocks",
                    message="Overlapping text blocks; extraction order may be unreliable.",
                    severity="warning",
                )
            )

    title = str(metadata.get("title") or "").strip() or stem
    return DocumentStructure(
        source_path=str(pdf_path),
        document_name=pdf_path.name,
        title=title,
        page_count=page_count,
        blocks=blocks,
        tables=tables,
        repeating_headers=repeating_headers,
        repeating_footers=repeating_footers,
        warnings=warnings,
        complex_layout_pages=sorted(set(complex_pages)),
        empty_or_image_pages=empty_pages,
        metadata={
            "pdf_metadata": metadata,
            "body_font_size": body_size,
            "page_count": page_count,
        },
    )


def _block_text(block: dict[str, Any]) -> str:
    parts: list[str] = []
    for line in block.get("lines", []):
        spans = [str(sp.get("text") or "") for sp in line.get("spans", [])]
        line_text = "".join(spans).strip()
        if line_text:
            parts.append(line_text)
    return " ".join(parts).strip()


def _span_style(block: dict[str, Any]) -> tuple[str, float, bool, int]:
    sizes: list[float] = []
    fonts: list[str] = []
    flags = 0
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            sizes.append(float(span.get("size") or 0))
            fonts.append(str(span.get("font") or ""))
            flags |= int(span.get("flags") or 0)
    size = max(sizes) if sizes else 0.0
    font = fonts[0] if fonts else ""
    bold = bool(flags & 16) or ("Bold" in font) or ("bold" in font)
    return font, size, bold, flags


def _table_as_text(table: TableBlock) -> str:
    lines = []
    if table.caption:
        lines.append(table.caption)
    for row in table.cells:
        lines.append(" | ".join(c.strip() for c in row if True))
    return "\n".join(lines).strip()


def _excessive_overlap(blocks: list[ContentBlock], thresh: int = 4) -> bool:
    hits = 0
    for i, a in enumerate(blocks):
        for b in blocks[i + 1 : i + 8]:
            if bbox_overlap(a.bbox, b.bbox, iou_thresh=0.5):
                hits += 1
            if hits >= thresh:
                return True
    return False
