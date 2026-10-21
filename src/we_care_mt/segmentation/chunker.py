"""Turn a DocumentStructure into translation-safe Segment objects."""

from __future__ import annotations

import re
from typing import Any

from we_care_mt.schema import ContentBlock, DocumentStructure, Segment
from we_care_mt.segmentation.splitter import pack_sentences, split_sentences

SKIP_KINDS = {"header", "footer", "page_number"}
NON_TRANSLATABLE = re.compile(
    r"""^(
        [\dIVXLC.,;:%/\-\s]+
        |https?://\S+
        |[\w.+-]+@[\w.-]+
        )$""",
    re.VERBOSE | re.I,
)


def segments_from_document(doc: DocumentStructure, config: dict[str, Any] | None = None) -> list[Segment]:
    cfg = config or {}
    max_chars = int(cfg.get("max_chars", 420))
    keep_lists_atomic = bool(cfg.get("keep_list_items_atomic", True))
    segments: list[Segment] = []
    n = 0

    for block in doc.blocks:
        if block.kind in SKIP_KINDS or "excluded_from_translation" in block.flags:
            continue
        if block.kind == "table" and block.table is not None:
            segments.extend(_segments_from_table(doc, block, max_chars, n))
            n = _next_index(segments, n)
            continue

        text = (block.text or "").strip()
        if not text:
            continue

        if block.kind in {"heading"} or (keep_lists_atomic and block.kind == "list_item"):
            chunks = pack_sentences([text], max_chars)
        else:
            chunks = pack_sentences(split_sentences(text), max_chars)

        for chunk in chunks:
            n += 1
            translatable = _is_translatable(chunk)
            seg = Segment(
                segment_id=f"{doc.document_name}::s{n:05d}",
                document_name=doc.document_name,
                page_number=block.page,
                block_id=block.block_id,
                kind=block.kind,
                source_text=chunk,
                is_translatable=translatable,
                status="skipped" if not translatable else "pending",
                translated_text=chunk if not translatable else "",
                flags=list(block.flags),
            )
            if not translatable:
                seg.flags.append("non_translatable")
            if block.source_confidence < 0.8:
                seg.flags.append("uncertain_extraction")
            segments.append(seg)
    return segments


def _segments_from_table(
    doc: DocumentStructure,
    block: ContentBlock,
    max_chars: int,
    start_n: int,
) -> list[Segment]:
    table = block.table
    assert table is not None
    segs: list[Segment] = []
    n = start_n
    for r_i, row in enumerate(table.cells):
        for c_i, cell in enumerate(row):
            cell = (cell or "").strip()
            if not cell:
                continue
            pieces = pack_sentences(split_sentences(cell), max_chars) if len(cell) > max_chars else [cell]
            for piece in pieces:
                n += 1
                translatable = _is_translatable(piece)
                flags = ["table_cell"]
                if table.uncertain:
                    flags.append("uncertain_table")
                segs.append(
                    Segment(
                        segment_id=f"{doc.document_name}::s{n:05d}",
                        document_name=doc.document_name,
                        page_number=block.page,
                        block_id=f"{block.block_id}#r{r_i}c{c_i}",
                        kind="table_cell",
                        source_text=piece,
                        is_translatable=translatable,
                        status="skipped" if not translatable else "pending",
                        translated_text=piece if not translatable else "",
                        flags=flags,
                    )
                )
    return segs


def _is_translatable(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if NON_TRANSLATABLE.match(t):
        return False
    letters = [ch for ch in t if ch.isalpha()]
    if not letters:
        return False
    return True


def _next_index(segments: list[Segment], fallback: int) -> int:
    if not segments:
        return fallback
    last = segments[-1].segment_id.rsplit("s", 1)[-1]
    try:
        return int(last)
    except ValueError:
        return fallback + len(segments)
