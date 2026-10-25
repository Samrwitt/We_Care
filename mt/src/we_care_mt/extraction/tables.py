"""Table extraction via PyMuPDF, with uncertainty flags for complex layouts."""

from __future__ import annotations

from we_care_mt.schema import TableBlock


def _clean_cell(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split())


def extract_tables(page, page_index: int, document_stem: str) -> list[TableBlock]:
    tables: list[TableBlock] = []
    finder = getattr(page, "find_tables", None)
    if finder is None:
        return tables
    try:
        found = finder()
    except Exception as exc:  # pragma: no cover - depends on PyMuPDF internals
        tables.append(
            TableBlock(
                table_id=f"{document_stem}-p{page_index}-table-failed",
                page=page_index,
                bbox=[0, 0, 0, 0],
                n_rows=0,
                n_cols=0,
                cells=[],
                uncertain=True,
                notes=[f"find_tables failed: {exc}"],
            )
        )
        return tables

    tab_list = getattr(found, "tables", None) or []
    for i, tab in enumerate(tab_list):
        notes: list[str] = []
        uncertain = False
        try:
            raw = tab.extract()
        except Exception as exc:
            raw = []
            uncertain = True
            notes.append(f"cell extract failed: {exc}")
        cells = [[_clean_cell(c) for c in (row or [])] for row in raw]
        n_rows = len(cells)
        n_cols = max((len(r) for r in cells), default=0)
        empty_ratio = 0.0
        if n_rows and n_cols:
            empty = sum(1 for r in cells for c in r if not str(c).strip())
            empty_ratio = empty / max(n_rows * n_cols, 1)
        if n_rows < 2 or n_cols < 2:
            uncertain = True
            notes.append("Table has fewer than 2 rows or 2 columns.")
        if empty_ratio > 0.45:
            uncertain = True
            notes.append(f"High empty-cell ratio ({empty_ratio:.2f}).")
        # Merged / InDesign tables often collapse to noisy one-row text.
        long_cells = sum(1 for r in cells for c in r if len(c) > 180)
        if long_cells >= 3:
            uncertain = True
            notes.append("Several very long cells; possible merged-cell / layout failure.")
        bbox = [float(x) for x in getattr(tab, "bbox", (0, 0, 0, 0))]
        tables.append(
            TableBlock(
                table_id=f"{document_stem}-p{page_index}-t{i+1}",
                page=page_index,
                bbox=bbox,
                n_rows=n_rows,
                n_cols=n_cols,
                cells=cells,
                uncertain=uncertain,
                notes=notes,
            )
        )
    return tables


def bbox_overlap(a: list[float], b: list[float], iou_thresh: float = 0.35) -> bool:
    if len(a) < 4 or len(b) < 4:
        return False
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max((ax1 - ax0) * (ay1 - ay0), 1.0)
    return (inter / area_a) >= iou_thresh
