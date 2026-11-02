import pytest

from we_care_mt.extraction.pdf_extractor import extract_pdf
from we_care_mt.segmentation.chunker import segments_from_document


pymupdf = pytest.importorskip("pymupdf")


def test_extracts_headings_paragraphs_and_lists(mini_pdf):
    doc = extract_pdf(mini_pdf, {"exclude_repeating_headers": True, "detect_tables": True})
    assert doc.page_count == 2
    kinds = {b.kind for b in doc.blocks}
    assert "heading" in kinds or "paragraph" in kinds
    texts = " ".join(b.text for b in doc.blocks)
    assert "Family planning" in texts
    assert "doxycycline 100 mg" in texts
    # Running footer should be classified, not silently dropped from the structure.
    running = [b for b in doc.blocks if b.kind in {"header", "footer", "page_number"}]
    assert running
    segs = segments_from_document(doc, {"max_chars": 220})
    src = " ".join(s.source_text for s in segs)
    assert "Family planning" in src
    # Footers excluded from translation segments
    assert not any(s.kind in {"header", "footer", "page_number"} for s in segs)


def test_empty_pdf_is_flagged(tmp_path):
    import pymupdf as fitz

    path = tmp_path / "blank.pdf"
    d = fitz.open()
    d.new_page()
    d.save(path)
    d.close()
    doc = extract_pdf(path, {})
    assert doc.empty_or_image_pages or any(w.code == "empty_page" for w in doc.warnings)
