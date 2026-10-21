from we_care_mt.output.alignment import write_alignment
from we_care_mt.output.bilingual import write_bilingual_pdf
from we_care_mt.output.docx_writer import write_docx
from we_care_mt.output.fonts import locate_fonts
from we_care_mt.output.pdf_writer import write_amharic_pdf
from we_care_mt.schema import ContentBlock, DocumentStructure, Segment


def _doc():
    block = ContentBlock(
        block_id="b1",
        page=1,
        bbox=[72, 700, 500, 740],
        text="Family planning reduces maternal mortality.",
        kind="paragraph",
    )
    return DocumentStructure(
        source_path="x.pdf",
        document_name="x.pdf",
        title="የቤተሰብ ምጣኔ መመሪያ",
        page_count=1,
        blocks=[block],
    )


def _seg():
    return Segment(
        segment_id="s1",
        document_name="x.pdf",
        page_number=1,
        block_id="b1",
        kind="paragraph",
        source_text="Family planning reduces maternal mortality.",
        translated_text="የቤተሰብ ምጣኔ የእናቶች ሞትን ይቀንሳል።",
        model_id="mock",
        status="translated",
    )


def test_amharic_pdf_contains_ethiopic(tmp_path, fonts_dir):
    fonts = locate_fonts(fonts_dir)
    path = write_amharic_pdf(_doc(), [_seg()], tmp_path / "am.pdf", fonts)
    assert path.exists() and path.stat().st_size > 1000
    try:
        import pymupdf as fitz

        extracted = fitz.open(path)[0].get_text()
        assert "የቤተሰብ" in extracted or "ምጣኔ" in extracted
    except ImportError:
        raw = path.read_bytes()
        assert "NotoSansEthiopic".encode() in raw or b"Ethiopic" in raw


def test_bilingual_and_docx_and_alignment(tmp_path, fonts_dir):
    fonts = locate_fonts(fonts_dir)
    doc, seg = _doc(), _seg()
    bi = write_bilingual_pdf(doc, [seg], tmp_path / "bi.pdf", fonts)
    dx = write_docx(doc, [seg], tmp_path / "out.docx", fonts)
    al = write_alignment(doc, [seg], tmp_path)
    assert bi.exists() and dx.exists()
    assert al["json"].exists() and al["csv"].exists()
    csv_text = al["csv"].read_text(encoding="utf-8")
    assert "የቤተሰብ ምጣኔ" in csv_text
    assert "Family planning" in csv_text
