from we_care_mt.glossary.manager import Glossary
from we_care_mt.qa.checks import extract_numbers, run_qa
from we_care_mt.schema import Segment


def _seg(sid, src, tgt, page=1, **kwargs):
    return Segment(
        segment_id=sid,
        document_name="t.pdf",
        page_number=page,
        block_id="b1",
        kind="paragraph",
        source_text=src,
        translated_text=tgt,
        model_id="mock",
        status="translated",
        **kwargs,
    )


def test_number_extraction_keeps_dosage():
    assert "100mg" in [n.replace(" ", "") for n in extract_numbers("Take 100 mg twice daily")]


def test_flags_missing_negation(glossary_csv):
    g = Glossary.load(glossary_csv)
    report = run_qa(
        [
            _seg(
                "s1",
                "Do not use emergency contraception as a regular method.",
                "የአስቸኳይ የእርግዝና መከላከያ እንደ መደበኛ ዘዴ ይጠቀሙ።",
            )
        ],
        "t.pdf",
        "mock",
        g,
    )
    checks = {f.check for f in report.findings}
    assert "negation_error" in checks
    assert report.n_critical >= 1
    assert "not establish medical safety" in report.medically_safe_claim.lower() or "human review" in report.medically_safe_claim.lower()


def test_flags_hallucinated_dosage():
    report = run_qa(
        [_seg("s2", "Give oxytocin 10 IU now.", "ኦክሲቶሲን 40 IU አሁን ይስጡ።")],
        "t.pdf",
        "mock",
        None,
    )
    checks = {f.check for f in report.findings}
    assert "number_mismatch" in checks or "hallucinated_number" in checks


def test_missing_translation_is_critical():
    seg = _seg("s3", "Seek emergency care.", "")
    seg.status = "failed"
    report = run_qa([seg], "t.pdf", "mock", None)
    assert any(f.check == "missing_translation" and f.severity == "critical" for f in report.findings)
