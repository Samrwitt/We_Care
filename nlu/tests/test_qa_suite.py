"""Multi-topic Q&A suite: asserts checks and always writes qa_results.json."""

from pathlib import Path

from we_care_nlu.eval.qa_suite import DEFAULT_CASES, DEFAULT_OUT, run_suite

RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "eval"


def test_multi_topic_qa_suite_writes_results_json(tmp_path: Path):
    out = RESULTS_DIR / "qa_results.json"
    # Also mirror into tmp for CI isolation of the assertion path.
    tmp_out = tmp_path / "qa_results.json"
    report = run_suite(
        cases_path=DEFAULT_CASES,
        out_path=out,
        model="hash",
        with_docs=True,
        fail_fast=False,
    )
    tmp_out.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")

    assert out.exists()
    assert report["summary"]["total"] >= 15
    assert "results" in report
    for row in report["results"]:
        assert "id" in row
        assert "question" in row
        assert "passed" in row
        if not row.get("skipped"):
            assert row.get("answer") is not None
            assert "amharic_reply" in (row["answer"] or {})

    failed = [r for r in report["results"] if not r["passed"] and not r.get("skipped")]
    assert not failed, (
        "Q&A suite failures (see data/eval/qa_results.json): "
        + ", ".join(f"{r['id']}" for r in failed)
    )
