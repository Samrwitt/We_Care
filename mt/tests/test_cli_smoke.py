"""End-to-end smoke test. Uses the mock backend — no neural model download."""

from __future__ import annotations

from pathlib import Path

from we_care_mt.config import load_config
from we_care_mt.evaluation.metrics import corpus_bleu, corpus_chrfpp
from we_care_mt.pipeline import process_pdf


def test_smoke_pipeline_mock(mini_pdf, tmp_path, project_root: Path):
    cfg = load_config(project_root / "configs" / "default.yaml")
    summary = process_pdf(
        mini_pdf,
        cfg,
        model_key="mock",
        backend="mock",
        output_dir=tmp_path / "out",
    )
    assert summary["segments"] >= 3
    artifacts = summary["artifacts"]
    for key in ("amharic_pdf", "bilingual_pdf", "docx", "json", "csv", "qa_markdown"):
        assert key in artifacts, key
        assert Path(artifacts[key]).exists()
    assert summary["qa"]["medically_safe_claim"]
    # Failed translations must be explicit, never silently omitted.
    align = Path(artifacts["json"]).read_text(encoding="utf-8")
    assert "Family planning" in align
    assert "100" in align


def test_sacrebleu_runs_on_known_strings():
    hyps = ["የቤተሰብ ምጣኔ የእናቶችን ጤና ያሻሽላል።"]
    refs = ["የቤተሰብ ምጣኔ የእናቶችን ጤና ያሻሽላል።"]
    bleu = corpus_bleu(hyps, refs)
    chrf = corpus_chrfpp(hyps, refs)
    assert bleu > 90
    assert chrf > 90


def test_benchmark_mock_does_not_fabricate_model_scores(project_root, tmp_path):
    from we_care_mt.config import load_config
    from we_care_mt.evaluation.benchmark import run_benchmark

    cfg = load_config(project_root / "configs" / "default.yaml")
    report = run_benchmark(
        cfg,
        ["mock"],
        eval_path=project_root / "data" / "evaluation" / "sample_eval.jsonl",
        backend="mock",
        include_google=False,
        out_path=tmp_path / "bench.json",
    )
    assert report["models"][0]["status"] == "ok"
    assert report["models"][0]["bleu"] is not None
    assert "not a clinical validation" in report["recommendation"]["reason"].lower() or "not" in report["clinical_disclaimer"].lower()
