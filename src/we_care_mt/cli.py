"""Command-line interface for the We Care English→Amharic medical MT pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from we_care_mt import __version__
from we_care_mt.config import load_config
from we_care_mt.logging_utils import setup_logging

app = typer.Typer(
    add_completion=False,
    help="English-to-Amharic medical PDF translation pipeline. Output is not clinically validated.",
)


def _cfg(config: Optional[Path], models: Optional[Path]):
    return load_config(config, models)


@app.callback()
def main(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to default.yaml"),
    models: Optional[Path] = typer.Option(None, "--models-config", help="Path to models.yaml"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    cfg = _cfg(config, models)
    setup_logging(
        log_dir=cfg.path("log_dir"),
        level="DEBUG" if verbose else str(cfg.section("logging").get("level", "INFO")),
        json_logs=bool(cfg.section("logging").get("json_logs", False)),
    )
    ctx.obj = {"cfg": cfg}


@app.command()
def version() -> None:
    """Print package version."""
    typer.echo(__version__)


@app.command()
def translate(
    ctx: typer.Context,
    input_path: Path = typer.Argument(..., exists=True, help="PDF file or folder of PDFs"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="dsfsi_600m | dsfsi_1_3b | nllb_baseline_600m | mock"),
    backend: Optional[str] = typer.Option(None, "--backend", help="nllb | mock"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size"),
    device: Optional[str] = typer.Option(None, "--device", help="auto | cuda | cpu"),
    beams: Optional[int] = typer.Option(None, "--beams"),
) -> None:
    """Translate one PDF or a directory of PDFs."""
    from tqdm import tqdm

    from we_care_mt.pipeline import collect_pdfs

    cfg = ctx.obj["cfg"]
    inf = cfg.section("inference")
    if batch_size:
        inf["batch_size"] = batch_size
    if device:
        inf["device"] = device
    if beams:
        inf["num_beams"] = beams
    cfg.raw["inference"] = inf

    pdfs = collect_pdfs(input_path)
    if not pdfs:
        raise typer.BadParameter(f"No PDFs found under {input_path}")
    typer.echo(f"Processing {len(pdfs)} PDF(s) with model={model or cfg.default_model_key()}")
    bar = tqdm(total=len(pdfs), unit="pdf")
    results = []
    for pdf in pdfs:
        from we_care_mt.pipeline import process_pdf

        results.append(
            process_pdf(pdf, cfg, model_key=model, backend=backend, output_dir=output_dir)
        )
        bar.update(1)
    bar.close()
    failed = [r for r in results if r.get("status") == "error"]
    typer.echo(f"Done. {len(results) - len(failed)} ok, {len(failed)} failed.")
    if failed:
        raise typer.Exit(code=1)


@app.command()
def extract(
    ctx: typer.Context,
    input_path: Path = typer.Argument(..., exists=True),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
) -> None:
    """Extract structure from a PDF without translating."""
    import json

    from we_care_mt.extraction.pdf_extractor import extract_pdf
    from we_care_mt.schema import to_plain

    cfg = ctx.obj["cfg"]
    doc = extract_pdf(input_path, cfg.section("extraction"))
    dest = output or Path("output") / f"{input_path.stem}.structure.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(to_plain(doc), ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"Wrote {dest} ({doc.page_count} pages, {len(doc.blocks)} blocks, {len(doc.warnings)} warnings)")


@app.command("qa")
def qa_cmd(
    ctx: typer.Context,
    alignment: Path = typer.Argument(..., exists=True, help="alignment JSON from a previous run"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
) -> None:
    """Re-run QA on an existing alignment file."""
    import json

    from we_care_mt.glossary.manager import Glossary
    from we_care_mt.output.fonts import locate_fonts
    from we_care_mt.qa.checks import run_qa
    from we_care_mt.qa.report import write_qa_reports
    from we_care_mt.schema import Segment

    cfg = ctx.obj["cfg"]
    data = json.loads(alignment.read_text(encoding="utf-8"))
    segs = []
    for row in data.get("segments", []):
        segs.append(
            Segment(
                segment_id=row["segment_id"],
                document_name=row.get("document_name", ""),
                page_number=int(row.get("page_number") or 0),
                block_id=row.get("block_id", ""),
                kind=row.get("kind", "paragraph"),
                source_text=row.get("source_text", ""),
                translated_text=row.get("translated_text", ""),
                model_id=row.get("model_id", ""),
                status=row.get("status", "translated"),
                glossary_hits=[x for x in (row.get("glossary_hits") or "").split("|") if x],
                flags=[x for x in (row.get("flags") or "").split("|") if x],
            )
        )
    glossary = None
    gp = cfg.path("glossary")
    if gp.exists():
        fp = cfg.path("freeze_patterns")
        glossary = Glossary.load(gp, fp if fp.exists() else None)
    report = run_qa(segs, data.get("document_name", alignment.stem), segs[0].model_id if segs else "", glossary, cfg.section("qa"))
    dest = output_dir or alignment.parent
    paths = write_qa_reports(report, dest, locate_fonts(cfg.path("fonts_dir")))
    typer.echo(f"QA findings={report.n_findings} critical={report.n_critical} → {paths['markdown']}")


@app.command()
def benchmark(
    ctx: typer.Context,
    models: str = typer.Option(
        "dsfsi_600m,dsfsi_1_3b,nllb_baseline_600m",
        "--models",
        help="Comma-separated model keys. Add mock for a no-download run.",
    ),
    eval_set: Optional[Path] = typer.Option(None, "--eval-set"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    backend: Optional[str] = typer.Option(None, "--backend"),
) -> None:
    """Compare models on the evaluation set. Does not fabricate scores."""
    from we_care_mt.evaluation.benchmark import run_benchmark

    cfg = ctx.obj["cfg"]
    keys = [k.strip() for k in models.split(",") if k.strip()]
    dest = output or cfg.path("output_dir") / "benchmark" / "benchmark_report.json"
    report = run_benchmark(cfg, keys, eval_path=eval_set, backend=backend, out_path=dest)
    rec = report.get("recommendation") or {}
    typer.echo(f"Wrote {dest}")
    typer.echo(f"Recommended: {rec.get('recommended')} — {rec.get('reason')}")
    for row in report.get("models", []):
        typer.echo(
            f"  {row.get('model_key')}: status={row.get('status')} "
            f"BLEU={row.get('bleu')} chrF++={row.get('chrfpp')} "
            f"crit={row.get('critical_mistranslation_rate')} err={row.get('error', '')}"
        )


@app.command("download-models")
def download_models(
    ctx: typer.Context,
    model: str = typer.Option("dsfsi_600m", "--model", "-m", help="Model key or 'all'"),
) -> None:
    """Download Hugging Face weights into the configured cache."""
    from we_care_mt.translation.nllb import NLLBTranslator

    cfg = ctx.obj["cfg"]
    keys = list(cfg.models.get("models", {}).keys()) if model == "all" else [model]
    cache = str(cfg.path("hf_cache_dir"))
    Path(cache).mkdir(parents=True, exist_ok=True)
    for key in keys:
        entry = cfg.model_entry(key)
        hf_id = str(entry["hf_id"])
        typer.echo(f"Downloading {hf_id} (license={entry.get('license')}, clinical={entry.get('clinically_validated')})")
        tr = NLLBTranslator(hf_id=hf_id, device="cpu", cache_dir=cache)
        tr.load()
        tr.unload()
        typer.echo(f"  ok: {hf_id}")


@app.command("verify-models")
def verify_models(ctx: typer.Context) -> None:
    """Print registry metadata: license, language codes, VRAM, clinical status."""
    cfg = ctx.obj["cfg"]
    lang = cfg.models.get("language", {})
    typer.echo(f"Source {lang.get('source_code')} → target {lang.get('target_code')}")
    for key, entry in (cfg.models.get("models") or {}).items():
        typer.echo(
            f"- {key}: {entry.get('hf_id')} | license={entry.get('license')} "
            f"| VRAM~{entry.get('recommended_vram_gb')}GB | clinical={entry.get('clinically_validated')} "
            f"| max_in={entry.get('max_input_tokens')}"
        )
        typer.echo(f"    {str(entry.get('notes', '')).strip()[:200]}")


if __name__ == "__main__":
    app()
