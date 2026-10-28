from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from we_care_nlu.config import load_config
from we_care_nlu.pipeline import NLUEngine

app = typer.Typer(add_completion=False, help="We Care Amharic NLU (no generative decoding).")


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True),
    model: str = typer.Option("hash", "--model"),
    review_status: str = typer.Option("unreviewed"),
) -> None:
    eng = NLUEngine(load_config(), model_key=model)
    n = eng.ingest_file(str(path), default_review_status=review_status)
    typer.echo(json.dumps({"ingested": n, "total": len(eng.retriever.index.chunks)}, ensure_ascii=False))


@app.command()
def analyze(
    text: str = typer.Argument(...),
    model: str = typer.Option("hash", "--model"),
    kb: Optional[Path] = typer.Option(None, "--kb"),
) -> None:
    eng = NLUEngine(load_config(), model_key=model)
    if kb:
        eng.ingest_file(str(kb))
    from we_care_nlu.contracts.schemas import AnalyzeRequest

    out = eng.analyze(AnalyzeRequest(text=text))
    typer.echo(out.model_dump_json(indent=2, ensure_ascii=False))


@app.command()
def reply(
    text: str = typer.Argument(...),
    model: str = typer.Option("hash", "--model"),
    kb: Optional[Path] = typer.Option(None, "--kb"),
) -> None:
    """Translate an Amharic question to English, retrieve, and render an Amharic reply."""
    eng = NLUEngine(load_config(), model_key=model)
    if kb:
        eng.ingest_file(str(kb), default_review_status="source")
    from we_care_nlu.contracts.schemas import AnalyzeRequest

    out = eng.reply(AnalyzeRequest(text=text))
    typer.echo(out.model_dump_json(indent=2, ensure_ascii=False))


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8088) -> None:
    import uvicorn

    uvicorn.run("we_care_nlu.api.app:app", host=host, port=port, reload=False)


@app.command("list-models")
def list_models() -> None:
    cfg = load_config()
    for key, spec in (cfg.get("embeddings", {}).get("models") or {}).items():
        typer.echo(f"{key}: {spec.get('hf_id')} kind={spec.get('kind')} sts={spec.get('amharic_sentence_encoder')}")


@app.command("kb-audit")
def kb_audit(
    kb_dir: Optional[Path] = typer.Option(None, "--kb-dir", help="Directory of KB JSON files"),
    out: Optional[Path] = typer.Option(None, "--out", help="Write JSON report path"),
) -> None:
    """Report KB coverage, failed docs, and terminology candidate counts."""
    from we_care_nlu.terminology.audit import audit_kb, write_audit_report

    report = audit_kb(kb_dir=kb_dir)
    if out:
        write_audit_report(out, report)
        typer.echo(f"wrote {out}")
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command("extract-terminology")
def extract_terminology(
    kb: Optional[Path] = typer.Option(None, "--kb", help="Optional single KB JSON to scan"),
    seed_gazetteer: bool = typer.Option(True, "--seed-gazetteer/--no-seed-gazetteer"),
) -> None:
    """Extract pending medical concept candidates from KB; seed approved from gazetteer."""
    from we_care_nlu.terminology.index import ConceptIndex

    eng = NLUEngine(load_config(), model_key="hash")
    idx = ConceptIndex().load()
    if seed_gazetteer:
        n = idx.ensure_seeded_from_gazetteer()
        typer.echo(json.dumps({"seeded": n, **idx.stats()}, ensure_ascii=False))
    if kb:
        eng.ingest_file(str(kb), default_review_status="source", rebuild=True)
    else:
        from we_care_nlu.terminology.audit import discover_kb_files

        for path in discover_kb_files():
            eng.ingest_file(str(path), default_review_status="unreviewed", rebuild=False)
        eng.rebuild_index()
    cands = idx.rebuild_candidates_from_chunks(eng.retriever.index.chunks)
    typer.echo(
        json.dumps(
            {"candidates_written": len(cands), "candidates_path": str(idx.candidates_path), **idx.stats()},
            ensure_ascii=False,
        )
    )
