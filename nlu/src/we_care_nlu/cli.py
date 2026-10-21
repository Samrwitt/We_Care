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
def serve(host: str = "0.0.0.0", port: int = 8088) -> None:
    import uvicorn

    uvicorn.run("we_care_nlu.api.app:app", host=host, port=port, reload=False)


@app.command("list-models")
def list_models() -> None:
    cfg = load_config()
    for key, spec in (cfg.get("embeddings", {}).get("models") or {}).items():
        typer.echo(f"{key}: {spec.get('hf_id')} kind={spec.get('kind')} sts={spec.get('amharic_sentence_encoder')}")
