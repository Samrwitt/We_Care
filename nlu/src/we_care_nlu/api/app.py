"""FastAPI HTTP surface for We Care NLU.

Routes are thin wrappers around NLUEngine. Startup/KB loading lives in
`lifecycle.py`. Chat UI files are under `api/static/` (see README there).

Newcomers: read docs/BACKEND.md first.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from we_care_nlu import __version__
from we_care_nlu.api.lifecycle import runtime
from we_care_nlu.config import load_config
from we_care_nlu.contracts.schemas import AnalyzeRequest, AnalyzeResponse, ReplyResponse, SearchRequest

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_UI = _STATIC_DIR / "index.html"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Bind HTTP immediately; load embedder+KB in the background (unless under pytest)."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        runtime.warm_sync_for_tests()
    else:
        runtime.warm_in_background()
    yield


app = FastAPI(
    title="We Care NLU",
    version=__version__,
    description="Amharic medical understanding and retrieval. No generative decoding.",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Request bodies (API-only; core contracts live in contracts/schemas.py)
# ---------------------------------------------------------------------------


class IngestBody(BaseModel):
    path: str | None = None
    alignment: dict[str, Any] | None = None
    document_id: str | None = None
    document_version: str = "unknown"
    review_status: str = "unreviewed"


class ResolveBody(BaseModel):
    text: str
    context: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# UI + meta
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    return HTMLResponse(_UI.read_text(encoding="utf-8"))


@app.get("/meta")
def meta() -> dict[str, Any]:
    return {
        "service": "we-care-nlu",
        "component": 2,
        "status": "ok",
        "ui": "/",
        "docs": "/docs",
        "endpoints": [
            "GET /",
            "GET /health",
            "GET /models",
            "POST /documents/ingest",
            "POST /nlp/analyze",
            "POST /nlp/reply",
            "POST /retrieval/search",
            "POST /nlp/resolve-context",
        ],
        "generative_decoding": False,
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return runtime.health_payload(version=__version__)


@app.get("/models")
def models() -> dict[str, Any]:
    cfg = load_config()
    active = ""
    if runtime.ready.is_set() and runtime.engine is not None:
        active = runtime.engine.model_key
    return {
        "embeddings": cfg.get("embeddings", {}).get("models", {}),
        "active": active,
        "intent_backend": "lexicon_prototype",
        "ner_backend": "gazetteer_regex",
        "reranker": (cfg.get("retrieval") or {}).get("reranker"),
        "notes": [
            "AfroXLMR-base is an MLM encoder; mean pooling is a baseline, not an STS model.",
            "LaBSE lists Amharic among 109 languages.",
            "rasyosef Amharic embedding models are general passage retrievers, not clinical.",
            "Intent/NER labels are lexicon/regex until a labeled model is trained.",
        ],
    }


# ---------------------------------------------------------------------------
# Core NLU endpoints
# ---------------------------------------------------------------------------


@app.post("/documents/ingest")
def ingest(body: IngestBody) -> dict[str, Any]:
    eng = runtime.get()
    if body.alignment is None and not body.path:
        raise HTTPException(400, "Provide path or alignment JSON.")
    if body.path:
        n = eng.ingest_file(
            body.path,
            document_id=body.document_id,
            document_version=body.document_version,
            default_review_status=body.review_status,
        )
    else:
        n = eng.ingest_alignment_dict(
            body.alignment or {},
            document_id=body.document_id,
            document_version=body.document_version,
            default_review_status=body.review_status,
        )
    return {
        "ingested_chunks": n,
        "total_chunks": len(eng.retriever.index.chunks),
        "review_status": body.review_status,
    }


@app.post("/nlp/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """Structured understanding only (intent, entities, retrieval hits)."""
    return runtime.get().analyze(req)


@app.post("/nlp/reply", response_model=ReplyResponse)
def reply(req: AnalyzeRequest) -> ReplyResponse:
    """Full chat turn: analyze + compose Amharic/English answer text."""
    return runtime.get().reply(req)


@app.post("/retrieval/search")
def search(req: SearchRequest) -> dict[str, Any]:
    eng = runtime.get()
    hits = eng.retriever.search(req.query, top_k=req.top_k, filters=req.filters)
    return {"query": req.query, "hits": [h.model_dump() for h in hits]}


@app.post("/nlp/resolve-context")
def resolve_ctx(body: ResolveBody) -> dict[str, Any]:
    from we_care_nlu.contracts.schemas import ContextState

    ctx = ContextState.model_validate(body.context) if body.context else None
    return runtime.get().resolve_only(body.text, ctx)


@app.get("/kb/coverage")
def kb_coverage() -> dict[str, Any]:
    """Audit indexed KB files + terminology index (does not block on warm-up)."""
    from we_care_nlu.terminology.audit import audit_kb
    from we_care_nlu.terminology.index import ConceptIndex

    idx = ConceptIndex().load()
    if runtime.ready.is_set() and runtime.engine is not None:
        live = getattr(runtime.engine, "concepts", None)
        if live is not None:
            idx = live
    report = audit_kb(concept_index=idx)
    report["engine_ready"] = bool(runtime.ready.is_set() and runtime.engine is not None)
    if runtime.engine is not None:
        report["live_chunks"] = len(runtime.engine.retriever.index.chunks)
        report["live_embedder"] = runtime.engine.model_key
    return report

