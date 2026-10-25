from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from we_care_nlu import __version__
from we_care_nlu.config import load_config
from we_care_nlu.contracts.schemas import AnalyzeRequest, AnalyzeResponse, ReplyResponse, SearchRequest
from we_care_nlu.pipeline import NLUEngine

_UI = Path(__file__).resolve().parent / "static" / "index.html"
_KB_DIR = Path(__file__).resolve().parents[3] / "data" / "kb"
_GUIDELINES_KB = _KB_DIR / "rh_guidelines_en.json"
_DEFAULT_KB = _KB_DIR / "fp_user_kb.json"

_ENGINE: NLUEngine | None = None


def _ensure_kb() -> None:
    import os

    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    eng = get_engine()
    if len(eng.retriever.index.chunks) == 0:
        if _GUIDELINES_KB.exists():
            eng.ingest_file(str(_GUIDELINES_KB), document_version="source", default_review_status="source")
        elif _DEFAULT_KB.exists():
            eng.ingest_file(str(_DEFAULT_KB), document_version="demo", default_review_status="unreviewed")


def get_engine() -> NLUEngine:
    global _ENGINE
    if _ENGINE is None:
        cfg = load_config()
        # API default: hash unless WE_CARE_NLU_MODEL is set, so the service starts without weights.
        import os

        key = os.environ.get("WE_CARE_NLU_MODEL") or cfg.get("embeddings", {}).get("test_model") or "hash"
        _ENGINE = NLUEngine(cfg, model_key=key)
    return _ENGINE


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_kb()
    yield


app = FastAPI(
    title="We Care NLU",
    version=__version__,
    description="Amharic medical understanding and retrieval. No generative decoding.",
    lifespan=lifespan,
)


class IngestBody(BaseModel):
    path: str | None = None
    alignment: dict[str, Any] | None = None
    document_id: str | None = None
    document_version: str = "unknown"
    review_status: str = "unreviewed"


class ResolveBody(BaseModel):
    text: str
    context: dict[str, Any] | None = None


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
    eng = get_engine()
    return {
        "status": "ok",
        "component": 2,
        "generative_decoding": False,
        "chunks": len(eng.retriever.index.chunks),
        "embedder": eng.model_key,
        "translation_backend": getattr(eng.translator, "backend", ""),
        "version": __version__,
    }


@app.get("/models")
def models() -> dict[str, Any]:
    cfg = load_config()
    return {
        "embeddings": cfg.get("embeddings", {}).get("models", {}),
        "active": get_engine().model_key,
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


@app.post("/documents/ingest")
def ingest(body: IngestBody) -> dict[str, Any]:
    eng = get_engine()
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
    return {"ingested_chunks": n, "total_chunks": len(eng.retriever.index.chunks), "review_status": body.review_status}


@app.post("/nlp/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    return get_engine().analyze(req)


@app.post("/nlp/reply", response_model=ReplyResponse)
def reply(req: AnalyzeRequest) -> ReplyResponse:
    return get_engine().reply(req)


@app.post("/retrieval/search")
def search(req: SearchRequest) -> dict[str, Any]:
    eng = get_engine()
    hits = eng.retriever.search(req.query, top_k=req.top_k, filters=req.filters)
    return {"query": req.query, "hits": [h.model_dump() for h in hits]}


@app.post("/nlp/resolve-context")
def resolve_ctx(body: ResolveBody) -> dict[str, Any]:
    from we_care_nlu.contracts.schemas import ContextState

    ctx = ContextState.model_validate(body.context) if body.context else None
    return get_engine().resolve_only(body.text, ctx)
