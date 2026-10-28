from __future__ import annotations

import logging
import threading
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

logger = logging.getLogger("we_care_nlu")

_UI = Path(__file__).resolve().parent / "static" / "index.html"
_KB_DIR = Path(__file__).resolve().parents[3] / "data" / "kb"
# Document-first Amharic stack (Docs/ blogs + guideline DOCX, then curated fillers).
_DOCS_KB = _KB_DIR / "docs_amharic_kb.json"
_BILINGUAL_KB = _KB_DIR / "fp_guideline_bilingual.json"
_DEFAULT_KB = _KB_DIR / "fp_user_kb.json"
_ARTICLES_KB = _KB_DIR / "wecare_articles_am.json"
_GUIDELINES_KB = _KB_DIR / "rh_guidelines_en.json"

_ENGINE: NLUEngine | None = None
_READY = threading.Event()
_LOAD_ERROR: str | None = None
_LOAD_LOCK = threading.Lock()


def _ingest_default_kb(eng: NLUEngine) -> None:
    """Load Docs Amharic KB + optional fillers. Skip if already populated."""
    if len(eng.retriever.index.chunks) > 0:
        return
    loaded = False
    # Defer dense rebuild until all sources are loaded (encode once).
    if _DOCS_KB.exists():
        eng.ingest_file(
            str(_DOCS_KB),
            document_version="docs-am",
            default_review_status="source",
            rebuild=False,
        )
        loaded = True
    if _BILINGUAL_KB.exists():
        eng.ingest_file(
            str(_BILINGUAL_KB),
            document_version="2021-06",
            default_review_status="unreviewed",
            rebuild=False,
        )
        loaded = True
    if _DEFAULT_KB.exists():
        eng.ingest_file(
            str(_DEFAULT_KB),
            document_version="demo",
            default_review_status="unreviewed",
            rebuild=False,
        )
        loaded = True
    if _ARTICLES_KB.exists():
        eng.ingest_file(
            str(_ARTICLES_KB),
            document_version="wecare-articles",
            default_review_status="source",
            rebuild=False,
        )
        loaded = True
    if not loaded and _GUIDELINES_KB.exists():
        eng.ingest_file(
            str(_GUIDELINES_KB),
            document_version="source",
            default_review_status="source",
            rebuild=False,
        )
        loaded = True
    if loaded:
        eng.rebuild_index()


def _warm() -> None:
    """Load embedder + KB off the request path so uvicorn can bind immediately."""
    global _ENGINE, _LOAD_ERROR
    import os

    if os.environ.get("PYTEST_CURRENT_TEST"):
        _READY.set()
        return
    try:
        cfg = load_config()
        emb = cfg.get("embeddings") or {}
        key = (
            os.environ.get("WE_CARE_NLU_MODEL")
            or emb.get("default_model")
            or emb.get("test_model")
            or "hash"
        )
        print(f"we-care-nlu: loading embedder={key} (this can take several minutes on first boot)", flush=True)
        try:
            eng = NLUEngine(cfg, model_key=key)
        except Exception as load_exc:
            fallback = emb.get("test_model") or "hash"
            print(
                f"we-care-nlu: embedder={key} failed ({load_exc.__class__.__name__}: {load_exc}); "
                f"falling back to {fallback}",
                flush=True,
            )
            eng = NLUEngine(cfg, model_key=fallback)
        print(f"we-care-nlu: embedder ready ({eng.model_key}); ingesting KB", flush=True)
        _ingest_default_kb(eng)
        with _LOAD_LOCK:
            _ENGINE = eng
            _LOAD_ERROR = None
        print(
            f"we-care-nlu: ready embedder={eng.model_key} chunks={len(eng.retriever.index.chunks)}",
            flush=True,
        )
    except Exception as exc:
        logger.exception("NLU failed to warm up")
        print(f"we-care-nlu: warm-up failed: {exc.__class__.__name__}: {exc}", flush=True)
        with _LOAD_LOCK:
            _LOAD_ERROR = f"{exc.__class__.__name__}: {exc}"
    finally:
        _READY.set()


def get_engine() -> NLUEngine:
    if not _READY.is_set():
        raise HTTPException(503, "NLU is still loading the embedding model and knowledge base.")
    with _LOAD_LOCK:
        if _LOAD_ERROR:
            raise HTTPException(503, f"NLU failed to start: {_LOAD_ERROR}")
        if _ENGINE is None:
            raise HTTPException(503, "NLU engine is not ready.")
        return _ENGINE


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os

    if os.environ.get("PYTEST_CURRENT_TEST"):
        # Tests construct engines directly; keep sync path for local TestClient setups.
        cfg = load_config()
        key = os.environ.get("WE_CARE_NLU_MODEL") or cfg.get("embeddings", {}).get("test_model") or "hash"
        global _ENGINE
        _ENGINE = NLUEngine(cfg, model_key=key)
        _READY.set()
    else:
        threading.Thread(target=_warm, name="nlu-warm", daemon=True).start()
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
    # Never block on model download — Docker and the UI need a live HTTP server.
    if not _READY.is_set():
        return {
            "status": "loading",
            "component": 2,
            "generative_decoding": False,
            "chunks": 0,
            "embedder": "",
            "translation_backend": "",
            "version": __version__,
        }
    with _LOAD_LOCK:
        if _LOAD_ERROR:
            return {
                "status": "error",
                "component": 2,
                "error": _LOAD_ERROR,
                "generative_decoding": False,
                "chunks": 0,
                "embedder": "",
                "translation_backend": "",
                "version": __version__,
            }
        eng = _ENGINE
    if eng is None:
        return {
            "status": "error",
            "component": 2,
            "error": "engine missing",
            "generative_decoding": False,
            "chunks": 0,
            "embedder": "",
            "translation_backend": "",
            "version": __version__,
        }
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
    active = ""
    if _READY.is_set() and _ENGINE is not None:
        active = _ENGINE.model_key
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
