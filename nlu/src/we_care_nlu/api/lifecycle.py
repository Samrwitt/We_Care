"""Engine startup and shared runtime for the FastAPI app.

At process start we load the embedder + knowledge base in a background thread
so HTTP can bind immediately (`/health` returns status=loading until ready).

Tests can call `runtime.install_for_tests(engine)` to skip warm-up.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from we_care_nlu.config import load_config
from we_care_nlu.pipeline import NLUEngine

logger = logging.getLogger("we_care_nlu")

# KB files (repo nlu/data/kb). Loaded in order when present.
_KB_DIR = Path(__file__).resolve().parents[3] / "data" / "kb"
_KB_SOURCES: tuple[tuple[Path, str, str], ...] = (
    # path, document_version, review_status
    (_KB_DIR / "docs_amharic_kb.json", "docs-am", "source"),
    (_KB_DIR / "fp_guideline_bilingual.json", "2021-06", "unreviewed"),
    (_KB_DIR / "fp_user_kb.json", "demo", "unreviewed"),
    (_KB_DIR / "wecare_articles_am.json", "wecare-articles", "source"),
)
_GUIDELINES_FALLBACK = (_KB_DIR / "rh_guidelines_en.json", "source", "source")


class EngineRuntime:
    """Process-wide NLUEngine holder (one engine per server process)."""

    def __init__(self) -> None:
        self.engine: NLUEngine | None = None
        self.load_error: str | None = None
        self.ready = threading.Event()
        self._lock = threading.Lock()

    def get(self) -> NLUEngine:
        """Return the ready engine or raise HTTP 503."""
        if not self.ready.is_set():
            raise HTTPException(503, "NLU is still loading the embedding model and knowledge base.")
        with self._lock:
            if self.load_error:
                raise HTTPException(503, f"NLU failed to start: {self.load_error}")
            if self.engine is None:
                raise HTTPException(503, "NLU engine is not ready.")
            return self.engine

    def install_for_tests(self, engine: NLUEngine) -> None:
        """Swap in a test engine and mark ready (used by TestClient fixtures)."""
        with self._lock:
            self.engine = engine
            self.load_error = None
            self.ready.set()

    def ingest_default_kb(self, eng: NLUEngine) -> None:
        """Load Docs Amharic KB + optional fillers. Skip if already populated."""
        if len(eng.retriever.index.chunks) > 0:
            return
        loaded = False
        for path, version, review in _KB_SOURCES:
            if not path.exists():
                continue
            eng.ingest_file(
                str(path),
                document_version=version,
                default_review_status=review,
                rebuild=False,
            )
            loaded = True
        if not loaded:
            path, version, review = _GUIDELINES_FALLBACK
            if path.exists():
                eng.ingest_file(
                    str(path),
                    document_version=version,
                    default_review_status=review,
                    rebuild=False,
                )
                loaded = True
        if loaded:
            eng.rebuild_index()

    def warm_in_background(self) -> None:
        """Start embedder+KB load on a daemon thread."""
        threading.Thread(target=self._warm, name="nlu-warm", daemon=True).start()

    def warm_sync_for_tests(self) -> None:
        """Build a hash engine inline when PYTEST_CURRENT_TEST is set."""
        cfg = load_config()
        key = os.environ.get("WE_CARE_NLU_MODEL") or cfg.get("embeddings", {}).get("test_model") or "hash"
        with self._lock:
            self.engine = NLUEngine(cfg, model_key=key)
            self.load_error = None
            self.ready.set()

    def _warm(self) -> None:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            self.ready.set()
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
            print(
                f"we-care-nlu: loading embedder={key} (this can take several minutes on first boot)",
                flush=True,
            )
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
            self.ingest_default_kb(eng)
            with self._lock:
                self.engine = eng
                self.load_error = None
            print(
                f"we-care-nlu: ready embedder={eng.model_key} chunks={len(eng.retriever.index.chunks)}",
                flush=True,
            )
        except Exception as exc:
            logger.exception("NLU failed to warm up")
            print(f"we-care-nlu: warm-up failed: {exc.__class__.__name__}: {exc}", flush=True)
            with self._lock:
                self.load_error = f"{exc.__class__.__name__}: {exc}"
        finally:
            self.ready.set()

    def health_payload(self, *, version: str) -> dict[str, Any]:
        """JSON for GET /health (never blocks on model download)."""
        base = {
            "component": 2,
            "generative_decoding": False,
            "version": version,
        }
        if not self.ready.is_set():
            return {
                **base,
                "status": "loading",
                "chunks": 0,
                "embedder": "",
                "translation_backend": "",
            }
        with self._lock:
            if self.load_error:
                return {
                    **base,
                    "status": "error",
                    "error": self.load_error,
                    "chunks": 0,
                    "embedder": "",
                    "translation_backend": "",
                }
            eng = self.engine
        if eng is None:
            return {
                **base,
                "status": "error",
                "error": "engine missing",
                "chunks": 0,
                "embedder": "",
                "translation_backend": "",
            }
        return {
            **base,
            "status": "ok",
            "chunks": len(eng.retriever.index.chunks),
            "embedder": eng.model_key,
            "translation_backend": getattr(eng.translator, "backend", ""),
        }


# Singleton used by app.py
runtime = EngineRuntime()
