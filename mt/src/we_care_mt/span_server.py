"""HTTP service: translate the one sentence the chat retrieved."""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from we_care_mt.config import AppConfig
from we_care_mt.glossary.manager import Glossary
from we_care_mt.span import SpanTranslator
from we_care_mt.translation.cache import TranslationCache
from we_care_mt.translation.nllb import NLLBTranslator

logger = logging.getLogger("we_care_mt")


class _ReadySpan:
    def __init__(self, cfg: AppConfig, model_key: str):
        self.model_key = model_key
        self.model_id = ""
        self._span: SpanTranslator | None = None
        self._error: str | None = None
        self._ready = threading.Event()
        glossary = None
        gloss_path = cfg.path("glossary")
        if gloss_path.exists():
            freeze = cfg.path("freeze_patterns")
            glossary = Glossary.load(gloss_path, freeze if freeze.exists() else None)
        span_cache = TranslationCache(cfg.path("cache_dir") / f"span.{model_key}.sqlite")
        use_google = model_key in {"google", "google_translate"}

        if use_google:
            from we_care_mt.translation.google import GoogleTranslator

            self.model_id = GoogleTranslator.hf_id
            translator: Any = GoogleTranslator()
            beams = 1
            thread_name = "google-span"
        else:
            entry = cfg.model_entry(model_key)
            self.model_id = str(entry["hf_id"])
            weight_cache = str(cfg.path("hf_cache_dir"))
            translator = NLLBTranslator(hf_id=self.model_id, device="auto", cache_dir=weight_cache)
            beams = 1
            thread_name = "nllb-span"

        def _load() -> None:
            nonlocal beams
            try:
                logger.info("Loading %s for span translation", self.model_id)
                translator.load()
                if not use_google and getattr(translator, "device", "cpu") == "cuda":
                    beams = 4
                self._span = SpanTranslator(
                    translator,
                    glossary,
                    cache=span_cache,
                    model_id=self.model_id,
                    num_beams=beams,
                )
                logger.info("Span translator ready: %s beams=%s", self.model_id, beams)
            except Exception as exc:
                self._error = f"{exc.__class__.__name__}: {exc}"
                logger.exception("Span translator failed to load")
            finally:
                self._ready.set()

        threading.Thread(target=_load, name=thread_name, daemon=True).start()

    @property
    def ready(self) -> bool:
        return self._span is not None

    def wait_translate(self, text: str, direction: str, timeout: float = 600) -> dict[str, Any]:
        if not self._ready.wait(timeout):
            raise TimeoutError("Translator is still loading")
        if self._span is None:
            raise RuntimeError(self._error or "Translator failed to load")
        result = self._span.translate(text, direction)
        result["model"] = self.model_id
        return result


def serve_span(cfg: AppConfig, *, model_key: str, host: str, port: int) -> None:
    holder = _ReadySpan(cfg, model_key)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _send(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/health":
                self._send(404, {"error": "not found"})
                return
            if holder.ready:
                self._send(200, {"status": "ok", "model": holder.model_id})
                return
            self._send(503, {"status": "loading", "model": holder.model_id, "error": holder._error})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/translate":
                self._send(404, {"error": "not found"})
                return
            if not holder.ready:
                self._send(
                    503,
                    {
                        "status": "error" if holder._error else "loading",
                        "error": holder._error or "Translator is still loading",
                        "model": holder.model_id,
                    },
                )
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                result = holder.wait_translate(str(payload.get("text") or ""), str(payload.get("direction") or ""), timeout=0)
            except ValueError as exc:
                self._send(400, {"error": str(exc)})
                return
            except Exception as exc:
                logger.exception("Span translation failed")
                self._send(503, {"error": f"{exc.__class__.__name__}: {exc}"})
                return
            self._send(200, result)

    server = ThreadingHTTPServer((host, port), Handler)
    logger.info("Span translator listening on %s:%s", host, port)
    server.serve_forever()
