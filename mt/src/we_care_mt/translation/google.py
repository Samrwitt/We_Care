"""Google Translate backend for span testing.

Uses the public web endpoint via deep-translator. This is for quality comparison
only — not clinically validated, and not for production medical traffic.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from we_care_mt.constants import NLLB_SRC_LANG, NLLB_TGT_LANG

logger = logging.getLogger("we_care_mt")

_LANG = {
    NLLB_SRC_LANG: "en",
    NLLB_TGT_LANG: "am",
    "eng_Latn": "en",
    "amh_Ethi": "am",
    "en": "en",
    "am": "am",
}


def _google_web_translate(text: str, source: str, target: str, *, timeout: float = 30) -> str:
    """Unofficial Google Translate web call used only for offline quality tests."""
    import json

    url = (
        "https://translate.googleapis.com/translate_a/single"
        f"?client=gtx&sl={quote(source)}&tl={quote(target)}&dt=t&q={quote(text)}"
    )
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 WeCareMT-Test/1.0"})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 — intentional test call
        payload = json.loads(resp.read().decode("utf-8"))
    chunks = payload[0] if payload else []
    return "".join(part[0] for part in chunks if part and part[0])


class GoogleTranslator:
    """LangTranslator-compatible Google Translate wrapper."""

    model_id = "google_translate_web"
    hf_id = "external:google-translate"

    def __init__(self, pause_s: float = 0.8, retries: int = 4) -> None:
        self.pause_s = pause_s
        self.retries = retries

    def load(self) -> None:
        probe = self._translate_once("ok", "en", "am")
        if not probe:
            raise RuntimeError("Google Translate probe returned empty text")
        logger.info("Google Translate web backend ready (test use only)")

    def unload(self) -> None:
        return

    def _translate_once(self, text: str, source: str, target: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                if self.pause_s:
                    time.sleep(self.pause_s if attempt == 0 else self.pause_s * (attempt + 1))
                return _google_web_translate(text, source, target)
            except Exception as exc:  # noqa: BLE001 — retry network / rate limits
                last_exc = exc
                logger.warning("Google Translate attempt %s failed: %s", attempt + 1, exc)
        raise RuntimeError(f"Google Translate failed after retries: {last_exc}") from last_exc

    def translate_langs(
        self,
        texts: list[str],
        *,
        src_lang: str,
        tgt_lang: str,
        **_kwargs: Any,
    ) -> list[str]:
        source = _LANG.get(src_lang)
        target = _LANG.get(tgt_lang)
        if not source or not target:
            raise ValueError(f"Unsupported language pair: {src_lang} → {tgt_lang}")
        out: list[str] = []
        for text in texts:
            raw = (text or "").strip()
            if not raw:
                out.append("")
                continue
            out.append(self._translate_once(raw, source, target))
        return out
