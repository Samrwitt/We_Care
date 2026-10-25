"""Ask the PDF translator to translate one retrieved sentence."""

from __future__ import annotations

import os
from typing import Any

import httpx


class TranslatorNotReady(RuntimeError):
    """The PDF translator has not finished loading Facebook NLLB."""


class MTSpanClient:
    """Chat lookup stays here. English↔Amharic for that turn goes to we-care-mt."""

    can_translate_reply = True
    backend = "mt"

    def __init__(self, base_url: str | None = None, client: httpx.Client | None = None):
        self.base_url = (base_url or os.environ.get("WE_CARE_MT_URL") or "http://127.0.0.1:8090").rstrip("/")
        self._client = client

    def to_english(self, text: str) -> str:
        return self._call(text, "am-en")

    def to_amharic(self, text: str) -> str:
        return self._call(text, "en-am")

    def _call(self, text: str, direction: str) -> str:
        payload = {"text": text, "direction": direction}
        try:
            if self._client is not None:
                response = self._client.post(f"{self.base_url}/translate", json=payload)
            else:
                response = httpx.post(f"{self.base_url}/translate", json=payload, timeout=120)
        except httpx.HTTPError as exc:
            raise TranslatorNotReady("The PDF translator did not answer.") from exc
        if response.status_code == 503:
            raise TranslatorNotReady("Facebook NLLB is still loading")
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        return str(body.get("text") or "")
