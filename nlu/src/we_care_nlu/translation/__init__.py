"""Phrase map, MT client, reply composition helpers, and no-KB advice templates."""

from we_care_nlu.translation.mt_client import MTSpanClient
from we_care_nlu.translation.phrase_map import PhraseMapTranslator

__all__ = ["MTSpanClient", "PhraseMapTranslator", "build_translator"]


def build_translator(cfg: dict | None = None, backend: str | None = None):
    import os

    cfg = cfg or {}
    translation = cfg.get("translation") or {}
    chosen = backend or os.environ.get("WE_CARE_TRANSLATION") or translation.get("backend") or "map"
    if chosen == "mt":
        return MTSpanClient(translation.get("mt_url") or os.environ.get("WE_CARE_MT_URL"))
    return PhraseMapTranslator()
