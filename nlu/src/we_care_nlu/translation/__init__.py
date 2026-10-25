"""Turn-level translation. The knowledge base stays in English."""

from we_care_nlu.translation.mt_client import MTSpanClient
from we_care_nlu.translation.nllb_turn import NLLBTurnTranslator
from we_care_nlu.translation.phrase_map import PhraseMapTranslator

__all__ = ["MTSpanClient", "NLLBTurnTranslator", "PhraseMapTranslator", "build_translator"]


def build_translator(cfg: dict | None = None, backend: str | None = None):
    import os

    cfg = cfg or {}
    translation = cfg.get("translation") or {}
    chosen = backend or os.environ.get("WE_CARE_TRANSLATION") or translation.get("backend") or "map"
    if chosen == "mt":
        return MTSpanClient(translation.get("mt_url") or os.environ.get("WE_CARE_MT_URL"))
    if chosen == "nllb":
        return NLLBTurnTranslator(
            hf_id=translation.get("model") or "facebook/nllb-200-distilled-600M",
            device=translation.get("device") or "auto",
            num_beams=int(translation.get("num_beams") or 4),
        )
    return PhraseMapTranslator()
