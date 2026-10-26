"""Hugging Face NLLB / AfriScience-MT inference wrapper."""

from __future__ import annotations

import logging
import os
from typing import Any

from we_care_mt.constants import NLLB_SRC_LANG, NLLB_TGT_LANG

logger = logging.getLogger("we_care_mt")


def resolve_device(requested: str) -> str:
    requested = (requested or "auto").lower()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for neural translation.") from exc
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU.")
            return "cpu"
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def lang_token_id(tokenizer, lang: str) -> int:
    mapping = getattr(tokenizer, "lang_code_to_id", None)
    if isinstance(mapping, dict) and lang in mapping:
        return int(mapping[lang])
    # transformers >= 4.41 sometimes uses convert_tokens_to_ids only
    tid = tokenizer.convert_tokens_to_ids(lang)
    unk = getattr(tokenizer, "unk_token_id", None)
    if tid is None or (unk is not None and tid == unk):
        raise ValueError(
            f"Language code {lang!r} is not in the tokenizer vocabulary. "
            "NLLB models require codes such as eng_Latn and amh_Ethi."
        )
    return int(tid)


class NLLBTranslator:
    def __init__(
        self,
        hf_id: str,
        device: str = "auto",
        src_lang: str = NLLB_SRC_LANG,
        tgt_lang: str = NLLB_TGT_LANG,
        cache_dir: str | None = None,
    ):
        self.hf_id = hf_id
        self.model_id = hf_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.device = resolve_device(device)
        self.cache_dir = cache_dir
        self._model = None
        self._tokenizer = None
        self._tgt_id: int | None = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading tokenizer %s", self.hf_id)
        kwargs: dict[str, Any] = {}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        tokenizer = AutoTokenizer.from_pretrained(self.hf_id, **kwargs)
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = self.src_lang
        logger.info("Loading model %s on %s", self.hf_id, self.device)
        model_kwargs = dict(kwargs)
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        else:
            # Cap threads so chat latency stays stable under concurrent load.
            requested = int(os.environ.get("WE_CARE_TORCH_THREADS") or 0)
            cores = os.cpu_count() or 1
            threads = requested if requested > 0 else min(4, cores)
            torch.set_num_threads(max(1, threads))
            logger.info("CPU inference threads=%s", torch.get_num_threads())
        model = AutoModelForSeq2SeqLM.from_pretrained(self.hf_id, **model_kwargs)
        model.to(self.device)
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        self._tgt_id = lang_token_id(tokenizer, self.tgt_lang)

    def unload(self) -> None:
        import torch

        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def translate_batch(
        self,
        texts: list[str],
        *,
        max_input_tokens: int = 200,
        max_new_tokens: int = 256,
        num_beams: int = 5,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.05,
        do_sample: bool = False,
    ) -> list[str]:
        return self.translate_langs(
            texts,
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )

    def translate_langs(
        self,
        texts: list[str],
        *,
        src_lang: str,
        tgt_lang: str,
        max_input_tokens: int = 200,
        max_new_tokens: int = 256,
        num_beams: int = 5,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.05,
        do_sample: bool = False,
    ) -> list[str]:
        """Translate with one loaded model. src_lang and tgt_lang are NLLB codes."""
        if self._model is None or self._tokenizer is None:
            self.load()
        assert self._model is not None and self._tokenizer is not None
        import torch

        tokenizer = self._tokenizer
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = src_lang
        tgt_id = lang_token_id(tokenizer, tgt_lang)
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        gen_kwargs: dict[str, Any] = {
            "forced_bos_token_id": tgt_id,
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
        }
        if num_beams > 1:
            gen_kwargs["early_stopping"] = True
        with torch.inference_mode():
            outputs = self._model.generate(**encoded, **gen_kwargs)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [t.strip() for t in decoded]

    def peak_vram_gb(self) -> float | None:
        try:
            import torch
        except ImportError:
            return None
        if self.device != "cuda" or not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
