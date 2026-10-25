"""Batch translation with caching, checkpointing, glossary protection, and OOM recovery."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

from we_care_mt.glossary.manager import Glossary
from we_care_mt.glossary.protect import apply_protection, restore_protection
from we_care_mt.schema import Segment
from we_care_mt.translation.cache import TranslationCache
from we_care_mt.translation.checkpoint import CheckpointStore
from we_care_mt.translation.mock import MockTranslator
from we_care_mt.translation.nllb import NLLBTranslator

logger = logging.getLogger("we_care_mt")

ProgressCb = Callable[[int, int, Segment], None]


def generation_signature(inference: dict[str, Any]) -> str:
    keys = (
        "num_beams",
        "max_new_tokens",
        "max_input_tokens",
        "length_penalty",
        "repetition_penalty",
        "do_sample",
    )
    return "|".join(f"{k}={inference.get(k)}" for k in keys)


class TranslationEngine:
    def __init__(
        self,
        model_key: str,
        model_entry: dict[str, Any],
        inference: dict[str, Any],
        glossary: Glossary | None = None,
        cache_dir: Path | None = None,
        checkpoint_path: Path | None = None,
        hf_cache_dir: Path | None = None,
        backend: str | None = None,
    ):
        self.model_key = model_key
        self.model_entry = model_entry
        self.inference = inference
        self.glossary = glossary
        self.backend_name = backend or ("mock" if model_key == "mock" else "nllb")
        self.hf_id = str(model_entry.get("hf_id") or model_key)
        self.model_id = self.hf_id if self.backend_name != "mock" else "mock"
        self.cache = (
            TranslationCache(cache_dir / "translations.sqlite")
            if cache_dir and inference.get("cache_translations", True)
            else None
        )
        self.checkpoint = CheckpointStore(checkpoint_path) if checkpoint_path else None
        self.hf_cache_dir = str(hf_cache_dir) if hf_cache_dir else None
        self._translator: Any = None
        self.stats: dict[str, Any] = {
            "translated": 0,
            "cached": 0,
            "failed": 0,
            "skipped": 0,
            "seconds": 0.0,
            "peak_vram_gb": None,
        }

    def _get_translator(self):
        if self._translator is not None:
            return self._translator
        if self.backend_name == "mock":
            self._translator = MockTranslator(self.glossary)
            return self._translator
        self._translator = NLLBTranslator(
            hf_id=self.hf_id,
            device=str(self.inference.get("device", "auto")),
            cache_dir=self.hf_cache_dir,
        )
        self._translator.load()
        return self._translator

    def translate_segments(
        self,
        segments: list[Segment],
        progress: ProgressCb | None = None,
    ) -> list[Segment]:
        if self.checkpoint:
            restored = self.checkpoint.apply(segments)
            if restored:
                logger.info("Restored %s segments from checkpoint", restored)

        pending = [s for s in segments if s.is_translatable and s.status in {"pending", "failed"}]
        skipped = [s for s in segments if not s.is_translatable]
        for s in skipped:
            s.status = "skipped"
            if not s.translated_text:
                s.translated_text = s.source_text
        self.stats["skipped"] += len(skipped)

        apply_protect = True
        if self.glossary is None:
            apply_protect = False
        # Config flag lives on the engine via inference extras if provided.
        apply_protect = bool(self.inference.get("apply_glossary_protection", True)) and self.glossary is not None
        enforce = bool(self.inference.get("apply_glossary_enforcement", True)) and self.glossary is not None

        batch_size = int(self.inference.get("batch_size", 4))
        min_batch = int(self.inference.get("min_batch_size", 1))
        ckpt_every = int(self.inference.get("checkpoint_every", 20))
        gen_sig = generation_signature(self.inference)
        t0 = time.perf_counter()
        completed_since_ckpt: list[Segment] = []

        i = 0
        total = len(pending)
        while i < total:
            current_bs = max(min_batch, min(batch_size, total - i))
            batch = pending[i : i + current_bs]
            try:
                self._translate_batch(batch, gen_sig, apply_protect, enforce)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and current_bs > min_batch and self.inference.get("oom_retry", True):
                    logger.warning("GPU OOM at batch_size=%s; retrying smaller batches", current_bs)
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                    batch_size = max(min_batch, current_bs // 2)
                    continue
                for seg in batch:
                    seg.status = "failed"
                    seg.error = str(exc)
                    self.stats["failed"] += 1
                logger.exception("Batch translation failed")
            completed_since_ckpt.extend(batch)
            i += current_bs
            if progress:
                done = i
                progress(done, total, batch[-1])
            if self.checkpoint and len(completed_since_ckpt) >= ckpt_every:
                self.checkpoint.save_many(completed_since_ckpt)
                completed_since_ckpt = []

        if self.checkpoint and completed_since_ckpt:
            self.checkpoint.save_many(completed_since_ckpt)

        self.stats["seconds"] = time.perf_counter() - t0
        tr = self._translator
        if tr is not None and hasattr(tr, "peak_vram_gb"):
            self.stats["peak_vram_gb"] = tr.peak_vram_gb()
        return segments

    def _translate_batch(
        self,
        batch: list[Segment],
        gen_sig: str,
        apply_protect: bool,
        enforce: bool,
    ) -> None:
        translator = self._get_translator()
        to_run: list[tuple[int, str, dict[str, str]]] = []
        outputs: list[str | None] = [None] * len(batch)

        for idx, seg in enumerate(batch):
            source = seg.source_text
            mapping: dict[str, str] = {}
            protected = source
            if apply_protect and self.glossary is not None:
                protected, mapping = apply_protection(source, self.glossary)
            seg.protected = mapping
            if self.glossary:
                seg.glossary_hits = [e.english for e in self.glossary.find_hits(source)]
            if self.cache:
                hit = self.cache.get(self.model_id, protected, gen_sig)
                if hit is not None:
                    outputs[idx] = hit
                    self.stats["cached"] += 1
                    continue
            to_run.append((idx, protected, mapping))

        if to_run:
            texts = [t[1] for t in to_run]
            translated = translator.translate_batch(
                texts,
                max_input_tokens=int(self.inference.get("max_input_tokens", 200)),
                max_new_tokens=int(self.inference.get("max_new_tokens", 256)),
                num_beams=int(self.inference.get("num_beams", 5)),
                length_penalty=float(self.inference.get("length_penalty", 1.0)),
                repetition_penalty=float(self.inference.get("repetition_penalty", 1.05)),
                do_sample=bool(self.inference.get("do_sample", False)),
            )
            if len(translated) != len(to_run):
                raise RuntimeError("Translator returned a different number of hypotheses than sources.")
            for (idx, protected, mapping), hyp in zip(to_run, translated):
                if self.cache:
                    self.cache.put(self.model_id, protected, gen_sig, hyp)
                outputs[idx] = hyp
                self.stats["translated"] += 1

        for seg, hyp in zip(batch, outputs):
            if hyp is None:
                seg.status = "failed"
                seg.error = "missing hypothesis"
                self.stats["failed"] += 1
                continue
            text = restore_protection(hyp, seg.protected)
            if enforce and self.glossary is not None:
                text, notes = self.glossary.enforce_target(seg.source_text, text)
                seg.flags.extend(notes)
            if not text.strip():
                seg.status = "failed"
                seg.error = "empty translation"
                self.stats["failed"] += 1
                continue
            seg.translated_text = text
            seg.model_id = self.model_id
            seg.status = "translated"
