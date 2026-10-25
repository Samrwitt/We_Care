"""Reproducible model comparison on identical segments and inference settings."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from we_care_mt.config import AppConfig
from we_care_mt.evaluation.metrics import (
    adequacy_proxy,
    corpus_bleu,
    corpus_chrfpp,
    critical_mistranslation_rate,
    ethiopic_fluency_proxy,
    number_accuracy,
    recommend_model,
    terminology_accuracy,
    timed_translate,
)
from we_care_mt.glossary.manager import Glossary
from we_care_mt.translation.mock import MockTranslator
from we_care_mt.translation.nllb import NLLBTranslator

logger = logging.getLogger("we_care_mt")


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def run_benchmark(
    cfg: AppConfig,
    model_keys: list[str],
    eval_path: Path | None = None,
    backend: str | None = None,
    include_google: bool = True,
    out_path: Path | None = None,
) -> dict[str, Any]:
    eval_path = eval_path or cfg.path("eval_set")
    rows = load_eval_set(eval_path)
    sources = [r["source"] for r in rows]
    refs = [r.get("reference") or "" for r in rows]
    has_refs = all(bool(r.strip()) for r in refs)
    reference_type = {r.get("reference_type", "unknown") for r in rows}

    glossary = None
    gloss_path = cfg.path("glossary")
    if gloss_path.exists():
        freeze = cfg.path("freeze_patterns") if "freeze_patterns" in cfg.paths else None
        glossary = Glossary.load(gloss_path, freeze if freeze and freeze.exists() else None)

    inference = cfg.section("inference")
    results: list[dict[str, Any]] = []

    for key in model_keys:
        entry = cfg.model_entry(key) if key != "mock" else {"hf_id": "mock"}
        use_backend = backend or ("mock" if key == "mock" else "nllb")
        logger.info("Benchmarking %s (%s)", key, use_backend)
        try:
            if use_backend == "mock":
                translator = MockTranslator(glossary)

                def _fn(texts, tr=translator):
                    return tr.translate_batch(texts)

                hyps, seconds = timed_translate(_fn, sources)
                vram = None
                hf_id = "mock"
            else:
                nllb = NLLBTranslator(
                    hf_id=str(entry.get("hf_id")),
                    device=str(inference.get("device", "auto")),
                    cache_dir=str(cfg.path("hf_cache_dir")) if cfg.paths.get("hf_cache_dir") else None,
                )
                nllb.load()

                def _fn(texts, tr=nllb, inf=inference):
                    return tr.translate_batch(
                        texts,
                        max_input_tokens=int(inf.get("max_input_tokens", 200)),
                        max_new_tokens=int(inf.get("max_new_tokens", 256)),
                        num_beams=int(inf.get("num_beams", 5)),
                    )

                hyps, seconds = timed_translate(_fn, sources)
                vram = nllb.peak_vram_gb()
                hf_id = nllb.hf_id
                nllb.unload()

            row: dict[str, Any] = {
                "model_key": key,
                "hf_id": hf_id,
                "status": "ok",
                "n_segments": len(sources),
                "inference": {
                    "num_beams": inference.get("num_beams"),
                    "max_input_tokens": inference.get("max_input_tokens"),
                    "max_new_tokens": inference.get("max_new_tokens"),
                    "device": inference.get("device"),
                },
                "seconds": round(seconds, 3),
                "sentences_per_second": round(len(sources) / seconds, 3) if seconds else None,
                "peak_vram_gb": None if vram is None else round(vram, 3),
                "numerical_dosage_accuracy": number_accuracy(hyps, sources),
                "ethiopic_fluency_proxy": ethiopic_fluency_proxy(hyps),
                "adequacy_proxy": adequacy_proxy(hyps, sources, glossary),
                "medical_terminology_accuracy": terminology_accuracy(hyps, sources, glossary),
                "critical_mistranslation_rate": critical_mistranslation_rate(hyps, sources),
                "hypotheses": hyps,
            }
            if has_refs:
                row["bleu"] = corpus_bleu(hyps, refs)
                row["chrfpp"] = corpus_chrfpp(hyps, refs)
            else:
                row["bleu"] = None
                row["chrfpp"] = None
                row["note"] = "No complete reference translations; BLEU/chrF++ not computed."
            results.append(row)
        except Exception as exc:
            logger.exception("Benchmark failed for %s", key)
            results.append(
                {
                    "model_key": key,
                    "hf_id": entry.get("hf_id"),
                    "status": "error",
                    "error": str(exc),
                    "note": "Model weights were not downloaded or failed to load. No scores were fabricated.",
                }
            )

    google_path = cfg.path("google_translate_baseline") if "google_translate_baseline" in cfg.paths else None
    google_row = None
    if include_google and google_path and google_path.exists():
        google_hyps = _load_google(google_path, rows)
        if google_hyps:
            google_row = {
                "model_key": "google_translate_external",
                "hf_id": "external:google-translate",
                "status": "ok",
                "note": "Scores use user-supplied Google Translate outputs. This pipeline does not call Google.",
                "n_segments": len(sources),
                "numerical_dosage_accuracy": number_accuracy(google_hyps, sources),
                "ethiopic_fluency_proxy": ethiopic_fluency_proxy(google_hyps),
                "adequacy_proxy": adequacy_proxy(google_hyps, sources, glossary),
                "medical_terminology_accuracy": terminology_accuracy(google_hyps, sources, glossary),
                "critical_mistranslation_rate": critical_mistranslation_rate(google_hyps, sources),
                "bleu": corpus_bleu(google_hyps, refs) if has_refs else None,
                "chrfpp": corpus_chrfpp(google_hyps, refs) if has_refs else None,
            }
            results.append(google_row)

    report = {
        "eval_set": str(eval_path),
        "n_items": len(rows),
        "has_human_references": has_refs,
        "reference_types": sorted(reference_type),
        "reference_disclaimer": (
            "Developer-draft references are not gold clinical translations. "
            "Blind bilingual medical review is required before trusting ranking."
        ),
        "inference_held_constant": {
            "num_beams": inference.get("num_beams"),
            "max_input_tokens": inference.get("max_input_tokens"),
            "max_new_tokens": inference.get("max_new_tokens"),
        },
        "models": results,
        "recommendation": recommend_model(results),
        "clinical_disclaimer": (
            "None of these models is clinically validated. Automated metrics, including BLEU, "
            "must not be used as a safety decision."
        ),
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Hypotheses can be large; keep them but write UTF-8 JSON.
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def _load_google(path: Path, eval_rows: list[dict[str, Any]]) -> list[str] | None:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            mapping[rec.get("id") or rec.get("source")] = rec.get("google_translate") or rec.get("translation") or ""
    hyps = []
    for row in eval_rows:
        key = row.get("id") or row.get("source")
        if key not in mapping:
            return None
        hyps.append(mapping[key])
    return hyps
