"""End-to-end document pipeline: extract → segment → translate → QA → write outputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from we_care_mt.config import AppConfig
from we_care_mt.extraction.pdf_extractor import extract_pdf
from we_care_mt.glossary.manager import Glossary
from we_care_mt.output.alignment import write_alignment
from we_care_mt.output.bilingual import write_bilingual_pdf
from we_care_mt.output.docx_writer import write_docx
from we_care_mt.output.fonts import locate_fonts
from we_care_mt.output.pdf_writer import write_amharic_pdf
from we_care_mt.qa.checks import run_qa
from we_care_mt.qa.report import write_qa_reports
from we_care_mt.schema import to_plain
from we_care_mt.segmentation.chunker import segments_from_document
from we_care_mt.translation.engine import TranslationEngine

logger = logging.getLogger("we_care_mt")


def process_pdf(
    pdf_path: Path,
    cfg: AppConfig,
    model_key: str | None = None,
    backend: str | None = None,
    output_dir: Path | None = None,
    progress: Callable[[int, int, Any], None] | None = None,
) -> dict[str, Any]:
    pdf_path = Path(pdf_path)
    model_key = model_key or cfg.default_model_key()
    if backend is None and model_key == "mock":
        backend = "mock"
    out_root = Path(output_dir or cfg.path("output_dir"))
    dest = out_root / pdf_path.stem
    dest.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting %s", pdf_path)
    structure = extract_pdf(pdf_path, cfg.section("extraction"))
    (dest / f"{pdf_path.stem}.structure.json").write_text(
        json.dumps(to_plain(structure), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("Segmenting %s (%s blocks)", pdf_path.name, len(structure.blocks))
    segs = segments_from_document(structure, cfg.section("segmentation"))

    glossary = None
    gloss_path = cfg.path("glossary")
    if gloss_path.exists():
        freeze = cfg.path("freeze_patterns")
        glossary = Glossary.load(gloss_path, freeze if freeze.exists() else None)

    inference = cfg.section("inference")
    gloss_cfg = cfg.section("glossary")
    inference = {
        **inference,
        "apply_glossary_protection": gloss_cfg.get("apply_pre_translation_protection", True),
        "apply_glossary_enforcement": gloss_cfg.get("apply_post_translation_enforcement", True),
    }
    ckpt = cfg.path("checkpoint_dir") / f"{pdf_path.stem}.{model_key}.jsonl"
    engine = TranslationEngine(
        model_key=model_key,
        model_entry=cfg.model_entry(model_key) if model_key != "mock" else {"hf_id": "mock"},
        inference=inference,
        glossary=glossary,
        cache_dir=cfg.path("cache_dir"),
        checkpoint_path=ckpt,
        hf_cache_dir=cfg.path("hf_cache_dir") if cfg.paths.get("hf_cache_dir") else None,
        backend=backend,
    )
    logger.info("Translating %s segments with %s", len(segs), engine.model_id)
    segs = engine.translate_segments(segs, progress=progress)

    qa = run_qa(
        segs,
        document_name=structure.document_name,
        model_id=engine.model_id,
        glossary=glossary,
        config=cfg.section("qa"),
    )
    fonts = locate_fonts(cfg.path("fonts_dir"))
    out_cfg = cfg.section("output")
    artifacts: dict[str, str] = {}

    if out_cfg.get("generate_alignment_json", True) or out_cfg.get("generate_alignment_csv", True):
        align = write_alignment(structure, segs, dest)
        artifacts.update({k: str(v) for k, v in align.items()})
    if out_cfg.get("generate_qa_report", True):
        qa_paths = write_qa_reports(qa, dest, fonts)
        artifacts.update({f"qa_{k}": str(v) for k, v in qa_paths.items()})
    if out_cfg.get("generate_amharic_pdf", True):
        try:
            p = write_amharic_pdf(structure, segs, dest / f"{pdf_path.stem}.am.pdf", fonts, title=structure.title)
            artifacts["amharic_pdf"] = str(p)
        except Exception as exc:
            logger.exception("Amharic PDF layout failed; other artifacts still written")
            artifacts["amharic_pdf_error"] = str(exc)
    if out_cfg.get("generate_bilingual_pdf", True):
        try:
            p = write_bilingual_pdf(structure, segs, dest / f"{pdf_path.stem}.bilingual.pdf", fonts)
            artifacts["bilingual_pdf"] = str(p)
        except Exception as exc:
            logger.exception("Bilingual PDF layout failed")
            artifacts["bilingual_pdf_error"] = str(exc)
    if out_cfg.get("generate_docx", True):
        try:
            p = write_docx(structure, segs, dest / f"{pdf_path.stem}.docx", fonts)
            artifacts["docx"] = str(p)
        except Exception as exc:
            logger.exception("DOCX write failed")
            artifacts["docx_error"] = str(exc)

    summary = {
        "document": structure.document_name,
        "pages": structure.page_count,
        "segments": len(segs),
        "model_id": engine.model_id,
        "model_key": model_key,
        "backend": engine.backend_name,
        "stats": engine.stats,
        "qa": {
            "n_findings": qa.n_findings,
            "n_critical": qa.n_critical,
            "n_major": qa.n_major,
            "medically_safe_claim": qa.medically_safe_claim,
        },
        "extraction_warnings": len(structure.warnings),
        "complex_layout_pages": structure.complex_layout_pages,
        "artifacts": artifacts,
        "output_dir": str(dest),
    }
    (dest / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("Finished %s → %s", pdf_path.name, dest)
    return summary


def process_many(
    inputs: list[Path],
    cfg: AppConfig,
    model_key: str | None = None,
    backend: str | None = None,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    results = []
    for path in inputs:
        try:
            results.append(process_pdf(path, cfg, model_key=model_key, backend=backend, output_dir=output_dir))
        except Exception as exc:
            logger.exception("Failed on %s", path)
            results.append({"document": str(path), "status": "error", "error": str(exc)})
    return results


def collect_pdfs(input_path: Path) -> list[Path]:
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*.pdf") if p.is_file())
