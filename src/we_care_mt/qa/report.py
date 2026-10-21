"""Write QA reports as JSON, Markdown, and a printable PDF."""

from __future__ import annotations

import json
from pathlib import Path

from we_care_mt.output.fonts import FontBundle, register_reportlab_fonts
from we_care_mt.schema import QAReport, to_plain


def write_qa_reports(report: QAReport, out_dir: Path, fonts: FontBundle | None = None) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe(report.document_name)
    json_path = out_dir / f"{stem}.qa.json"
    md_path = out_dir / f"{stem}.qa.md"
    pdf_path = out_dir / f"{stem}.qa.pdf"

    json_path.write_text(json.dumps(to_plain(report), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")
    _to_pdf(report, pdf_path, fonts)
    return {"json": json_path, "markdown": md_path, "pdf": pdf_path}


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in Path(name).stem)[:80]


def _to_markdown(report: QAReport) -> str:
    lines = [
        f"# Translation quality report: {report.document_name}",
        "",
        f"- Model: `{report.model_id}`",
        f"- Generated: {report.generated_at}",
        f"- Segments: {report.n_segments}",
        f"- Findings: {report.n_findings} (critical {report.n_critical}, major {report.n_major})",
        "",
        f"**{report.medically_safe_claim}**",
        "",
        report.summary,
        "",
        "| Severity | Page | Check | Segment | Issue | Recommended action |",
        "|---|---|---|---|---|---|",
    ]
    for f in report.findings:
        msg = f.message.replace("|", "/").replace("\n", " ")
        action = f.recommended_action.replace("|", "/")
        lines.append(
            f"| {f.severity} | {f.page_number} | {f.check} | `{f.segment_id}` | {msg} | {action} |"
        )
    if not report.findings:
        lines.append("| — | — | — | — | No automated findings. Human review is still required. | Review full document. |")
    lines.append("")
    return "\n".join(lines)


def _to_pdf(report: QAReport, path: Path, fonts: FontBundle | None) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    names = register_reportlab_fonts(fonts)
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "QATitle",
        parent=styles["Title"],
        fontName=names.latin_bold,
        fontSize=14,
        leading=18,
    )
    body = ParagraphStyle(
        "QABody",
        parent=styles["BodyText"],
        fontName=names.amharic,
        fontSize=9,
        leading=12,
    )
    latin = ParagraphStyle(
        "QALatin",
        parent=styles["BodyText"],
        fontName=names.latin,
        fontSize=9,
        leading=12,
    )
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm)
    story = [
        Paragraph("Translation quality report", title),
        Paragraph(f"Document: {report.document_name}<br/>Model: {report.model_id}", latin),
        Spacer(1, 6),
        Paragraph(report.medically_safe_claim, body),
        Spacer(1, 6),
        Paragraph(report.summary, latin),
        Spacer(1, 8),
    ]
    for f in report.findings[:400]:
        story.append(
            Paragraph(
                f"<b>[{f.severity.upper()}] p.{f.page_number} {f.check}</b> — { _esc(f.message) }",
                latin,
            )
        )
        if f.source_excerpt:
            story.append(Paragraph(f"EN: {_esc(f.source_excerpt)}", latin))
        if f.target_excerpt:
            story.append(Paragraph(f"AM: {_esc(f.target_excerpt)}", body))
        story.append(Paragraph(f"Action: {_esc(f.recommended_action)}", latin))
        story.append(Spacer(1, 4))
    if not report.findings:
        story.append(Paragraph("No automated findings. Human bilingual medical review is still required.", latin))
    doc.build(story)


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
