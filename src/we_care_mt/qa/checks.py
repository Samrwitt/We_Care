"""Automated medical translation QA. Passing checks ≠ medically safe."""

from __future__ import annotations

import re
from typing import Any

from we_care_mt.constants import (
    CONTRAINDICATION_CUES_AM,
    CONTRAINDICATION_CUES_EN,
    NEGATION_CUES_AM,
    NEGATION_CUES_EN,
    WARNING_CUES_AM,
    WARNING_CUES_EN,
)
from we_care_mt.glossary.manager import Glossary
from we_care_mt.schema import QAFinding, QAReport, Segment

_NUMBER_RE = re.compile(
    r"(?<![A-Za-z])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*(?:mg|mcg|µg|g|kg|ml|mL|L|iu|IU|%|/100,?000|/1,?000))?",
    re.I,
)
_UNIT_RE = re.compile(r"\b(?:mg|mcg|µg|g|kg|ml|mL|L|iu|IU|mmHg|mmol|mEq)\b")
_LATIN_WORD = re.compile(r"[A-Za-z]{3,}")
_ETHIOPIC = re.compile(r"[\u1200-\u137F]")


def extract_numbers(text: str) -> list[str]:
    return [m.group(0).replace(" ", "").lower() for m in _NUMBER_RE.finditer(text)]


def _has_any(text: str, cues: set[str]) -> bool:
    low = text.lower()
    return any(c.lower() in low for c in cues)


def run_qa(
    segments: list[Segment],
    document_name: str,
    model_id: str,
    glossary: Glossary | None = None,
    config: dict[str, Any] | None = None,
) -> QAReport:
    cfg = config or {}
    latin_ratio_max = float(cfg.get("untranslated_latin_ratio", 0.55))
    ratio_min = float(cfg.get("length_ratio_min", 0.25))
    ratio_max = float(cfg.get("length_ratio_max", 4.0))
    findings: list[QAFinding] = []

    translated = [s for s in segments if s.kind not in {"header", "footer", "page_number"}]
    source_pages = {s.page_number for s in translated}
    target_pages = {s.page_number for s in translated if s.translated_text.strip()}
    missing_pages = source_pages - target_pages
    for page in sorted(missing_pages):
        findings.append(
            QAFinding(
                segment_id="",
                page_number=page,
                check="structural_omission",
                severity="major",
                message=f"No translated text recorded for source page {page}.",
                recommended_action="Compare the page against the source PDF; restore omitted blocks.",
            )
        )

    for seg in translated:
        src = seg.source_text or ""
        tgt = seg.translated_text or ""
        page = seg.page_number
        sid = seg.segment_id

        if seg.status == "failed" or (seg.is_translatable and not tgt.strip()):
            findings.append(
                QAFinding(
                    sid, page, "missing_translation", "critical",
                    "Segment failed or produced an empty translation.",
                    src[:240], tgt[:240],
                    "Retranslate this segment and review the source sentence.",
                )
            )
            continue

        if not seg.is_translatable:
            continue

        src_nums = extract_numbers(src)
        tgt_nums = extract_numbers(tgt)
        if src_nums != tgt_nums:
            # Extra or missing numbers, including dosage quantities.
            severity = "critical" if _looks_like_dose(src) or _looks_like_dose(tgt) else "major"
            findings.append(
                QAFinding(
                    sid, page, "number_mismatch", severity,
                    f"Numbers differ. source={src_nums} target={tgt_nums}",
                    src[:240], tgt[:240],
                    "Restore original numerals, units, ratios, and percentages exactly.",
                )
            )

        src_units = {m.group(0) for m in _UNIT_RE.finditer(src)}
        tgt_units = {m.group(0) for m in _UNIT_RE.finditer(tgt)}
        if src_units - tgt_units:
            findings.append(
                QAFinding(
                    sid, page, "unit_mismatch", "critical",
                    f"Dosage/measurement units missing in target: {sorted(src_units - tgt_units)}",
                    src[:240], tgt[:240],
                    "Keep clinical units identical to the English source (mg, mL, IU, %).",
                )
            )

        src_neg = _has_any(src, NEGATION_CUES_EN)
        tgt_neg = _has_any(tgt, NEGATION_CUES_AM) or _has_any(tgt, NEGATION_CUES_EN)
        if src_neg and not tgt_neg:
            findings.append(
                QAFinding(
                    sid, page, "negation_error", "critical",
                    "Source contains negation that was not detected in the Amharic output.",
                    src[:240], tgt[:240],
                    "Confirm that prohibitions and 'do not' / 'not indicated' meaning is preserved.",
                )
            )

        src_warn = _has_any(src, WARNING_CUES_EN)
        tgt_warn = _has_any(tgt, WARNING_CUES_AM) or _has_any(tgt, WARNING_CUES_EN)
        if src_warn and not tgt_warn:
            findings.append(
                QAFinding(
                    sid, page, "missing_warning", "critical",
                    "Source warning/caution language is not clearly present in the translation.",
                    src[:240], tgt[:240],
                    "Restore warning meaning; do not soften clinical cautions.",
                )
            )

        src_ci = _has_any(src, CONTRAINDICATION_CUES_EN)
        tgt_ci = _has_any(tgt, CONTRAINDICATION_CUES_AM) or _has_any(tgt, CONTRAINDICATION_CUES_EN)
        if src_ci and not tgt_ci:
            findings.append(
                QAFinding(
                    sid, page, "missing_contraindication", "critical",
                    "Contraindication language may be missing from the translation.",
                    src[:240], tgt[:240],
                    "Verify contraindication wording with a bilingual clinician.",
                )
            )

        if re.search(r"\bpregnan|\bpostpartum|\bantenatal|\btrimester|\blactat|\bbreastfeed", src, re.I):
            if not (
                _ETHIOPIC.search(tgt)
                or re.search(r"እርግዝ|ነፍሰ ጡር|ወሊድ|ጡት|ቅድመ ወሊድ|ድህረ", tgt)
                or re.search(r"pregnan|postpartum|antenatal|breast", tgt, re.I)
            ):
                findings.append(
                    QAFinding(
                        sid, page, "pregnancy_reference", "major",
                        "Pregnancy / postpartum / lactation reference may be missing.",
                        src[:240], tgt[:240],
                        "Confirm age, pregnancy, and treatment-duration references.",
                    )
                )

        latin = _LATIN_WORD.findall(tgt)
        ethiopic = _ETHIOPIC.findall(tgt)
        if ethiopic:
            ratio = len("".join(latin)) / max(len(tgt), 1)
            if ratio >= latin_ratio_max and len("".join(latin)) >= 20:
                findings.append(
                    QAFinding(
                        sid, page, "untranslated_english", "major",
                        "High leftover English content in the Amharic segment.",
                        src[:240], tgt[:240],
                        "Retranslate leftover English except protected drug names and abbreviations.",
                    )
                )
        elif latin and seg.is_translatable:
            findings.append(
                QAFinding(
                    sid, page, "missing_amharic", "critical",
                    "No Ethiopic characters in a translatable segment.",
                    src[:240], tgt[:240],
                    "Retranslate; confirm the target language token is amh_Ethi.",
                )
            )

        if src and tgt:
            ratio = len(tgt) / max(len(src), 1)
            if ratio < ratio_min or ratio > ratio_max:
                findings.append(
                    QAFinding(
                        sid, page, "length_anomaly", "minor",
                        f"Target/source length ratio {ratio:.2f} is outside [{ratio_min}, {ratio_max}].",
                        src[:240], tgt[:240],
                        "Check for dropped clauses or hallucinated extra sentences.",
                    )
                )

        extra_nums = set(tgt_nums) - set(src_nums)
        if extra_nums:
            findings.append(
                QAFinding(
                    sid, page, "hallucinated_number", "critical",
                    f"Target contains numbers absent from source: {sorted(extra_nums)}",
                    src[:240], tgt[:240],
                    "Remove unsupported numerals; never invent dosages.",
                )
            )

        if glossary:
            for entry in glossary.find_hits(src):
                if entry.confidence == "provisional":
                    findings.append(
                        QAFinding(
                            sid, page, "uncertain_terminology", "minor",
                            f"Provisional glossary term {entry.english!r} — do not treat the Amharic as verified.",
                            src[:240], tgt[:240],
                            "Have a bilingual clinician confirm this term.",
                        )
                    )
                if entry.confidence in {"verified", "standard"} and entry.amharic:
                    if entry.amharic not in tgt and entry.english.lower() not in tgt.lower():
                        findings.append(
                            QAFinding(
                                sid, page, "inconsistent_terminology", "major",
                                f"Expected glossary rendering {entry.amharic!r} for {entry.english!r} was not found.",
                                src[:240], tgt[:240],
                                "Align with the verified Ethiopian medical glossary term.",
                            )
                        )

        if "uncertain_extraction" in seg.flags or "uncertain_table" in seg.flags:
            findings.append(
                QAFinding(
                    sid, page, "extraction_uncertainty", "major",
                    "Source extraction was flagged as uncertain (complex layout or table).",
                    src[:240], tgt[:240],
                    "Compare against the original PDF page before accepting the translation.",
                )
            )

    n_crit = sum(1 for f in findings if f.severity == "critical")
    n_maj = sum(1 for f in findings if f.severity == "major")
    summary = (
        f"{len(findings)} automated findings ({n_crit} critical, {n_maj} major) "
        f"across {len(translated)} segments. "
        "Automated checks do not classify the translation as medically safe."
    )
    return QAReport(
        document_name=document_name,
        model_id=model_id,
        n_segments=len(translated),
        n_findings=len(findings),
        n_critical=n_crit,
        n_major=n_maj,
        findings=findings,
        summary=summary,
    )


def _looks_like_dose(text: str) -> bool:
    return bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ml|mL|iu|IU|%)\b", text))
