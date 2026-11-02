"""SacreBLEU / chrF++ plus medical heuristic metrics. No fabricated scores."""

from __future__ import annotations

import re
import time
from typing import Any

from we_care_mt.glossary.manager import Glossary
from we_care_mt.qa.checks import extract_numbers


def corpus_bleu(hyps: list[str], refs: list[str]) -> float:
    import sacrebleu

    if not hyps:
        return 0.0
    return float(sacrebleu.corpus_bleu(hyps, [refs]).score)


def corpus_chrfpp(hyps: list[str], refs: list[str]) -> float:
    import sacrebleu

    if not hyps:
        return 0.0
    # word_order=2 → chrF++
    return float(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score)


def number_accuracy(hyps: list[str], srcs: list[str]) -> float:
    if not srcs:
        return 1.0
    ok = 0
    for hyp, src in zip(hyps, srcs):
        if extract_numbers(src) == extract_numbers(hyp):
            ok += 1
    return ok / len(srcs)


def terminology_accuracy(hyps: list[str], srcs: list[str], glossary: Glossary | None) -> float:
    if glossary is None:
        return float("nan")
    total = 0
    ok = 0
    for hyp, src in zip(hyps, srcs):
        hits = [e for e in glossary.find_hits(src) if e.confidence in {"verified", "standard"} and e.amharic]
        for entry in hits:
            total += 1
            if entry.amharic in hyp or entry.preserve_source:
                ok += 1
    if total == 0:
        return float("nan")
    return ok / total


def ethiopic_fluency_proxy(hyps: list[str]) -> float:
    """Character-level Ethiopic ratio. Proxy only — not a human fluency score."""
    if not hyps:
        return 0.0
    scores = []
    for hyp in hyps:
        letters = [ch for ch in hyp if ch.isalpha() or ("\u1200" <= ch <= "\u137F")]
        if not letters:
            scores.append(0.0)
            continue
        eth = sum(1 for ch in letters if "\u1200" <= ch <= "\u137F")
        scores.append(eth / len(letters))
    return sum(scores) / len(scores)


def adequacy_proxy(hyps: list[str], srcs: list[str], glossary: Glossary | None) -> float:
    """Heuristic 0–1 mix of number match, length, and glossary. Not human adequacy."""
    if not srcs:
        return 0.0
    scores = []
    for hyp, src in zip(hyps, srcs):
        num = 1.0 if extract_numbers(src) == extract_numbers(hyp) else 0.0
        ratio = len(hyp) / max(len(src), 1)
        length = 1.0 if 0.3 <= ratio <= 3.5 else 0.4
        term = 1.0
        if glossary:
            hits = [e for e in glossary.find_hits(src) if e.amharic]
            if hits:
                term = sum(1 for e in hits if e.amharic in hyp or e.preserve_source) / len(hits)
        scores.append(0.4 * num + 0.2 * length + 0.4 * term)
    return sum(scores) / len(scores)


def critical_mistranslation_rate(hyps: list[str], srcs: list[str]) -> float:
    """Share of segments with number mismatch or lost English negation cues."""
    if not srcs:
        return 0.0
    neg = re.compile(r"\b(?:not|never|no|avoid|contraindicat|do not|must not)\b", re.I)
    n_bad = 0
    for hyp, src in zip(hyps, srcs):
        bad = False
        if extract_numbers(src) != extract_numbers(hyp):
            bad = True
        if neg.search(src) and not (
            neg.search(hyp) or re.search(r"አይደለም|የለም|አይመከርም|የተከለከለ|አይገባም", hyp)
        ):
            bad = True
        if bad:
            n_bad += 1
    return n_bad / len(srcs)


def timed_translate(fn, texts: list[str]) -> tuple[list[str], float]:
    t0 = time.perf_counter()
    out = fn(texts)
    return out, time.perf_counter() - t0


def recommend_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank primarily by critical error rate, then terminology, then chrF++."""
    usable = [r for r in rows if r.get("status") == "ok"]
    if not usable:
        return {
            "recommended": None,
            "reason": "No successful model runs. Download models and rerun the benchmark.",
        }

    def key(row: dict[str, Any]) -> tuple:
        crit = float(row.get("critical_mistranslation_rate") or 1.0)
        term = float(row.get("medical_terminology_accuracy") or 0.0)
        if term != term:  # NaN
            term = 0.0
        chrf = float(row.get("chrfpp") or 0.0)
        return (crit, -term, -chrf)

    best = sorted(usable, key=key)[0]
    return {
        "recommended": best.get("model_key"),
        "hf_id": best.get("hf_id"),
        "reason": (
            "Selected by lowest critical-mistranslation rate, then medical terminology "
            "accuracy, then chrF++. BLEU is reported but is not the primary criterion. "
            "This ranking is not a clinical validation."
        ),
        "scores": {
            "critical_mistranslation_rate": best.get("critical_mistranslation_rate"),
            "medical_terminology_accuracy": best.get("medical_terminology_accuracy"),
            "chrfpp": best.get("chrfpp"),
            "bleu": best.get("bleu"),
        },
    }
