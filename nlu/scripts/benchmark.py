#!/usr/bin/env python3
"""Run labeled evaluation. Prints metrics only from supplied gold files. No invented scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from we_care_nlu.config import load_config  # noqa: E402
from we_care_nlu.contracts.schemas import AnalyzeRequest, ContextState  # noqa: E402
from we_care_nlu.eval.metrics import accuracy, macro_f1, mrr, recall_at_k  # noqa: E402
from we_care_nlu.pipeline import NLUEngine  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", type=Path, default=ROOT / "data" / "eval")
    p.add_argument("--kb", type=Path, default=ROOT / "tests" / "fixtures" / "mini_kb.json")
    p.add_argument("--model", default="hash")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    eng = NLUEngine(load_config(), model_key=args.model)
    if args.kb.exists():
        eng.ingest_file(str(args.kb), default_review_status="unreviewed")

    report: dict = {
        "model": eng.model_key,
        "note": "Scores are computed on the files in --eval-dir. Hash embeddings are not semantic.",
        "clinical": False,
    }

    intent_path = args.eval_dir / "intent.jsonl"
    if intent_path.exists():
        rows = load_jsonl(intent_path)
        gold = [r["intent"] for r in rows]
        pred = [eng.intents.predict(r["text"])["intent"] for r in rows]
        report["intent"] = {"n": len(rows), "accuracy": accuracy(gold, pred), "macro_f1": macro_f1(gold, pred)}

    ret_path = args.eval_dir / "retrieval.jsonl"
    if ret_path.exists():
        rows = load_jsonl(ret_path)
        ranked, rel = [], []
        for r in rows:
            hits = eng.retriever.search(r["query"], top_k=10)
            ranked.append([h.evidence_id for h in hits])
            rel.append(set(r.get("relevant_ids") or []))
        report["retrieval"] = {
            "n": len(rows),
            "recall@1": recall_at_k(ranked, rel, 1),
            "recall@5": recall_at_k(ranked, rel, 5),
            "mrr": mrr(ranked, rel),
        }

    ctx_path = args.eval_dir / "context.jsonl"
    if ctx_path.exists():
        rows = load_jsonl(ctx_path)
        ok = 0
        for r in rows:
            ctx = ContextState.model_validate(r.get("context") or {})
            out = eng.analyze(AnalyzeRequest(text=r["text"], context=ctx))
            want = r.get("expect_resolved_method")
            got = [e.value for e in out.entities if e.type == "contraceptive_method"]
            if want and want in got and not (r.get("expect_clarification") and not out.requires_clarification):
                ok += 1
            elif r.get("expect_clarification") and out.requires_clarification:
                ok += 1
        report["context_resolution_accuracy"] = ok / len(rows) if rows else 0.0

    saf_path = args.eval_dir / "safety.jsonl"
    if saf_path.exists():
        rows = load_jsonl(saf_path)
        tp = fp = fn = 0
        for r in rows:
            gold = bool(r.get("is_emergency"))
            pred = eng.analyze(AnalyzeRequest(text=r["text"])).safety["emergency"]["is_emergency"]
            tp += gold and pred
            fp += (not gold) and pred
            fn += gold and (not pred)
        rec = tp / (tp + fn) if tp + fn else 0.0
        report["safety_trigger_recall"] = rec

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
