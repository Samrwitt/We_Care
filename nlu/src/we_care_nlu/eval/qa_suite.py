"""Run multi-topic Q&A cases and persist question/answer JSON results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from we_care_nlu.config import load_config
from we_care_nlu.contracts.schemas import AnalyzeRequest, ContextState
from we_care_nlu.pipeline import NLUEngine

NLU_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES = NLU_ROOT / "data" / "eval" / "qa_cases.json"
DEFAULT_OUT = NLU_ROOT / "data" / "eval" / "qa_results.json"
USER_KB = NLU_ROOT / "data" / "kb" / "fp_user_kb.json"
DOCS_KB = NLU_ROOT / "data" / "kb" / "docs_amharic_kb.json"


def load_cases(path: Path | None = None) -> dict[str, Any]:
    p = path or DEFAULT_CASES
    return json.loads(p.read_text(encoding="utf-8"))


def build_engine(*, model: str = "hash", with_docs: bool = True) -> NLUEngine:
    eng = NLUEngine(load_config(), model_key=model)
    if with_docs and DOCS_KB.exists():
        eng.ingest_file(
            str(DOCS_KB),
            document_version="docs-am",
            default_review_status="source",
            rebuild=False,
        )
    if USER_KB.exists():
        eng.ingest_file(
            str(USER_KB),
            document_version="demo",
            default_review_status="unreviewed",
            rebuild=True,
        )
    return eng


def _ctx_from_dict(raw: dict[str, Any] | None) -> ContextState | None:
    if not raw:
        return None
    return ContextState(
        turn_index=int(raw.get("turn_index") or 0),
        last_intents=list(raw.get("last_intents") or []),
        last_topic=raw.get("last_topic"),
        last_entities=list(raw.get("last_entities") or []),
    )


def _reply_payload(out: Any) -> dict[str, Any]:
    return {
        "amharic_reply": out.amharic_reply or "",
        "english_reply": out.english_reply or "",
        "evidence_id": out.evidence_id,
        "document_title": out.document_title or "",
        "page": out.page,
        "intent": out.intent,
        "topic": getattr(out, "topic", None),
        "requires_clarification": bool(out.requires_clarification),
        "clarification_question_key": out.clarification_question_key,
        "confidence_status": out.confidence_status,
    }


def _check_expect(expect: dict[str, Any], out: Any, body: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    if "evidence_id" in expect:
        add("evidence_id", out.evidence_id == expect["evidence_id"], {
            "expected": expect["evidence_id"],
            "got": out.evidence_id,
        })
    if "evidence_id_not" in expect:
        banned = list(expect["evidence_id_not"] or [])
        add("evidence_id_not", out.evidence_id not in banned, {
            "banned": banned,
            "got": out.evidence_id,
        })
    if "must_contain" in expect:
        missing = [t for t in expect["must_contain"] if t not in body]
        add("must_contain", not missing, {"missing": missing})
    if "must_contain_any" in expect:
        opts = list(expect["must_contain_any"] or [])
        add("must_contain_any", any(t.lower() in body.lower() for t in opts), {"options": opts})
    if "must_not_contain" in expect:
        hit = [t for t in expect["must_not_contain"] if t.lower() in body.lower()]
        add("must_not_contain", not hit, {"found": hit})
    if "intent_in" in expect:
        allowed = list(expect["intent_in"] or [])
        add("intent_in", out.intent in allowed, {"allowed": allowed, "got": out.intent})
    if "clarify" in expect:
        want = bool(expect["clarify"])
        add("clarify", bool(out.requires_clarification) is want, {
            "expected": want,
            "got": bool(out.requires_clarification),
        })
    if "reply_starts_with_any" in expect:
        prefixes = list(expect["reply_starts_with_any"] or [])
        stripped = body.lstrip()
        add(
            "reply_starts_with_any",
            any(stripped.startswith(p) for p in prefixes),
            {"prefixes": prefixes, "head": stripped[:60]},
        )
    return checks


def run_case(eng: NLUEngine, case: dict[str, Any]) -> dict[str, Any]:
    case_id = str(case.get("id") or case.get("question") or "case")
    topic = case.get("topic") or ""
    expect = dict(case.get("expect") or {})
    turns = list(case.get("turns") or [])
    skipped = False
    skip_reason = None
    if case.get("needs_docs") and not DOCS_KB.exists():
        skipped = True
        skip_reason = "docs_amharic_kb.json missing"

    answers: list[dict[str, Any]] = []
    out = None
    body = ""
    question = case.get("question") or ""

    if not skipped:
        if turns:
            ctx = None
            for i, turn in enumerate(turns):
                q = str(turn.get("question") or "")
                req = AnalyzeRequest(text=q, context=ctx)
                out = eng.reply(req)
                analysis = eng.analyze(req)
                answers.append({
                    "turn": i,
                    "question": q,
                    "answer": _reply_payload(out),
                    "analysis_intent": analysis.intent,
                    "analysis_topic": analysis.topic,
                })
                # Carry context into next turn.
                ctx = ContextState(
                    turn_index=i + 1,
                    last_intents=[analysis.intent] if analysis.intent else [],
                    last_topic=analysis.topic or (ctx.last_topic if ctx else None),
                    last_entities=[
                        {"type": e.type, "value": e.value, "surface": e.surface}
                        for e in (analysis.entities or [])
                    ],
                )
            question = str(turns[-1].get("question") or question)
        else:
            ctx = _ctx_from_dict(case.get("context"))
            out = eng.reply(AnalyzeRequest(text=str(question), context=ctx))
            answers.append({"turn": 0, "question": question, "answer": _reply_payload(out)})

        body = (out.amharic_reply if out else "") or ""
        checks = _check_expect(expect, out, body) if out is not None else []
        passed = all(c["ok"] for c in checks) if checks else True
    else:
        checks = [{"name": "skipped", "ok": False, "detail": skip_reason}]
        passed = False

    return {
        "id": case_id,
        "topic": topic,
        "question": question,
        "needs_docs": bool(case.get("needs_docs")),
        "skipped": skipped,
        "skip_reason": skip_reason,
        "passed": passed and not skipped,
        "checks": checks,
        "answers": answers,
        "answer": answers[-1]["answer"] if answers else None,
    }


def run_suite(
    *,
    cases_path: Path | None = None,
    out_path: Path | None = None,
    model: str = "hash",
    with_docs: bool = True,
    fail_fast: bool = False,
) -> dict[str, Any]:
    payload = load_cases(cases_path)
    cases = list(payload.get("cases") or [])
    eng = build_engine(model=model, with_docs=with_docs)
    results: list[dict[str, Any]] = []
    for case in cases:
        row = run_case(eng, case)
        results.append(row)
        if fail_fast and not row["passed"] and not row.get("skipped"):
            break

    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"] and not r.get("skipped"))
    skipped = sum(1 for r in results if r.get("skipped"))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "with_docs": with_docs,
        "cases_path": str(cases_path or DEFAULT_CASES),
        "cases_version": payload.get("version"),
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
        },
        "results": results,
    }
    dest = out_path or DEFAULT_OUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["out_path"] = str(dest)
    return report
