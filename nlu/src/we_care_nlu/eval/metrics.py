"""Evaluation metrics. Callers must supply labeled data; nothing is fabricated."""

from __future__ import annotations

def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    scores = []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return sum(scores) / len(scores)


def span_prf(gold: list[tuple[int, int, str]], pred: list[tuple[int, int, str]]) -> dict[str, float]:
    gs, ps = set(gold), set(pred)
    tp = len(gs & ps)
    prec = tp / len(ps) if ps else 0.0
    rec = tp / len(gs) if gs else 0.0
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {"precision": prec, "recall": rec, "f1": f1}


def recall_at_k(ranked_ids: list[list[str]], relevant: list[set[str]], k: int) -> float:
    if not ranked_ids:
        return 0.0
    hits = 0
    for rank, rel in zip(ranked_ids, relevant):
        if rel & set(rank[:k]):
            hits += 1
    return hits / len(ranked_ids)


def mrr(ranked_ids: list[list[str]], relevant: list[set[str]]) -> float:
    if not ranked_ids:
        return 0.0
    total = 0.0
    for rank, rel in zip(ranked_ids, relevant):
        rr = 0.0
        for i, doc in enumerate(rank, start=1):
            if doc in rel:
                rr = 1.0 / i
                break
        total += rr
    return total / len(ranked_ids)
