#!/usr/bin/env python3
"""Contrastive fine-tuning for Amharic sentence embeddings.

Input JSONL fields: query, positive, negatives (list).
Does not ship a medical-tuned checkpoint; run this when you have pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_pairs(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--base", default="rasyosef/roberta-amharic-text-embedding-medium")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError:
        print("Install sentence-transformers and torch to fine-tune.")
        return 2

    rows = load_pairs(args.train)
    if not rows:
        print("No training pairs.")
        return 1
    examples = []
    for row in rows:
        pos = row["positive"]
        examples.append(InputExample(texts=[row["query"], pos]))
        for neg in row.get("negatives") or []:
            # MultipleNegativesRankingLoss uses in-batch negatives;
            # extra explicit negatives can be added as further positives of other queries.
            examples.append(InputExample(texts=[neg, neg]))
    model = SentenceTransformer(args.base)
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(train_objectives=[(loader, loss)], epochs=args.epochs, warmup_steps=10, output_path=str(args.out))
    print(f"Wrote {args.out}. Re-evaluate on held-out medical pairs before production use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
