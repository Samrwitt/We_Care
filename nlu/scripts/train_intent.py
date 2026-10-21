#!/usr/bin/env python3
"""Train an XLM-R / AfroXLMR intent classifier from labeled JSONL.

JSONL: {"text": "...", "intent": "side_effects", "cluster": "se-injectable"}
Hold out by `cluster` so paraphrases of the same question do not leak.
An untrained checkpoint is not a production intent model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--base", default="Davlan/afro-xlmr-base")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args()
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError:
        print("Install torch and transformers to train.")
        return 2

    rows = [json.loads(l) for l in args.train.read_text(encoding="utf-8").splitlines() if l.strip()]
    labels = sorted({r["intent"] for r in rows})
    lab2id = {l: i for i, l in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(args.base)

    class DS(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            r = rows[i]
            enc = tok(r["text"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(lab2id[r["intent"]])
            return item

    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=len(labels))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loader = DataLoader(DS(), batch_size=8, shuffle=True)
    steps = max(1, args.epochs * len(loader))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    model.train()
    for _ in range(args.epochs):
        for batch in loader:
            opt.zero_grad()
            out = model(**{k: v for k, v in batch.items()})
            out.loss.backward()
            opt.step()
            sched.step()
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    (args.out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}. Evaluate on a cluster-held-out split before replacing the lexicon backend.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
