#!/usr/bin/env python3
"""Compare registered embedders on labeled Amharic pairs. Skips missing weights. Does not invent scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from we_care_nlu.config import load_config  # noqa: E402
from we_care_nlu.embeddings.registry import cosine, load_embedder  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, default=ROOT / "data" / "training" / "embedding_pairs.jsonl")
    p.add_argument("--models", default="hash,labse,amharic_medium,afro_xlmr")
    args = p.parse_args()
    cfg = load_config()
    specs = cfg["embeddings"]["models"]
    pairs = [json.loads(l) for l in args.pairs.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for key in [k.strip() for k in args.models.split(",") if k.strip()]:
        spec = specs.get(key)
        if not spec:
            out.append({"model": key, "status": "unknown_key"})
            continue
        try:
            enc = load_embedder(key, spec)
        except Exception as exc:
            out.append({"model": key, "status": "not_downloaded", "error": str(exc), "hf_id": spec.get("hf_id")})
            continue
        n_ok = 0
        for row in pairs:
            q, pos = enc.encode([row["query"], row["positive"]])
            negs = row.get("negatives") or []
            nvec = enc.encode(negs) if negs else None
            pos_s = float(cosine(q, pos)[0, 0])
            neg_s = float(cosine(q, nvec).max()) if nvec is not None else -1.0
            if pos_s > neg_s:
                n_ok += 1
        out.append(
            {
                "model": key,
                "hf_id": spec.get("hf_id"),
                "status": "ok",
                "amharic_sentence_encoder": spec.get("amharic_sentence_encoder"),
                "triplet_accuracy": n_ok / len(pairs) if pairs else None,
                "n": len(pairs),
            }
        )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
