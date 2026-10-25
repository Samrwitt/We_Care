#!/usr/bin/env python3
"""Download configured NLLB / AfriScience-MT weights into the local Hugging Face cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from we_care_mt.config import load_config  # noqa: E402
from we_care_mt.translation.nllb import NLLBTranslator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dsfsi_600m", help="Model key or 'all'")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    keys = list(cfg.models.get("models", {}).keys()) if args.model == "all" else [args.model]
    cache = cfg.path("hf_cache_dir")
    cache.mkdir(parents=True, exist_ok=True)
    print("Language codes: eng_Latn → amh_Ethi")
    print("None of these models is clinically validated.")
    for key in keys:
        entry = cfg.model_entry(key)
        hf_id = entry["hf_id"]
        print(
            f"Downloading {hf_id}\n"
            f"  license={entry.get('license')}  VRAM~{entry.get('recommended_vram_gb')}GB  "
            f"max_input={entry.get('max_input_tokens')}"
        )
        if str(entry.get("license", "")).upper().startswith("CC-BY-NC"):
            print("  WARNING: CC-BY-NC — research / non-commercial use only.")
        tr = NLLBTranslator(hf_id=hf_id, device="cpu", cache_dir=str(cache))
        tr.load()
        tr.unload()
        print("  done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
