#!/usr/bin/env python3
"""Translate sample guideline sentences with Google and print the Amharic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from we_care_mt.config import load_config
from we_care_mt.glossary.manager import Glossary
from we_care_mt.span import SpanTranslator
from we_care_mt.translation.google import GoogleTranslator

SAMPLES = [
    ("en-am", "Contraceptive method options include condoms, oral contraceptive pills, injectables, implants, IUCD, emergency contraception, and permanent methods."),
    ("en-am", "An injectable contraceptive is a progestin given by intramuscular injection, commonly every 13 weeks."),
    ("en-am", "Combined hormonal contraceptives are contraindicated during the first six months of breastfeeding."),
    ("en-am", "Progestin-only injectables can be used while breastfeeding after the first six weeks postpartum."),
    ("en-am", "Side effects of the injectable may include irregular bleeding and delayed return to fertility."),
    ("en-am", "If one pill is missed, take it as soon as remembered and continue the pack."),
    ("am-en", "የእርግዝና መከላከያ ዘዴዎች ምንድን ናቸው?"),
    ("am-en", "ጡት እያጠባሁ መርፌ መጠቀም ይቻላል?"),
]


def main() -> int:
    cfg = load_config()
    glossary = None
    gloss_path = cfg.path("glossary")
    if gloss_path.exists():
        freeze = cfg.path("freeze_patterns")
        glossary = Glossary.load(gloss_path, freeze if freeze.exists() else None)
    backend = GoogleTranslator()
    backend.load()
    span = SpanTranslator(backend, glossary, model_id=backend.hf_id, num_beams=1)
    rows = []
    for direction, text in SAMPLES:
        out = span.translate(text, direction)
        rows.append(
            {
                "direction": direction,
                "source": text,
                "google": out["text"],
                "qa_notes": out["qa_notes"],
                "model": out.get("model") or backend.hf_id,
            }
        )
        print(f"\n[{direction}]")
        print("SRC:", text)
        print("GT :", out["text"])
        if out["qa_notes"]:
            print("QA :", ", ".join(out["qa_notes"]))
    out_path = cfg.path("output_dir") / "google_span_compare.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
