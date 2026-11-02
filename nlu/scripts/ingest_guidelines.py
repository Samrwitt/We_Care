"""Build an English-only knowledge base from guideline PDFs. Does not translate them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from we_care_nlu.kb.pdf_english import bundle_pdf_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract English guideline PDFs into NLU chunks.")
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "kb" / "rh_guidelines_en.json",
    )
    args = parser.parse_args()
    bundle = bundle_pdf_dir(args.pdf_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = [(doc.get("title"), len(doc.get("segments") or [])) for doc in bundle["documents"]]
    empty = [title for title, count in counts if count == 0]
    print(json.dumps({
        "documents": len(counts),
        "segments": sum(count for _, count in counts),
        "empty_documents": empty,
        "out": str(args.out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
