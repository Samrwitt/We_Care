#!/usr/bin/env python3
"""Build an extensive Amharic KB from Docs/blogs and Docs/guidelines_am."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NLU_ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NLU_ROOT / "src"))

from we_care_nlu.kb.docx_amharic import collect_docx, docx_to_alignment  # noqa: E402
from we_care_nlu.kb.ingest import ingest_alignment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=REPO / "Docs",
        help="Repo Docs/ folder with blogs/ and guidelines_am/",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=NLU_ROOT / "data" / "kb" / "docs_amharic_kb.json",
    )
    parser.add_argument("--max-chars", type=int, default=700)
    args = parser.parse_args()

    documents: list[dict] = []
    blogs = collect_docx(args.docs_dir / "blogs")
    guides = collect_docx(args.docs_dir / "guidelines_am")

    for path in blogs:
        try:
            documents.append(
                docx_to_alignment(
                    path,
                    max_chars=args.max_chars,
                    flags="wecare_article,source_amharic",
                    kb_kind="blog_amharic",
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"skip blog {path.name}: {exc}", file=sys.stderr)

    for path in guides:
        try:
            documents.append(
                docx_to_alignment(
                    path,
                    max_chars=args.max_chars,
                    flags="source_amharic,translated_guideline",
                    kb_kind="guideline_amharic",
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"skip guide {path.name}: {exc}", file=sys.stderr)

    payload = {
        "kb_kind": "docs_amharic",
        "language": "am",
        "document_count": len(documents),
        "documents": documents,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    chunks = []
    for doc in documents:
        chunks.extend(ingest_alignment(doc, default_review_status="source", document_version="docs-am"))
    print(
        json.dumps(
            {
                "out": str(args.out),
                "documents": len(documents),
                "blogs": len(blogs),
                "guidelines": len(guides),
                "segments": sum(len(d.get("segments") or []) for d in documents),
                "ingestable_chunks": len(chunks),
                "bytes": args.out.stat().st_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
