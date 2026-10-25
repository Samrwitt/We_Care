"""Extract English guideline PDFs into alignment documents. No translation."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from we_care_nlu.kb.sentences import clean_text, pack_sentences, split_sentences

_PAGE_NO = re.compile(r"^\d{1,4}$")
_WORD = re.compile(r"[A-Za-z]+")


def _readable_block(text: str) -> bool:
    letters = sum(ch.isalpha() for ch in text)
    if len(text) < 40 or letters / len(text) < 0.55:
        return False
    words = _WORD.findall(text)
    if any(len(word) > 28 for word in words):
        return False
    tokens = text.split()
    if len(tokens) >= 12:
        short = sum(1 for token in tokens if len(token.strip(".,;:()[]")) <= 2)
        if short / len(tokens) > 0.34:
            return False
    return True


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _page_blocks(page) -> list[str]:
    raw = page.get_text("blocks") or []
    ordered = sorted(raw, key=lambda block: (round(float(block[1]), 1), float(block[0])))
    texts: list[str] = []
    for block in ordered:
        if len(block) > 6 and int(block[6]) != 0:
            continue
        text = clean_text(str(block[4] if len(block) > 4 else ""))
        if text:
            texts.append(text)
    return texts


def _repeating_lines(pages: list[list[str]]) -> set[str]:
    counts: dict[str, int] = {}
    for blocks in pages:
        seen: set[str] = set()
        for block in blocks:
            line = block.strip()
            if 8 <= len(line) <= 140:
                seen.add(line.casefold())
        for line in seen:
            counts[line] = counts.get(line, 0) + 1
    return {line for line, count in counts.items() if count >= 3}


def _pages_pymupdf(pdf_path: Path) -> list[list[str]] | None:
    try:
        import pymupdf
    except ImportError:
        return None
    doc = pymupdf.open(pdf_path)
    try:
        return [_page_blocks(page) for page in doc]
    finally:
        doc.close()


def _pages_pdftotext(pdf_path: Path) -> list[list[str]]:
    import shutil
    import subprocess

    if shutil.which("pdftotext") is None:
        raise RuntimeError("Install pymupdf==1.26.4 or poppler-utils (pdftotext).")
    completed = subprocess.run(
        ["pdftotext", "-raw", "-enc", "UTF-8", str(pdf_path), "-"],
        check=True,
        capture_output=True,
    )
    raw = completed.stdout.decode("utf-8", errors="replace")
    pages: list[list[str]] = []
    for page in raw.split("\f"):
        blocks: list[str] = []
        paragraph: list[str] = []
        for line in page.splitlines():
            stripped = line.strip()
            if not stripped:
                if paragraph:
                    blocks.append(clean_text(" ".join(paragraph)))
                    paragraph = []
                continue
            paragraph.append(stripped)
        if paragraph:
            blocks.append(clean_text(" ".join(paragraph)))
        pages.append([block for block in blocks if block])
    return pages


def pdf_to_alignment(path: str | Path, *, max_chars: int = 700) -> dict:
    pdf_path = Path(path)
    pages = _pages_pymupdf(pdf_path)
    if pages is None:
        pages = _pages_pdftotext(pdf_path)
    repeating = _repeating_lines(pages)
    segments: list[dict] = []
    seen: set[str] = set()
    section = ""
    for page_no, blocks in enumerate(pages, start=1):
        for block in blocks:
            if block.casefold() in repeating or _PAGE_NO.match(block):
                continue
            if len(block) < 80 and block.count(".") == 0:
                section = block
                continue
            if not _readable_block(block):
                continue
            for piece in pack_sentences(split_sentences(block), max_chars=max_chars):
                if len(piece) < 40:
                    continue
                key = _hash(piece)
                if key in seen:
                    continue
                seen.add(key)
                segments.append(
                    {
                        "segment_id": f"{pdf_path.stem}:p{page_no}:{key}",
                        "kind": "paragraph",
                        "page_number": page_no,
                        "section": section,
                        "source_text": piece,
                        "translated_text": "",
                        "flags": "source_english",
                    }
                )
    return {
        "document_name": pdf_path.name,
        "title": pdf_path.stem,
        "segments": segments,
    }


def bundle_pdf_dir(directory: str | Path) -> dict:
    root = Path(directory)
    pdfs = sorted(path for path in root.rglob("*.pdf") if path.is_file())
    documents = []
    for pdf in pdfs:
        documents.append(pdf_to_alignment(pdf))
    return {
        "kind": "english_guideline_bundle",
        "source_dir": str(root),
        "documents": documents,
    }
