"""Extract Amharic (or mixed) DOCX paragraphs into alignment-shaped documents."""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from we_care_nlu.kb.sentences import clean_text, pack_sentences, split_sentences

_W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
_ETH = re.compile(r"[\u1200-\u137F]")
_SKIP = re.compile(
    r"^(page\s*\d+|ፊደል\s*\d+|\d{1,4}|table\s+of\s+contents|ዝርዝር\s*ሰንጠረዥ)\.?$",
    re.I,
)
_ETH_MONTH = (
    "መስከረም|ጥቅምት|ኅዳር|ህዳር|ታህሳስ|ጥር|የካቲት|መጋቢት|ሚያዝያ|ግንቦት|ሰኔ|ሐምሌ|ነሐሴ|ጳጉሜ|"
    "january|february|march|april|may|june|july|august|september|october|november|december"
)
_DATE_LINE = re.compile(
    rf"^(?:{_ETH_MONTH})[^\d]{{0,20}}\d{{4}}\s*$|"
    rf"^\d{{1,2}}[./-]\d{{1,2}}[./-]\d{{2,4}}\s*$|"
    rf"^\d{{4}}\s*$",
    re.I,
)
_TOC_TEASER = re.compile(r"[፦:]")
_METHOD_NAMES = (
    "ኮንዶም", "ክኒን", "መርፌ", "ኢምፕላንት", "አይዩ", "አይ.ዩ", "ቋሚ", "አስቸኳይ",
    "condom", "pill", "implant", "injectable", "iud", "iucd", "steriliz",
)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def docx_paragraphs(path: str | Path) -> list[str]:
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    out: list[str] = []
    for node in root.findall(".//w:p", _W_NS):
        parts = [t.text or "" for t in node.findall(".//w:t", _W_NS)]
        text = clean_text("".join(parts))
        if text:
            out.append(text)
    return out


def _is_heading(text: str, *, index: int) -> bool:
    """Conservative heading detector — avoid treating short answers as titles."""
    if len(text) > 100:
        return False
    words = text.split()
    if index == 0 and len(words) <= 16:
        return True
    if text.endswith((":", "፡")) and len(words) <= 14:
        return True
    # Numbered section labels, e.g. "1. ዳራ" / "ዓይነቶች"
    if re.match(r"^[\dIVXLC]+[.)]\s+\S+", text) and len(words) <= 12:
        return True
    if len(words) <= 6 and not text.endswith((".", "።", "?", "!", "፤", "፣")):
        return True
    return False


def _is_date_line(text: str) -> bool:
    return bool(_DATE_LINE.match(text.strip()))


def _is_toc_teaser(text: str) -> bool:
    """Drop outline lines like 'Methods: barriers, hormones, … including.'"""
    if not _TOC_TEASER.search(text):
        return False
    named = sum(1 for name in _METHOD_NAMES if name.lower() in text.lower())
    # Category teaser without concrete method names.
    if named < 2 and ("ጨምሮ" in text or "including" in text.lower()) and len(text) < 220:
        return True
    if named < 2 and text.count("፣") + text.count(",") >= 3 and len(text) < 180:
        return True
    return False


def _usable(text: str) -> bool:
    if len(text) < 25:
        return False
    if _SKIP.match(text) or _is_date_line(text):
        return False
    if _is_toc_teaser(text):
        return False
    # Keep Amharic-heavy or mixed clinical text; drop pure tiny Latin crumbs.
    if not _ETH.search(text) and len(text) < 60:
        return False
    # Fragments that trail off mid-list.
    if text.rstrip().endswith(("፣", ",", "እና", "and", "or", "ወይም")):
        return False
    return True


def docx_to_alignment(
    path: str | Path,
    *,
    max_chars: int = 700,
    flags: str = "source_amharic",
    kb_kind: str = "amharic_docx",
) -> dict:
    path = Path(path)
    paras = docx_paragraphs(path)
    # Prefer a real title over cover dates like "ታህሳስ፣ 2024".
    title = path.stem.replace("_", " ").strip()
    for para in paras[:8]:
        if _is_date_line(para) or _is_toc_teaser(para):
            continue
        if _ETH.search(para) and 12 <= len(para) <= 160 and not para.endswith(("፣", ",")):
            title = para
            break

    segments: list[dict] = []
    current_section = title
    n = 0
    for i, para in enumerate(paras):
        text = clean_text(para)
        if not text:
            continue
        if _is_heading(text, index=i):
            # Track section for context only — do not index bare titles as answer chunks.
            current_section = text
            continue
        if not _usable(text):
            continue
        packed = pack_sentences(split_sentences(text) or [text], max_chars=max_chars)
        if not packed:
            packed = [text[:max_chars]]
        for chunk in packed:
            chunk = clean_text(chunk)
            if not _usable(chunk):
                continue
            n += 1
            # Amharic primary; leftover Latin stays empty in source_text for blogs/guidelines_am.
            am = chunk if _ETH.search(chunk) else ""
            en = "" if am else chunk
            if am and not en:
                # Mixed lines: keep full text as Amharic evidence.
                am = chunk
            segments.append(
                {
                    "segment_id": f"{path.stem}-s{n:04d}",
                    "kind": "paragraph",
                    "section": current_section,
                    "source_text": en,
                    "translated_text": am or chunk,
                    "flags": flags,
                }
            )

    return {
        "document_name": path.name,
        "title": title,
        "kb_kind": kb_kind,
        "language": "am",
        "content_hash": _hash(path.name + str(len(segments))),
        "segments": segments,
    }


def collect_docx(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(folder.glob("*.docx")) + sorted(folder.glob("*.DOCX"))
