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


_LIST_INTRO = re.compile(
    r"የሚከተሉት|እንደሚከተለው|እንደሚከተሉት|ከሚከተሉት|as follows|the following",
    re.I,
)
_CRITERION_UNIT = re.compile(
    r"ቀናት|ቀን|ሚ\.?\s*ሊ|ሴ\.?\s*ሜ|ዓመት|ሳምንት|ሰዓት|%|ፐርሰንት|"
    r"ያልበለጠ|መብለጥ|የለበትም|አለበት|ከ\s*\d|\d\s*[-–]\s*\d",
    re.I,
)


def _is_criterion_or_list_item(text: str) -> bool:
    """Bullet / numeric clinical criteria — content, not section titles."""
    t = (text or "").strip()
    if not t:
        return False
    if re.match(r"^[·•\-–\*]\s*\S", t):
        return True
    if re.search(r"\d", t) and _CRITERION_UNIT.search(t) and len(t) <= 160:
        return True
    # Short clinical constraints without a leading digit.
    if len(t) <= 120 and any(x in t for x in ("የለበትም", "ያልበለጠ", "መብለጥ የለበትም")):
        return True
    return False


def _is_list_intro(text: str) -> bool:
    return bool(_LIST_INTRO.search(text or ""))


def _is_heading(text: str, *, index: int) -> bool:
    """Conservative heading detector — avoid treating short answers as titles."""
    if len(text) > 100:
        return False
    # Clinical list criteria look short but are answers, not headings.
    if _is_criterion_or_list_item(text):
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


def _coalesce_list_blocks(paras: list[str]) -> list[str]:
    """Merge 'the following criteria…' with the bullet lines that follow."""
    out: list[str] = []
    i = 0
    while i < len(paras):
        p = paras[i]
        if _is_list_intro(p):
            parts = [p]
            j = i + 1
            while j < len(paras):
                nxt = paras[j]
                if _is_criterion_or_list_item(nxt):
                    parts.append(nxt)
                    j += 1
                    continue
                # Closing “these are the signs of a healthy cycle…” belongs with the list.
                if (
                    "እነዚህ" in nxt
                    and any(x in nxt for x in ("ናቸው", "ን", "መገለጫ", "መስፈርት"))
                    and len(nxt) < 220
                ):
                    parts.append(nxt)
                    j += 1
                    # Definition punchline that refers back to the list ("therefore … is called X").
                    if j < len(paras):
                        follow = paras[j]
                        if (
                            follow.startswith(("ስለዚህ", "ስለዚህም", "Therefore", "thus"))
                            and len(follow) < 280
                            and any(x in follow for x in ("ይሉታል", "ይባላል", "መዛባት", "called", "irregular"))
                        ):
                            parts.append(follow)
                            j += 1
                    break
                break
            out.append(" ".join(parts))
            i = j
            continue
        out.append(p)
        i += 1
    return out


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
    t = (text or "").strip()
    min_len = 12 if _is_criterion_or_list_item(t) else 25
    if len(t) < min_len:
        return False
    if _SKIP.match(t) or _is_date_line(t):
        return False
    if _is_toc_teaser(t):
        return False
    # Keep Amharic-heavy or mixed clinical text; drop pure tiny Latin crumbs.
    if not _ETH.search(t) and len(t) < 60:
        return False
    # Fragments that trail off mid-list.
    if t.rstrip().endswith(("፣", ",", "እና", "and", "or", "ወይም")):
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
    paras = _coalesce_list_blocks(docx_paragraphs(path))
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
