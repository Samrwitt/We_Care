"""Turn retrieved evidence into a short, spoken Amharic reply.

Grounding still comes from the guideline chunk. The user-facing Amharic is
cleaned and (when MT is available) re-rendered from a short English answer so
it is not a raw paste of PDF/table text.
"""

from __future__ import annotations

import re
from typing import Any

from we_care_nlu.kb.sentences import clean_text, split_sentences

_ETH = re.compile(r"[\u1200-\u137F]")
_BULLET = re.compile(r"^[\s§¸•▪◦\-–—*]+")
_TOC_DOTS = re.compile(r"[.\u2026]{2,}")
_PAGE_REF = re.compile(r"\s+\d{1,3}\s*$")
_LEAD = re.compile(
    r"^(for example,|in this document,?|in addition,?|additionally,?|"
    r"moreover,?|also,?|as a result;?|therefore,?|however,?)\s+",
    re.I,
)
_MULTI_SPACE = re.compile(r"\s+")
# Prefer Amharic wording over leftover Latin clinical abbreviations in replies.
_EXPAND = (
    (re.compile(r"\bSTIs?\b", re.I), "በወሲብ የሚተላለፉ በሽታዎች"),
    (re.compile(r"\bRTIs?\b", re.I), "የመራቢያ አካል ኢንፌክሽኖች"),
    (re.compile(r"HIV"), "ኤች.አይ.ቪ"),
    (re.compile(r"AIDS"), "ኤድስ"),
    (re.compile(r"\bFP\b"), "የቤተሰብ ምጣኔ"),
    (re.compile(r"\bOCP\b"), "በአፍ የሚወሰድ የእርግዝና መከላከያ ክኒን"),
    (re.compile(r"\bIUDs?\b", re.I), "በማሕፀን ውስጥ የሚገባ መከላከያ"),
    (re.compile(r"\bIUCDs?\b", re.I), "በማሕፀን ውስጥ የሚገባ መከላከያ"),
)


def clean_evidence(text: str) -> str:
    text = clean_text(text or "")
    text = _TOC_DOTS.sub(" ", text)
    text = _PAGE_REF.sub("", text)
    text = _BULLET.sub("", text)
    text = _LEAD.sub("", text)
    return _MULTI_SPACE.sub(" ", text).strip(" ;,")


def shape_english_answer(span: str, *, max_chars: int = 320) -> str:
    """One or two clean sentences suitable as a chat answer, not a PDF dump."""
    text = clean_evidence(span)
    if not text:
        return ""
    parts: list[str] = []
    for sentence in split_sentences(text):
        sentence = clean_evidence(sentence)
        if len(sentence) < 12:
            continue
        # Skip pure heading / label fragments.
        if sentence.endswith(":") and len(sentence.split()) <= 6:
            continue
        letters = re.sub(r"[^A-Za-z]", "", sentence)
        if letters and letters.isupper() and len(sentence.split()) <= 8:
            continue
        parts.append(sentence)
        joined = " ".join(parts)
        if len(joined) >= 80 or len(parts) >= 2:
            break
    out = " ".join(parts) if parts else text
    if len(out) > max_chars:
        cut = out[:max_chars].rsplit(" ", 1)[0].rstrip(",;:")
        out = cut + ("." if cut and cut[-1] not in ".!?" else "")
    if out and out[-1] not in ".!?":
        out += "."
    return out


_DATE_LINE = re.compile(
    r"(?m)^(?:መስከረም|ጥቅምት|ኅዳር|ህዳር|ታህሳስ|ጥር|የካቲት|መጋቢት|ሚያዝያ|ግንቦት|ሰኔ|ሐምሌ|ነሐሴ|ጳጉሜ)"
    r"[^\n\d]{0,20}\d{4}\s*$"
)


_METHOD_KEEP = (
    "ኮንዶም", "ክኒን", "መርፌ", "ኢምፕላንት", "አይዩ", "አስቸኳይ", "ቋሚ",
    "condom", "pill", "injectable", "implant", "iud", "iucd",
)


def polish_amharic(text: str, *, keep_list: bool = False) -> str:
    text = clean_evidence(text)
    text = _DATE_LINE.sub("", text)
    for pattern, repl in _EXPAND:
        text = pattern.sub(repl, text)
    # Drop leftover isolated Latin tokens that survived glossary expansion.
    text = re.sub(r"\b[A-Z]{2,}(?:/[A-Z]+)?\b", "", text)
    # Expanding "FP" after የ/of can leave "የ የቤተሰብ" — collapse to one.
    text = re.sub(r"የ\s+የቤተሰብ", "የቤተሰብ", text)
    text = re.sub(r"(የ|በ|ለ|ከ|ወደ)\s+\1\s+", r"\1 ", text)
    text = _MULTI_SPACE.sub(" ", text).strip(" ;,")
    if text and text[-1] not in ".!?።":
        text += "።"
    if keep_list:
        # Do not chop a named method overview into a truncated first line.
        return text[:900]
    return text


def _is_method_overview(text: str) -> bool:
    body = text or ""
    return sum(1 for cue in _METHOD_KEEP if cue.lower() in body.lower()) >= 4


def compose_amharic_reply(
    *,
    english_span: str,
    document_amharic: str = "",
    translator: Any | None = None,
) -> tuple[str, str, list[str]]:
    """Return (amharic_reply, backend_tag, notes).

    Prefers Google/MT rendering of a short English answer. Falls back to a
    polished document Amharic sentence when MT is unavailable.
    """
    notes: list[str] = ["composed_reply"]
    # Native Amharic curated answers are already user-facing — keep them intact.
    if document_amharic and _ETH.search(document_amharic) and (
        _is_method_overview(document_amharic) or len(document_amharic) <= 520
    ):
        polished = polish_amharic(document_amharic, keep_list=True)
        if polished and len(polished) >= 40:
            notes.append("document_native_keep")
            return polished, "composed", notes

    shaped = shape_english_answer(english_span, max_chars=480)
    if translator is not None and getattr(translator, "can_translate_reply", False) and shaped:
        try:
            rendered = translator.to_amharic(shaped)
            polished = polish_amharic(rendered, keep_list=_is_method_overview(shaped))
            if polished and _ETH.search(polished):
                notes.append("mt_rendered")
                return polished, "composed", notes
            notes.append("mt_empty")
        except Exception as exc:  # noqa: BLE001 — fall back to document Amharic
            notes.append(f"mt_fallback:{exc.__class__.__name__}")

    fallback = polish_amharic(document_amharic, keep_list=_is_method_overview(document_amharic)) if document_amharic else ""
    if not fallback and shaped:
        # Last resort: keep English shaped answer out of the Amharic field.
        fallback = polish_amharic(document_amharic or "")
    if fallback and _ETH.search(fallback):
        if not _is_method_overview(fallback):
            # Prefer a single sentence when dumping document Amharic.
            sentences = [s for s in re.split(r"(?<=[።!?])\s+", fallback) if s.strip()]
            if sentences:
                fallback = sentences[0].strip()
                if fallback[-1] not in ".!?።":
                    fallback += "።"
        notes.append("document_polished")
        return fallback, "composed", notes
    return "", "composed", notes + ["compose_failed"]
