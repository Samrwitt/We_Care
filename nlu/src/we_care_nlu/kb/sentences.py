"""Sentence packing for English guideline chunks and reply spans."""

from __future__ import annotations

import re

# Include Ethiopic full stop (።) used in Amharic articles.
_SENT_END = re.compile(r"([.!?።])([\"')\]]*)(\s+|$)")
_SPACE = re.compile(r"[ \t]+")
_ABBREV = {
    "dr", "mr", "mrs", "ms", "prof", "fig", "no", "vol", "pp", "ed",
    "e.g", "i.e", "vs", "etc", "mg", "ml", "mcg", "iu", "hiv", "aids",
    "fp", "sti", "iud", "iucd", "anc", "pnc", "pph",
}


def clean_text(text: str) -> str:
    text = (text or "").replace("\u00a0", " ").replace("\ufeff", "").replace("\u00ad", "")
    text = text.replace("\n", " ")
    return _SPACE.sub(" ", text).strip()


def _is_abbrev_period(text: str, period_index: int) -> bool:
    i = period_index - 1
    while i >= 0 and (text[i].isalpha() or text[i] == "."):
        i -= 1
    token = text[i + 1 : period_index].lower().rstrip(".")
    if token in _ABBREV:
        return True
    if len(token) == 1:
        return True
    if period_index > 0 and period_index + 1 < len(text):
        if text[period_index - 1].isdigit() and text[period_index + 1].isdigit():
            return True
    return False


def split_sentences(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    if len(text) < 40 or not any(mark in text for mark in ".!?።"):
        return [text]
    sentences: list[str] = []
    start = 0
    for match in _SENT_END.finditer(text):
        punct_at = match.start(1)
        if _is_abbrev_period(text, punct_at):
            continue
        piece = text[start : match.end(2)].strip()
        if piece:
            sentences.append(piece)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences or [text]


def pack_sentences(sentences: list[str], max_chars: int = 700) -> list[str]:
    packed: list[str] = []
    buf: list[str] = []
    size = 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > max_chars:
            if buf:
                packed.append(" ".join(buf))
                buf, size = [], 0
            for i in range(0, len(sent), max_chars):
                packed.append(sent[i : i + max_chars].strip())
            continue
        extra = len(sent) + (1 if buf else 0)
        if buf and size + extra > max_chars:
            packed.append(" ".join(buf))
            buf, size = [sent], len(sent)
        else:
            buf.append(sent)
            size += extra
    if buf:
        packed.append(" ".join(buf))
    return packed
