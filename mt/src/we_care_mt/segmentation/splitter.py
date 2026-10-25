"""Sentence-aware splitting that avoids breaking medical abbreviations."""

from __future__ import annotations

import re

from we_care_mt.constants import ABBREVIATIONS

_SENT_END = re.compile(r"([.!?])([\"')\]]*)(\s+|$)")
_MULTI_SPACE = re.compile(r"[ \t]+")


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\ufeff", "")
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def _is_abbrev_period(text: str, period_index: int) -> bool:
    # Walk back to the token before the period.
    i = period_index - 1
    while i >= 0 and text[i].isalpha():
        i -= 1
    token = text[i + 1 : period_index].lower().rstrip(".")
    if token in ABBREVIATIONS:
        return True
    # Initials: "M." or "J. Smith"
    if len(token) == 1:
        return True
    # Decimal numbers: 41.5 or 401.0
    if period_index > 0 and period_index + 1 < len(text):
        if text[period_index - 1].isdigit() and text[period_index + 1].isdigit():
            return True
    # e.g. / i.e.
    if token in {"e", "i"} and period_index >= 2:
        return True
    return False


def split_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    # Do not split very short captions or headings.
    if len(text) < 40 or text.count(".") + text.count("!") + text.count("?") == 0:
        return [text]

    sentences: list[str] = []
    start = 0
    for match in _SENT_END.finditer(text):
        punct_at = match.start(1)
        if _is_abbrev_period(text, punct_at):
            continue
        end = match.end(2)
        piece = text[start:end].strip()
        if piece:
            sentences.append(piece)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences or [text]


def pack_sentences(sentences: list[str], max_chars: int) -> list[str]:
    if max_chars <= 0:
        return sentences
    packed: list[str] = []
    buf: list[str] = []
    size = 0
    for sent in sentences:
        if len(sent) > max_chars:
            if buf:
                packed.append(" ".join(buf))
                buf, size = [], 0
            packed.extend(_hard_wrap(sent, max_chars))
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


def _hard_wrap(text: str, max_chars: int) -> list[str]:
    """Last-resort wrap on whitespace; never mid-word if avoidable."""
    words = text.split()
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for w in words:
        extra = len(w) + (1 if buf else 0)
        if buf and size + extra > max_chars:
            out.append(" ".join(buf))
            buf, size = [w], len(w)
        else:
            buf.append(w)
            size += extra
    if buf:
        out.append(" ".join(buf))
    return out or [text[:max_chars]]
