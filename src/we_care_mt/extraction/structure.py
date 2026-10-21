"""Heading / list / footnote classification from font and text cues."""

from __future__ import annotations

import re
from statistics import median

from we_care_mt.extraction.headers import looks_like_page_number

LIST_START = re.compile(
    r"""^
    (?:
        [\u2022\u2023\u25E6\u2043\u2219\u00B7\u25AA\u25CF\u25A0§Ø¸•▪‣-]
        |\d{1,2}[.)]
        |[A-Za-z][.)]
        |[ivxIVX]{1,4}[.)]
        |\(\d{1,2}\)
        |\[\d{1,2}\]
    )
    \s+
    """,
    re.VERBOSE,
)

ALL_CAPS_HEADING = re.compile(r"^[A-Z0-9][A-Z0-9 /,&:;()'%\-]{6,}$")
NUMBERED_HEADING = re.compile(r"^(?:\d+(?:\.\d+){0,4}|Annex\s+[IVXLC\d]+)[.\s]+.+$", re.I)


def body_font_size(sizes: list[float]) -> float:
    usable = [s for s in sizes if 6.0 <= s <= 18.0]
    if not usable:
        return 11.0
    return float(median(usable))


def classify_block(
    text: str,
    font_size: float,
    bold: bool,
    body_size: float,
    y0: float,
    page_height: float,
    heading_delta: float,
    in_header_band: bool,
    in_footer_band: bool,
    is_running: bool,
) -> tuple[str, int | None, str]:
    stripped = text.strip()
    if not stripped:
        return "paragraph", None, ""
    if is_running:
        return ("header" if in_header_band else "footer"), None, ""
    if in_header_band and len(stripped) < 180:
        return "header", None, ""
    if in_footer_band and looks_like_page_number(stripped):
        return "page_number", None, ""
    if in_footer_band and len(stripped) < 160:
        return "footer", None, ""
    if font_size <= body_size - 1.8 and y0 > page_height * 0.78 and len(stripped) < 350:
        return "footnote", None, ""

    list_m = LIST_START.match(stripped)
    if list_m:
        return "list_item", None, list_m.group(0).strip()

    level = None
    is_heading = False
    if font_size >= body_size + heading_delta:
        is_heading = True
    if bold and font_size >= body_size + 0.4 and len(stripped) < 180:
        is_heading = True
    if ALL_CAPS_HEADING.match(stripped) and len(stripped) < 120:
        is_heading = True
    num = NUMBERED_HEADING.match(stripped)
    if num and len(stripped) < 200:
        is_heading = True
        dots = stripped.split()[0].count(".")
        level = min(dots + 1, 4)

    if is_heading:
        if level is None:
            if font_size >= body_size + 5:
                level = 1
            elif font_size >= body_size + 3:
                level = 2
            else:
                level = 3
        return "heading", level, ""
    return "paragraph", None, ""
