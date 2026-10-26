#!/usr/bin/env python3
"""Scrape WeCare Amharic blog articles into a static KB JSON file.

Usage (from nlu/):
  python scripts/scrape_wecare_articles.py
  python scripts/scrape_wecare_articles.py --out data/kb/wecare_articles_am.json
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

ARTICLE_URLS = [
    "https://wecare.et/wecare/articles/649",
    "https://wecare.et/wecare/articles/716",
    "https://wecare.et/wecare/articles/704",
    "https://wecare.et/wecare/articles/703",
    "https://wecare.et/wecare/articles/690",
    "https://wecare.et/wecare/articles/680",
    "https://wecare.et/wecare/articles/677",
    "https://wecare.et/wecare/articles/670",
    "https://wecare.et/wecare/articles/651",
    "https://wecare.et/wecare/articles/718",
]

_SPACE = re.compile(r"\s+")
_ETH = re.compile(r"[\u1200-\u137F]")
_SENT_SPLIT = re.compile(r"(?<=[።.!?\n])\s+")
_MAX_CHARS = 480


class _ArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._capture_title = False
        self._in_main = False
        self._skip = 0
        self._tag_stack: list[str] = []
        self._current_heading = ""
        self.sections: list[dict[str, str]] = []
        self._buf: list[str] = []
        self._seen_h1 = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_d = {k: (v or "") for k, v in attrs}
        self._tag_stack.append(tag)
        if tag in {"script", "style", "nav", "footer", "header", "form", "noscript"}:
            self._skip += 1
            return
        if self._skip:
            return
        classes = attrs_d.get("class", "")
        if tag == "title":
            self._capture_title = True
        if tag in {"article", "main"} or "article" in classes or "content" in classes:
            self._in_main = True
        if tag in {"h1", "h2", "h3"} and (self._in_main or self._seen_h1 or tag == "h1"):
            self._flush_paragraph()
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer", "header", "form", "noscript"}:
            self._skip = max(0, self._skip - 1)
        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()
        elif tag in self._tag_stack:
            while self._tag_stack and self._tag_stack[-1] != tag:
                self._tag_stack.pop()
            if self._tag_stack:
                self._tag_stack.pop()
        if self._skip:
            return
        if tag == "title":
            self._capture_title = False
        if tag in {"h1", "h2", "h3"}:
            heading = _SPACE.sub(" ", "".join(self._buf)).strip()
            self._buf = []
            if heading:
                if tag == "h1":
                    self._seen_h1 = True
                    self._in_main = True
                    if not self.title or "WeCare" in self.title:
                        self.title = heading
                self._current_heading = heading
                if not any(s["section"] == heading and not s["text"] for s in self.sections):
                    self.sections.append({"section": heading, "text": ""})
        if tag in {"p", "li", "div", "section"}:
            self._flush_paragraph()

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._capture_title:
            self.title += data
            return
        if not (self._in_main or self._seen_h1 or self._capture_title):
            # Still collect body-ish text after first h1 appears via loose mode later.
            if not self._seen_h1:
                return
        self._buf.append(data)

    def _flush_paragraph(self) -> None:
        text = _SPACE.sub(" ", "".join(self._buf)).strip()
        self._buf = []
        if not text or len(text) < 20:
            return
        if not _ETH.search(text):
            return
        # Skip UI chrome.
        low = text.lower()
        if any(x in low for x in ("sign in", "sign up", "leave a comment", "related articles", "©")):
            return
        if self.sections and self.sections[-1]["section"] == self._current_heading:
            prev = self.sections[-1]["text"]
            self.sections[-1]["text"] = f"{prev} {text}".strip() if prev else text
        else:
            self.sections.append({"section": self._current_heading or self.title or "body", "text": text})


def _fetch(url: str) -> tuple[str, str]:
    # Canonical short form redirects to the slug URL.
    req = Request(url, headers={"User-Agent": "WeCareNLU-KB-Scraper/1.0"})
    with urlopen(req, timeout=60) as resp:  # noqa: S310 — fixed WeCare hosts only
        raw = resp.read()
        final = resp.geturl()
        charset = resp.headers.get_content_charset() or "utf-8"
    text = raw.decode(charset, errors="replace")
    return text, final


def _split_passages(text: str, max_chars: int = _MAX_CHARS) -> list[str]:
    text = _SPACE.sub(" ", (text or "").strip())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    if len(parts) <= 1:
        return [text[i : i + max_chars].strip() for i in range(0, len(text), max_chars)]
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for part in parts:
        extra = len(part) + (1 if buf else 0)
        if buf and size + extra > max_chars:
            out.append(" ".join(buf))
            buf, size = [part], len(part)
        else:
            buf.append(part)
            size += extra
    if buf:
        out.append(" ".join(buf))
    return out


def scrape_article(url: str) -> dict:
    html, final_url = _fetch(url)
    parser = _ArticleParser()
    parser.feed(html)
    parser.close()
    title = _SPACE.sub(" ", html_lib.unescape(parser.title)).strip()
    title = re.sub(r"\s*\|\s*WeCare\s*$", "", title, flags=re.I).strip() or f"article-{url.rstrip('/').split('/')[-1]}"
    article_id = re.search(r"/articles/(\d+)", final_url or url)
    doc_id = f"wecare-{article_id.group(1)}" if article_id else f"wecare-{url.rstrip('/').split('/')[-1]}"
    segments: list[dict] = []
    seg_i = 0
    for block in parser.sections:
        section = _SPACE.sub(" ", block["section"]).strip() or title
        for passage in _split_passages(block["text"]):
            if not _ETH.search(passage):
                continue
            if passage == section or passage.startswith(section) and len(passage) < len(section) + 5:
                continue
            seg_i += 1
            segments.append(
                {
                    "segment_id": f"{doc_id}-s{seg_i}",
                    "kind": "paragraph",
                    "section": section,
                    "source_text": "",
                    "translated_text": passage,
                    "source_url": final_url or url,
                    "flags": "wecare_article,source_amharic",
                }
            )
    return {
        "document_name": f"{doc_id}.html",
        "title": title,
        "source_url": final_url or url,
        "article_id": article_id.group(1) if article_id else "",
        "segments": segments,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "data" / "kb" / "wecare_articles_am.json",
        help="Output JSON path",
    )
    parser.add_argument("--urls", nargs="*", default=ARTICLE_URLS)
    args = parser.parse_args()
    documents = []
    for url in args.urls:
        print(f"fetch {url}", file=sys.stderr)
        try:
            doc = scrape_article(url)
        except Exception as exc:  # noqa: BLE001 — report and continue remaining URLs
            print(f"FAIL {url}: {exc}", file=sys.stderr)
            return 1
        n = len(doc["segments"])
        print(f"  -> {doc['title'][:60]} ({n} segments)", file=sys.stderr)
        if n == 0:
            print(f"FAIL {url}: no Amharic segments extracted", file=sys.stderr)
            return 1
        documents.append(doc)
    payload = {
        "kb_kind": "wecare_articles_am",
        "language": "am",
        "documents": documents,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(documents)} documents)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
