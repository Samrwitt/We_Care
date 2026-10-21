"""Bilingual medical glossary with longest-match lookup."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GlossaryEntry:
    english: str
    amharic: str
    category: str
    confidence: str  # verified | standard | provisional | do_not_translate
    notes: str = ""
    preserve_source: bool = False


class Glossary:
    def __init__(self, entries: list[GlossaryEntry], freeze_patterns: list[str] | None = None):
        self.entries = entries
        # Longest English phrase first so "emergency contraception" beats "contraception".
        self._sorted = sorted(entries, key=lambda e: len(e.english), reverse=True)
        self._by_en = {e.english.lower(): e for e in entries}
        self.freeze_patterns = [re.compile(p, re.I) for p in (freeze_patterns or [])]
        self._phrase_re = self._compile_phrase_regex()

    def _compile_phrase_regex(self) -> re.Pattern[str] | None:
        phrases = [re.escape(e.english) for e in self._sorted if e.english.strip()]
        if not phrases:
            return None
        # Word-boundary-ish: avoid matching inside longer tokens.
        joined = "|".join(phrases)
        return re.compile(rf"(?i)(?<![A-Za-z])(?:{joined})(?![A-Za-z])")

    @classmethod
    def load(cls, csv_path: str | Path, freeze_path: str | Path | None = None) -> "Glossary":
        path = Path(csv_path)
        entries: list[GlossaryEntry] = []
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            required = {"english", "amharic", "category", "confidence"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Glossary CSV missing columns {missing} in {path}")
            for row in reader:
                en = (row.get("english") or "").strip()
                am = (row.get("amharic") or "").strip()
                if not en:
                    continue
                conf = (row.get("confidence") or "provisional").strip().lower()
                preserve = str(row.get("preserve_source") or "").strip().lower() in {"1", "true", "yes"}
                if conf == "do_not_translate":
                    preserve = True
                entries.append(
                    GlossaryEntry(
                        english=en,
                        amharic=am,
                        category=(row.get("category") or "").strip(),
                        confidence=conf,
                        notes=(row.get("notes") or "").strip(),
                        preserve_source=preserve,
                    )
                )
        freeze: list[str] = []
        if freeze_path and Path(freeze_path).exists():
            data = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
            freeze = list(data.get("patterns") or [])
        return cls(entries, freeze)

    def lookup(self, english: str) -> GlossaryEntry | None:
        return self._by_en.get(english.strip().lower())

    def find_hits(self, text: str) -> list[GlossaryEntry]:
        if not self._phrase_re:
            return []
        found: list[GlossaryEntry] = []
        seen: set[str] = set()
        for match in self._phrase_re.finditer(text):
            key = match.group(0).lower()
            entry = self._by_en.get(key)
            if entry and entry.english.lower() not in seen:
                seen.add(entry.english.lower())
                found.append(entry)
        return found

    def unverified_in_text(self, text: str) -> list[GlossaryEntry]:
        return [e for e in self.find_hits(text) if e.confidence == "provisional"]

    def enforce_target(self, source: str, target: str) -> tuple[str, list[str]]:
        """If a verified English term is still present in the Amharic output, replace it."""
        notes: list[str] = []
        out = target
        for entry in self.find_hits(source):
            if entry.preserve_source or not entry.amharic:
                continue
            if entry.confidence not in {"verified", "standard"}:
                continue
            pattern = re.compile(rf"(?i)(?<![A-Za-z]){re.escape(entry.english)}(?![A-Za-z])")
            if pattern.search(out):
                out = pattern.sub(entry.amharic, out)
                notes.append(f"enforced:{entry.english}")
        return out, notes
