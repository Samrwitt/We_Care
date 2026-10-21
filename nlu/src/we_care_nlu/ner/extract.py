"""Gazetteer + regex slot filling. Does not invent patient facts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from we_care_nlu.config import package_root
from we_care_nlu.contracts.schemas import Entity

DOSAGE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|µg|ml|mL|IU|iu|g)\b")
AGE = re.compile(r"(ከ\s*)?(\d{1,2})\s*(ዓመት|አመት|years?|yrs?)", re.I)
DURATION = re.compile(r"(\d+)\s*(ሰዓት|ቀን|ሳምንት|ወር|ዓመት|hours?|days?|weeks?|months?)", re.I)
# "እርግዝና" inside "የእርግዝና መከላከያ" is contraception, not a pregnant patient.
_NOT_PREGNANCY = re.compile(r"እርግዝናን?\s*ለ?መከላከ")


def load_gazetteer(path: Path | None = None) -> dict[str, Any]:
    path = path or package_root() / "data" / "gazetteers" / "entities.yaml"
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class EntityExtractor:
    def __init__(self, gazetteer: dict[str, Any] | None = None):
        self.gazetteer = gazetteer or load_gazetteer()
        self._entries: list[tuple[str, str, str]] = []
        for typ, values in self.gazetteer.items():
            if not isinstance(values, dict):
                continue
            for canonical, surfaces in values.items():
                for surf in surfaces:
                    self._entries.append((surf, typ, canonical))
        self._entries.sort(key=lambda x: len(x[0]), reverse=True)

    def extract(self, text: str) -> list[Entity]:
        found: list[Entity] = []
        occupied: list[tuple[int, int]] = []
        low = text

        def overlaps(a: int, b: int) -> bool:
            return any(not (b <= s or a >= e) for s, e in occupied)

        for surface, typ, canonical in self._entries:
            start = 0
            while True:
                idx = low.lower().find(surface.lower(), start)
                if idx < 0:
                    break
                end = idx + len(surface)
                window = text[max(0, idx - 8) : min(len(text), end + 12)]
                if typ == "pregnancy_status" and _NOT_PREGNANCY.search(window):
                    start = end
                    continue
                if not overlaps(idx, end):
                    found.append(
                        Entity(
                            type=typ,
                            value=canonical,
                            surface=text[idx:end],
                            start=idx,
                            end=end,
                            confidence=0.86,
                            source="gazetteer",
                        )
                    )
                    occupied.append((idx, end))
                start = end

        for rx, typ in ((DOSAGE, "dosage"), (AGE, "age"), (DURATION, "duration")):
            for m in rx.finditer(text):
                if overlaps(m.start(), m.end()):
                    continue
                found.append(
                    Entity(
                        type=typ,
                        value=m.group(0),
                        surface=m.group(0),
                        start=m.start(),
                        end=m.end(),
                        confidence=0.9,
                        source="regex",
                    )
                )
                occupied.append((m.start(), m.end()))
        found.sort(key=lambda e: e.start)
        return found
