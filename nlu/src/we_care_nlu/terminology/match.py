"""Surface matching against the approved concept index (longest match wins)."""

from __future__ import annotations

from we_care_nlu.terminology.models import MedicalConcept


def match_concepts_in_text(
    text: str,
    surface_map: list[tuple[str, MedicalConcept]],
    *,
    limit: int = 8,
) -> list[MedicalConcept]:
    if not text or not surface_map:
        return []
    low = text
    occupied: list[tuple[int, int]] = []
    found: list[MedicalConcept] = []
    seen_ids: set[str] = set()

    def overlaps(a: int, b: int) -> bool:
        return any(not (b <= s or a >= e) for s, e in occupied)

    for surface, concept in surface_map:
        if not surface or len(surface) < 2:
            continue
        start = 0
        while True:
            idx = low.lower().find(surface.lower(), start)
            if idx < 0:
                break
            end = idx + len(surface)
            if not overlaps(idx, end) and concept.concept_id not in seen_ids:
                found.append(concept)
                seen_ids.add(concept.concept_id)
                occupied.append((idx, end))
                if len(found) >= limit:
                    return found
            start = end
    return found
