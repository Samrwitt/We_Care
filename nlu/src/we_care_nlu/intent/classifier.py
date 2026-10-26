"""Lexicon + prototype intent tagging. Not a pretrained medical classifier."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from we_care_nlu.config import package_root
from we_care_nlu.preprocess.normalize import preprocess


def _pattern_in(pattern: str, text: str) -> bool:
    pat = pattern.lower()
    # Latin words must match a whole word, so "hi" does not match inside "hiv".
    if re.fullmatch(r"[a-z]+", pat):
        return re.search(rf"(?<![a-z]){re.escape(pat)}(?![a-z])", text) is not None
    return pat in text


def load_lexicon(path: Path | None = None) -> dict[str, Any]:
    path = path or package_root() / "data" / "intents" / "lexicon.yaml"
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class IntentClassifier:
    def __init__(self, lexicon: dict[str, Any] | None = None):
        self.lexicon = lexicon or load_lexicon()
        self.backend = "lexicon_prototype"

    def predict(self, text: str) -> dict[str, Any]:
        pre = preprocess(text)
        norm = pre.normalized
        low = norm.lower()
        hits: list[tuple[str, int]] = []
        for name, spec in (self.lexicon.get("intents") or {}).items():
            score = 0
            for pat in spec.get("patterns") or []:
                if _pattern_in(pat, low):
                    score += max(1, len(pat) // 4)
            if score:
                hits.append((name, score))
        hits.sort(key=lambda x: x[1], reverse=True)
        priority = self.lexicon.get("priority") or []
        if hits:
            # Stable: highest score, then configured priority.
            best_score = hits[0][1]
            tied = [h for h in hits if h[1] == best_score]
            tied.sort(key=lambda x: priority.index(x[0]) if x[0] in priority else 99)
            primary = tied[0][0]
            labels = [h[0] for h in hits]
            conf = min(0.93, 0.45 + 0.08 * best_score)
            status = "requires_calibration"
        else:
            primary = "general_medical_info"
            labels = [primary]
            conf = 0.25
            status = "requires_training_data"
        if pre.has_negation and "contraindications" not in labels:
            if any(p in low for p in ("አይመከርም", "የተከለከለ", "contraindicat", "must not")):
                labels.insert(0, "contraindications")
                primary = "contraindications"
        # "ምንድን ነው" on a named method is a definition, unless another clinical
        # aspect (side effects, contraindication, eligibility, catalogue) is present.
        aspect_override = {
            "side_effects",
            "contraindications",
            "eligibility",
            "family_planning_methods",
            "genital_ulcer",
            "vaginal_discharge",
            "hiv",
            "sti",
            "menstruation",
            "pregnancy",
            "emergency",
        }
        if "definition" in {h[0] for h in hits} and not (aspect_override & set(labels)):
            if ("ምንድን ነው" in low or "what is" in low) and not any(
                k in low for k in ("ዘዴዎች", "አማራጮች", "methods", "options")
            ):
                primary = "definition"
                if "definition" in labels:
                    labels.remove("definition")
                labels.insert(0, "definition")
        return {
            "intent": primary,
            "intent_labels": labels,
            "intent_confidence": round(conf, 3),
            "intent_backend": self.backend,
            "confidence_status": status,
        }
