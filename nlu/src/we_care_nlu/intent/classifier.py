"""Rule-based Amharic intent/topic router; heuristic outputs are NOT probabilities.

Backward-compatible outputs: intent, intent_labels, intent_confidence,
intent_backend, confidence_status. Extra outputs: topic, topics, modifiers,
safety_cues, evidence_patterns. Safety cues require a separate contextual triage
layer and should never be treated as a diagnosis or reliable emergency classifier.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any
import yaml
from we_care_nlu.config import package_root
from we_care_nlu.preprocess.normalize import preprocess

_LATIN = re.compile(r'[a-zA-Z]')


def load_lexicon(path: Path | None = None) -> dict[str, Any]:
    path = path or package_root() / 'data' / 'intents' / 'lexicon.yaml'
    with Path(path).open(encoding='utf-8') as fh:
        result = yaml.safe_load(fh) or {}
    if not isinstance(result, dict) or 'intents' not in result:
        raise ValueError('Lexicon must be a mapping with intents')
    return result


def _matches(pattern: str, text: str) -> bool:
    p = pattern.casefold().strip()
    t = text.casefold()
    if not p:
        return False
    # Avoid substring false positives in Latin (hi inside HIV; art inside start).
    if _LATIN.search(p):
        return re.search(r'(?<![a-z])' + re.escape(p) + r'(?![a-z])', t) is not None
    return p in t


def _scan(groups: dict, text: str) -> tuple[dict[str, float], dict[str, list[str]]]:
    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    for label, spec in groups.items():
        patterns = spec.get('patterns', []) if isinstance(spec, dict) else spec
        matched = [str(p) for p in patterns if isinstance(p, str) and _matches(p, text)]
        if matched:
            evidence[label] = matched
            # The strongest explicit phrase dominates; multiple matches give a small bonus.
            scores[label] = max(len(p) for p in matched) + min(3, len(matched) - 1) * 2
    return scores, evidence


class IntentClassifier:
    def __init__(self, lexicon: dict[str, Any] | None = None):
        self.lexicon = lexicon if lexicon is not None else load_lexicon()
        self.backend = 'structured_lexicon_v2'

    def predict(self, text: str) -> dict[str, Any]:
        normalized = preprocess(text).normalized
        intents, intent_matches = _scan(self.lexicon.get('intents', {}), normalized)
        topics, topic_matches = _scan(self.lexicon.get('topics', {}), normalized)
        modifiers, modifier_matches = _scan(self.lexicon.get('modifiers', {}), normalized)
        safety, safety_matches = _scan(self.lexicon.get('safety_triggers', {}), normalized)
        priority = self.lexicon.get('priority', [])
        rank = {label: i for i, label in enumerate(priority)}

        # Topics are not intents. A question about HIV symptoms is a symptoms intent.
        # Context markers should not override an explicit question type.
        for label in ('follow_up', 'acknowledgement'):
            if label in intents and len(intents) > 1:
                intents[label] *= 0.25
        if 'definition' in intents and len(intents) > 1:
            # Very generic "what is" should not defeat explicit clinical aspects.
            intents['definition'] *= 0.55
        if 'missed_dose' in intents and 'treatment' in intents:
            # "ክኒን ከተረሳ ምን ላድርግ?" is missed-dose, not general treatment.
            intents['treatment'] *= 0.35
        if 'family_planning_methods' in intents and len(intents) > 1:
            # A methods catalogue query is specific; but not a global override.
            pass
        ordered = sorted(intents, key=lambda x: (-intents[x], rank.get(x, 999), x))
        primary = ordered[0] if ordered else self.lexicon.get('fallback_intent', 'general_medical_info')
        ordered_topics = sorted(topics, key=lambda x: (-topics[x], x))
        ordered_modifiers = sorted(modifiers, key=lambda x: (-modifiers[x], x))
        ordered_safety = sorted(safety, key=lambda x: (-safety[x], x))

        # Legacy schema requires a float. This is a compatibility placeholder,
        # deliberately not a claimed statistical confidence value.
        return {
            'intent': primary,
            'intent_labels': ordered or [primary],
            'intent_confidence': 0.0,
            'intent_backend': self.backend,
            'confidence_status': 'requires_calibration',
            'topic': ordered_topics[0] if ordered_topics else None,
            'topics': ordered_topics,
            'modifiers': ordered_modifiers,
            'safety_cues': ordered_safety,
            'evidence_patterns': {
                'intents': intent_matches,
                'topics': topic_matches,
                'modifiers': modifier_matches,
                'safety_triggers': safety_matches,
            },
        }