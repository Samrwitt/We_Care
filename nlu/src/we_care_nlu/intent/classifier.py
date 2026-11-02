"""Rule-based Amharic intent/topic router.

Outputs are heuristic labels — not calibrated probabilities.
See intent/lexicon.py for YAML loading. Data lives in data/intents/.
"""

from __future__ import annotations

from typing import Any

from we_care_nlu.intent.lexicon import _scan, load_lexicon
from we_care_nlu.preprocess.normalize import preprocess


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
        # "What is X?" about one method/condition should stay definition, not methods list.
        if 'definition' in intents and 'family_planning_methods' in intents:
            catalog_cues = (
                'ዘዴዎች', 'አማራጮች', 'methods', 'options', 'what methods', 'which methods',
                'ምን ዓይነት', 'ምን አማራጮች', 'types of',
            )
            low_norm = normalized.casefold()
            if not any(cue in low_norm for cue in catalog_cues):
                intents['family_planning_methods'] *= 0.2
        # Prefer specific method/condition topics over the broad family_planning umbrella.
        if 'family_planning' in topics and len(topics) > 1:
            specific = {
                'contraceptive_implant', 'contraceptive_injection', 'contraceptive_pill',
                'iud', 'condom', 'emergency_contraception', 'sterilization', 'hiv', 'sti',
                'menstruation', 'irregular_menses', 'breastfeeding', 'pregnancy',
            }
            if any(t in topics for t in specific):
                topics['family_planning'] *= 0.35
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