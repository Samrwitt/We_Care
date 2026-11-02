"""Lexicon-based intent and topic routing.

- lexicon.py     — load/merge YAML from data/intents/
- classifier.py  — IntentClassifier.predict(text) → intent + topic labels
"""

from we_care_nlu.intent.classifier import IntentClassifier
from we_care_nlu.intent.lexicon import load_lexicon

__all__ = ["IntentClassifier", "load_lexicon"]
