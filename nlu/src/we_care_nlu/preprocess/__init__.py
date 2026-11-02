"""Text normalization for Amharic/English asks (NFC, light spelling, negation kept)."""

from we_care_nlu.preprocess.normalize import detect_language, has_negation, preprocess

__all__ = ["detect_language", "has_negation", "preprocess"]
