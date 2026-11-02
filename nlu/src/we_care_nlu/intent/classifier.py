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


def _merge_pattern_groups(base: dict[str, Any] | None, extra: dict[str, Any] | None) -> dict[str, Any]:
    """Union patterns per label; extras may add new labels without dropping base ones."""
    out: dict[str, Any] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in (base or {}).items()}
    for label, spec in (extra or {}).items():
        extra_patterns = spec.get("patterns", []) if isinstance(spec, dict) else (spec or [])
        extra_patterns = [str(p) for p in extra_patterns if isinstance(p, str) and str(p).strip()]
        if label not in out:
            out[label] = {"patterns": list(extra_patterns)} if not isinstance(spec, dict) else {
                **spec,
                "patterns": list(extra_patterns),
            }
            continue
        cur = out[label]
        if not isinstance(cur, dict):
            cur = {"patterns": list(cur) if isinstance(cur, list) else []}
            out[label] = cur
        cur_patterns = [str(p) for p in (cur.get("patterns") or []) if isinstance(p, str)]
        seen = {p.casefold() for p in cur_patterns}
        for pat in extra_patterns:
            key = pat.casefold()
            if key in seen:
                continue
            cur_patterns.append(pat)
            seen.add(key)
        merged = dict(cur)
        if isinstance(spec, dict):
            for k, v in spec.items():
                if k == "patterns":
                    continue
                merged.setdefault(k, v)
        merged["patterns"] = cur_patterns
        out[label] = merged
    return out


def load_lexicon(path: Path | None = None) -> dict[str, Any]:
    root = package_root() / "data" / "intents"
    path = path or (root / "lexicon.yaml")
    with Path(path).open(encoding="utf-8") as fh:
        result = yaml.safe_load(fh) or {}
    if not isinstance(result, dict) or "intents" not in result:
        raise ValueError("Lexicon must be a mapping with intents")
    # Overlay lexicon_*.yaml (extensive paraphrases) onto the base file.
    for overlay in sorted(root.glob("lexicon_*.yaml")):
        if overlay.resolve() == Path(path).resolve():
            continue
        extra = yaml.safe_load(overlay.read_text(encoding="utf-8")) or {}
        if not isinstance(extra, dict):
            continue
        for section in ("intents", "topics", "modifiers", "safety_triggers"):
            if section in extra and isinstance(extra[section], dict):
                result[section] = _merge_pattern_groups(result.get(section), extra[section])
        if isinstance(extra.get("priority"), list):
            base_pri = list(result.get("priority") or [])
            seen = set(base_pri)
            for label in extra["priority"]:
                if label not in seen:
                    base_pri.append(label)
                    seen.add(label)
            result["priority"] = base_pri
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