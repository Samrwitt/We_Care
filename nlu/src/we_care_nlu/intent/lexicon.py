"""Load and merge intent/topic lexicons from data/intents/.

Base file: lexicon.yaml
Overlays: lexicon_*.yaml (patterns unioned per label)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from we_care_nlu.config import package_root

_LATIN = re.compile(r"[a-zA-Z]")

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

