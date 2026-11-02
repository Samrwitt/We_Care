"""Multi-turn context: inherit or clear method/topic across dialogue turns.

- cues.py     — follow-up / clinical cue lists
- resolve.py  — resolve_context(text, entities, state, intent)
"""

from we_care_nlu.context.resolve import resolve_context

__all__ = ["resolve_context"]
