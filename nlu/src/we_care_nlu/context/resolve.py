"""Resolve follow-ups using dialogue-manager state. Does not guess missing slots."""

from __future__ import annotations

from we_care_nlu.contracts.schemas import ContextEntity, ContextState, Entity
from we_care_nlu.preprocess.normalize import preprocess

FOLLOW_CUES = (
    "የጎንዮሽ ጉዳቱስ",
    "ጉዳቱስ",
    "እንዴትስ",
    "ለምንስ",
    "እና ከዚያ",
    "what about",
    "and the",
    "its side",
)


def looks_like_followup(text: str) -> bool:
    pre = preprocess(text)
    if any(c in pre.normalized for c in FOLLOW_CUES):
        return True
    # Very short question with no new method/condition entity often needs prior topic.
    return len(pre.normalized) < 28 and pre.normalized.endswith("?")


def resolve_context(
    text: str,
    entities: list[Entity],
    context: ContextState | None,
    intent: str,
) -> tuple[list[Entity], dict]:
    info = {
        "resolved_from_previous_turn": False,
        "assumed_missing_slot": False,
        "needs_clarification": False,
        "reason": "",
    }
    if context is None:
        return entities, info

    has_history = bool(
        context.last_entities or context.last_method or context.last_topic or context.last_intents
    )
    if not has_history:
        return entities, info

    has_method = any(e.type == "contraceptive_method" for e in entities)
    follow = looks_like_followup(text) or intent in {"follow_up", "side_effects", "contraindications", "eligibility"}

    if has_method or not follow:
        return entities, info

    prior_methods = [e for e in context.last_entities if e.type == "contraceptive_method"]
    if context.last_method and not prior_methods:
        prior_methods = [
            ContextEntity(type="contraceptive_method", value=context.last_method, surface=context.last_method)
        ]

    if len(prior_methods) == 1:
        m = prior_methods[0]
        value = m.value
        surface = m.surface or value
        entities = list(entities) + [
            Entity(
                type="contraceptive_method",
                value=value,
                surface=surface,
                start=-1,
                end=-1,
                confidence=0.7,
                source="context",
            )
        ]
        info["resolved_from_previous_turn"] = True
        info["reason"] = "Attached the single prior contraceptive method to a follow-up."
        return entities, info

    if len(prior_methods) > 1:
        info["needs_clarification"] = True
        info["reason"] = "Multiple prior methods; not assuming which one the follow-up refers to."
        return entities, info

    if follow and not prior_methods:
        info["needs_clarification"] = True
        info["reason"] = "Follow-up has no method/condition in this turn or in supplied state."
        return entities, info

    return entities, info
