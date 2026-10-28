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

CONDITION_TYPES = {"condition", "breastfeeding_status", "pregnancy_status"}
# Intents that are complete topical asks — never force "which FP method?".
STANDALONE_INTENTS = {
    "definition",
    "family_planning_methods",
    "access",
    "how_it_works",
    "duration",
    "emergency",
    "greeting",
    "conversation_end",
    "out_of_domain",
}
CARRY_INTENTS = {
    "eligibility",
    "contraindications",
    "side_effects",
    "pregnancy_breastfeeding",
    "follow_up",
    "treatment",
    "symptoms",
    "prevention",
    "causes",
    "screening",
    "medication",
    "dosage",
}


def looks_like_followup(text: str) -> bool:
    pre = preprocess(text)
    if any(c in pre.normalized for c in FOLLOW_CUES):
        return True
    # Elliptical follow-ups only (e.g. "ጎንዮሽ?" / "ለምን?"). Full questions like
    # "ክሪፕቲክ እርግዝና ምንድን ነው?" must not inherit "which method?" from prior turns.
    body = pre.normalized.strip().rstrip("?؟።")
    toks = [t for t in body.split() if t]
    return len(toks) <= 2 and pre.normalized.endswith(("?", "؟"))


def _as_entity(item: ContextEntity | Entity | dict) -> Entity:
    if isinstance(item, Entity):
        return item
    if isinstance(item, ContextEntity):
        return Entity(
            type=item.type,
            value=item.value,
            surface=item.surface or item.value,
            start=-1,
            end=-1,
            confidence=0.7,
            source="context",
        )
    return Entity(
        type=str(item.get("type") or ""),
        value=str(item.get("value") or ""),
        surface=str(item.get("surface") or item.get("value") or ""),
        start=-1,
        end=-1,
        confidence=0.7,
        source="context",
    )


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
        "inherited_intent": None,
    }
    if context is None:
        return entities, info

    has_history = bool(
        context.last_entities or context.last_method or context.last_topic or context.last_intents
    )
    if not has_history:
        return entities, info

    entities = list(entities)
    has_method = any(e.type == "contraceptive_method" for e in entities)
    has_condition = any(e.type in CONDITION_TYPES for e in entities)
    prior_methods = [e for e in context.last_entities if getattr(e, "type", None) == "contraceptive_method" or (isinstance(e, dict) and e.get("type") == "contraceptive_method")]
    prior_conditions = [
        e
        for e in context.last_entities
        if getattr(e, "type", None) in CONDITION_TYPES
        or (isinstance(e, dict) and e.get("type") in CONDITION_TYPES)
    ]
    if context.last_method and not prior_methods:
        prior_methods = [
            ContextEntity(type="contraceptive_method", value=context.last_method, surface=context.last_method)
        ]

    prior_intent = ""
    if context.last_intents:
        prior_intent = str(context.last_intents[0] or "")
    elif context.last_topic:
        prior_intent = str(context.last_topic)

    # Carry condition/BF status into a method-only follow-up (e.g. HIV ask → "መርፌ").
    if prior_conditions and not has_condition:
        for item in prior_conditions:
            entities.append(_as_entity(item))
        has_condition = True
        info["resolved_from_previous_turn"] = True
        info["reason"] = "Attached prior condition/breastfeeding status to this turn."

    bare_method = has_method and len((text or "").strip()) <= 24
    follow = (
        looks_like_followup(text)
        or intent in CARRY_INTENTS
        or (bare_method and prior_intent in CARRY_INTENTS)
        or (bare_method and bool(prior_conditions))
    )

    # New topical ask (definition / methods list / access / …) — answer on its own terms.
    if intent in STANDALONE_INTENTS and not bare_method and not looks_like_followup(text):
        return entities, info

    # Eligibility/condition asks are answerable without naming a method (e.g. HIV + FP).
    if has_condition and intent in {"eligibility", "contraindications", "pregnancy_breastfeeding", "hiv"}:
        return entities, info
    if has_condition and bare_method:
        # Keep clinical aspect from prior turn when user only names a method.
        if prior_intent in CARRY_INTENTS:
            info["inherited_intent"] = prior_intent
        info["resolved_from_previous_turn"] = True
        if not info["reason"]:
            info["reason"] = "Method follow-up under prior condition/eligibility context."
        return entities, info

    if has_method or not follow:
        return entities, info

    if len(prior_methods) == 1:
        entities.append(_as_entity(prior_methods[0]))
        info["resolved_from_previous_turn"] = True
        info["reason"] = "Attached the single prior contraceptive method to a follow-up."
        if prior_intent in CARRY_INTENTS and intent not in CARRY_INTENTS:
            info["inherited_intent"] = prior_intent
        return entities, info

    if len(prior_methods) > 1:
        info["needs_clarification"] = True
        info["reason"] = "Multiple prior methods; not assuming which one the follow-up refers to."
        return entities, info

    # Do not ask "which method?" when the user already named a condition (HIV, BF, …).
    if follow and not prior_methods and not has_condition and not prior_conditions:
        # Continue a prior medical *topic* (e.g. HIV symptoms → how is it treated?).
        if context.last_topic:
            info["resolved_from_previous_turn"] = True
            info["inherited_topic"] = context.last_topic
            info["reason"] = "Follow-up continues prior medical topic without requiring a method."
            return entities, info
        info["needs_clarification"] = True
        info["reason"] = "Follow-up has no method/condition in this turn or in supplied state."
        return entities, info

    return entities, info
