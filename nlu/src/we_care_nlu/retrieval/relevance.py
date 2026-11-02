"""Retrieval relevance — public facade.

Implementation is split for readability:
  - query_facets.py  — distinctive tokens / soft topics
  - chunk_signals.py — method, BF, HIV, curated chunk hints
  - scoring.py       — boost, force-include, answer_quality, relevant_enough

Import from this module in application code (stable path).
"""

from we_care_nlu.retrieval.chunk_signals import *  # noqa: F403
from we_care_nlu.retrieval.query_facets import *  # noqa: F403
from we_care_nlu.retrieval.scoring import *  # noqa: F403

# Explicit re-exports for star-import clarity / IDE jump-to-def.
from we_care_nlu.retrieval.chunk_signals import (  # noqa: F401
    ACCESS_SERVICE_CUES,
    BF_CUES,
    CLINICAL_CONDITION_INTENTS,
    FP_SCOPED_INTENTS,
    HIV_THERAPY_CUES,
    LAM_CUES,
    METHOD_LIST_CUES,
    blob_of,
    curated_intent_match,
    entity_condition_values,
    entity_method_values,
    has_breastfeeding,
    intent_needle_hit,
    is_demo_chunk,
    method_hit,
)
from we_care_nlu.retrieval.query_facets import (  # noqa: F401
    _distinctive_aliases,
    distinctive_coverage,
    distinctive_overlap,
    distinctive_query_tokens,
    soft_topic_groups,
    soft_topic_hit,
    unspecified_distractor,
)
from we_care_nlu.retrieval.scoring import (  # noqa: F401
    answer_quality,
    low_quality_penalty,
    relevance_boost,
    relevant_enough,
    should_force_include,
)
