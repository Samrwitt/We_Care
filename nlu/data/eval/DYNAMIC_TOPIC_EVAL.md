# Dynamic topic evaluation (human-reviewed)

Target: ≥300 Amharic items with **human-reviewed** intent, topic/concept, and evidence ids.

Do **not** treat auto-generated labels as ground truth.

## Schema (`dynamic_topics.jsonl`)

Each line:

```json
{
  "id": "dt_001",
  "question_am": "…",
  "intent": "symptoms",
  "topic": "hiv",
  "concept_ids": [],
  "evidence_ids": ["…"],
  "category": "lexicon_topic|kb_only|synonym|follow_up|negation|unsupported",
  "reviewed_by": "",
  "notes": ""
}
```

## Metrics to report (when labels exist)

- Intent macro-F1
- Concept recognition F1 (approved index)
- Retrieval Recall@1/5/10, MRR
- Context resolution accuracy
- Unsupported / safety trigger recall

Compare a baseline commit (lexicon-only topic) vs concept-index expansion using the same reviewed file.

A starter script can score a reviewed jsonl once populated: `scripts/eval_dynamic_topics.py` (add when the labeled set is ready).
