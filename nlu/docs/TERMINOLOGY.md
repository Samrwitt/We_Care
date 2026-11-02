# Dynamic medical terminology (Layer B)

Conversational **intent** stays in `data/intents/lexicon.yaml` (definition, symptoms, treatment, …).

Medical **topics/concepts** are indexed separately so new KB documents expand coverage without new intent rules.

## Files

| Path | Role |
|------|------|
| `data/terminology/approved.json` | Seeded/approved concepts used at query time |
| `data/terminology/candidates.jsonl` | Auto-extracted candidates (`pending_review`) |
| `src/we_care_nlu/terminology/` | Extract, index, match, KB audit |

## Workflow

```bash
cd nlu
# Seed gazetteer → approved; scan all data/kb/*.json → candidates
PYTHONPATH=src we-care-nlu extract-terminology

# Coverage report (failed docs, empty chunks, terminology counts)
PYTHONPATH=src we-care-nlu kb-audit --out /tmp/kb_audit.json
```

HTTP: `GET /kb/coverage` (when the API is up).

**Promotion:** edit/promote a candidate into `approved.json` (or call `ConceptIndex.promote_candidate`) before it participates in synonym expansion. Pending candidates never expand queries.

## Query path

1. Lexicon → conversational intent (+ optional lexicon topic).
2. Concept index → match approved surfaces → expansion terms + `medical_concepts`.
3. Hybrid BM25 + dense retrieval with concept-aware expansion.
4. Context may carry `last_topic` for follow-ups (`እንዴት ይታከማል?`) without forcing “which method?”.

## Safety / limits

- English–Amharic co-occurrence in a title is a **candidate link**, not a verified translation.
- Concept mention ≠ sufficient evidence to answer; abstention / clarify still apply.
- No generative models; replies remain grounded spans.
