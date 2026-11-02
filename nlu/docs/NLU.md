# NLU core (understanding + retrieval)

This is the non-HTTP heart of Component 2: turn a user question into **intent, topic, entities, and ranked evidence**. Reply text is built later (`reply_pipeline.py`).

## Pipeline stages

```
raw text
   → preprocess/normalize.py     NFC, spelling, Ethiopic punct
   → intent/classifier.py        question type (definition, side_effects, …)
   → intent topics (same pass)   subject (condom, hiv, irregular_menses, …)
   → ner/extract.py              methods, conditions, …
   → context/resolve.py          multi-turn carry / clear
   → retrieval/engine.py         BM25 + dense + relevance scoring
   → safety/rules.py             emergency / conflict flags
```

`pipeline.NLUEngine.analyze()` runs that sequence and returns `AnalyzeResponse`.

## Package cheat sheet

| Package | Read first | Notes |
|---------|------------|-------|
| `preprocess/` | `normalize.py` | Keep clinical negation; light spelling only |
| `intent/` | `classifier.py`, `lexicon.py` | YAML in `data/intents/` (+ `lexicon_*.yaml` overlays) |
| `ner/` | `extract.py` | Gazetteer + regex; not a neural NER |
| `context/` | `resolve.py`, `cues.py` | Follow-ups; clinical cues drop sticky FP methods |
| `retrieval/` | `engine.py` then `relevance.py` | Facade; internals split below |
| `retrieval/query_facets.py` | distinctive tokens / soft topics | “What is this ask about?” |
| `retrieval/chunk_signals.py` | method/BF/HIV/curated IDs | Features on a chunk |
| `retrieval/scoring.py` | `answer_quality`, `relevant_enough` | Rank / filter candidates |
| `retrieval/expand.py` | query expansion surfaces | Intent/topic needles |
| `terminology/` | `index.py` | KB-derived concepts (Layer B topics) |
| `safety/` | `rules.py` | Emergency heuristics (not diagnosis) |
| `kb/` | `ingest.py` | Load JSON/DOCX into `ChunkRecord`s |
| `embeddings/` | `registry.py` | `hash` for tests; MiniLM/etc for serve |
| `contracts/` | `schemas.py` | Shared request/response models |

## Intent vs topic (important)

- **Intent** = what kind of question (`definition`, `treatment`, `missed_dose`).
- **Topic** = what it is about (`condom`, `cryptic_pregnancy`, `hiv`).

Both come from the lexicon scanner; topics also come from the terminology concept index when the lexicon does not fire.

## Multi-turn gotcha

After “injectable + breastfeeding”, a later “ሲያመኝ ምን ላድርግ?” must **not** keep Depo+BF.  
`context/cues.py` lists clinical cues that clear prior method slots; `resolve_context()` applies that rule.

## Tests to run after NLU edits

```bash
cd nlu
PYTHONPATH=src pytest tests/test_lexicon.py tests/test_intent_entities.py \
  tests/test_preprocess.py tests/test_general_relevance.py \
  tests/test_retrieval_context.py -q
```
