# Backend map (for newcomers)

This service answers reproductive-health questions in **Amharic or English** by retrieving guideline/article text and composing a short reply. It does **not** call a chat LLM.

OpenAPI docs when the server is up: `http://127.0.0.1:8088/docs`

## Start here

| If you want to… | Open this |
|-----------------|-----------|
| See the HTTP API | `src/we_care_nlu/api/app.py` |
| Server startup / KB load | `src/we_care_nlu/api/lifecycle.py` |
| Main “ask → answer” logic | `src/we_care_nlu/pipeline.py` (`analyze`) + `reply_pipeline.py` (`reply`) |
| Request/response shapes | `src/we_care_nlu/contracts/schemas.py` |
| Intent labels (definition, symptoms, …) | `src/we_care_nlu/intent/classifier.py` + `data/intents/` |
| Conversation follow-ups | `src/we_care_nlu/context/resolve.py` |
| Search + ranking | `src/we_care_nlu/retrieval/` |
| Template answers when KB misses | `src/we_care_nlu/translation/advice.py` |
| Chat UI | `src/we_care_nlu/api/static/README.md` |

## Request flow

```
POST /nlp/reply  { "text": "ኮንዶም ምንድን ነው?", "context": … }
        │
        ▼
 lifecycle.get_engine()     → shared NLUEngine (loaded at startup)
        │
        ▼
 NLUEngine.reply()          → reply_pipeline.py
        │
        ├─ map language / search text
        ├─ NLUEngine.analyze()   → preprocess → intent → entities → context → retrieve
        ├─ pick best span        → translation/reply.py + retrieval/relevance.py
        └─ compose or advice     → compose.py / advice.py
        │
        ▼
 ReplyResponse  (amharic_reply, english_reply, evidence_id, safety, …)
```

`POST /nlp/analyze` stops after analysis (no user-facing reply text).

## Package layout (`src/we_care_nlu/`)

| Package | Responsibility |
|---------|----------------|
| `api/` | FastAPI routes, engine lifecycle, static chat UI |
| `contracts/` | Pydantic models shared with other components |
| `preprocess/` | Normalize Amharic/English text |
| `intent/` | Lexicon intent + topic routing |
| `ner/` | Gazetteer / regex entities |
| `context/` | Multi-turn method/topic inheritance |
| `retrieval/` | BM25 + dense search, expand, relevance scoring |
| `translation/` | Phrase map, MT client, compose, advice, span tidy |
| `kb/` | Ingest JSON/DOCX/PDF into chunks |
| `embeddings/` | Embedder registry + cache |
| `terminology/` | Concept index from KB |
| `safety/` | Emergency / conflict flags |
| `eval/` | Offline QA harness |

## Data on disk

| Path | What |
|------|------|
| `data/kb/*.json` | Knowledge-base chunk files loaded at startup |
| `data/intents/` | Intent/topic lexicons (`lexicon.yaml` + overlays) |
| `data/gazetteers/` | Entity surfaces |
| `data/synonyms/` | Medical synonym lists |
| `configs/default.yaml` | Embeddings, retrieval thresholds |

## Design rules (short)

1. **No generative chat model** — answers come from retrieved text or fixed advice templates.
2. **Intent ≠ topic** — intent is question type (`definition`); topic is subject (`condom`, `hiv`).
3. **User-facing replies** avoid article source dumps; booking link + 9394 appear on no-match advice / welcome UI.
4. **Clinical follow-ups** clear sticky contraceptive context (see `context/resolve.py`).

## Tests

```bash
cd nlu && PYTHONPATH=src pytest -q
```

Prefer `WE_CARE_NLU_MODEL=hash` in tests (no neural download).
