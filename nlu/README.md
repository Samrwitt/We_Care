# We Care NLU — Amharic Medical Understanding and Retrieval

**Component 2** of the non-generative Amharic medical chatbot.

This service understands Amharic questions, extracts medical slots, and retrieves evidence from translated guidelines. It returns **JSON only**. It does not call GPT/Gemini/Claude/Llama/Qwen and does not decode chatbot replies.

Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

It consumes Component 1 alignment JSON (`*.alignment.json`) and will feed Component 3 (rule-based dialogue manager).

## What works without downloading neural weights

- Amharic preprocessing
- Lexicon intent tagging
- Gazetteer/regex entities
- Context resolution from dialogue-manager state
- BM25 + dense (or hash-vector) hybrid retrieval (hash is **not** semantic)
- Rule-based emergency detection
- FastAPI
- Tests

Set `WE_CARE_NLU_MODEL=labse` or `amharic_base` after installing `sentence-transformers` and downloading weights for real dense retrieval.

## Install

```bash
cd nlu
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Run the API

```bash
we-care-nlu serve --port 8088
# or: uvicorn we_care_nlu.api.app:app --port 8088
```

```bash
curl -X POST localhost:8088/documents/ingest \
  -H 'Content-Type: application/json' \
  -d '{"path":"tests/fixtures/mini_kb.json","review_status":"unreviewed"}'

curl -X POST localhost:8088/nlp/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"የእርግዝና መከላከያ ዘዴዎች ምንድን ናቸው?"}'
```

Ingest a Phase 1 file:

```bash
curl -X POST localhost:8088/documents/ingest \
  -H 'Content-Type: application/json' \
  -d '{"path":"../examples/output/FAMILY PLANNING SERVICE INTEGRATION NATIONAL IMPLEMENTATION GUIDELINE/FAMILY PLANNING SERVICE INTEGRATION NATIONAL IMPLEMENTATION GUIDELINE.alignment.json","review_status":"unreviewed","document_version":"2021-06"}'
```

Unreviewed machine translations stay `review_status=unreviewed` and are **not** treated as clinically verified.

## Models

| Key | Hub ID | Use |
|---|---|---|
| `amharic_base` | `rasyosef/roberta-amharic-text-embedding-base` | Default Amharic dense retriever (passage retrieval, not clinical) |
| `amharic_medium` | `rasyosef/roberta-amharic-text-embedding-medium` | Smaller Amharic embedder |
| `labse` | `sentence-transformers/LaBSE` | Cross-lingual EN↔AM (Amharic listed) |
| `afro_xlmr` | `Davlan/afro-xlmr-base` | Mean-pooled MLM features; **not** an STS model |
| `st_multilingual_mpnet` | `paraphrase-multilingual-mpnet-base-v2` | Multilingual ST; evaluate on Amharic before preferring |
| `hash` | none | Tests only |

AfroXLMR is not treated as a sentence encoder. Intent/NER have **training scripts** (`scripts/train_intent.py`, BIO schema) but no pretend-ready classifier.

Default reranker is **medical feature polarity** (negation / contraindication / eligibility). The optional mMARCO cross-encoder is **not** Amharic-validated.

## Fine-tune embeddings

```bash
pip install -e ".[ml]"
python scripts/train_embeddings.py --train data/training/embedding_pairs.jsonl --out artifacts/am-embed
```

## Evaluate (computes real scores on labeled files; does not invent numbers)

```bash
python scripts/benchmark.py --model hash --kb tests/fixtures/mini_kb.json
```

`data/eval/*` labels are `developer_draft`. Replace with independently reviewed gold and keep paraphrase `cluster` groups out of both train and test.

## Docker

```bash
cd nlu
docker compose up --build
```

Postgres+pgvector is included. The API still runs BM25 in-process. Dense pgvector wiring is in `src/we_care_nlu/kb/postgres.py`.

## Contracts for later components

`POST /nlp/analyze` → `AnalyzeResponse` (intent, entities, retrieval hits with page + evidence_id, clarification, safety). `generative_answer` is always `null`.
