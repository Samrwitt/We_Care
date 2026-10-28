# We Care NLU — Amharic understanding and retrieval

**Component 2.** Amharic (or English) question → retrieve guideline evidence → compose a short Amharic reply. No GPT-style free-form generation. `generative_answer` is always `null`.

Architecture notes: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## Document-first KB

Startup loads (when present), in order:

1. `data/kb/docs_amharic_kb.json` — extensive Amharic blogs + guideline DOCX from repo `Docs/`
2. `data/kb/fp_guideline_bilingual.json` — Phase-1 PDF alignment (optional)
3. `data/kb/fp_user_kb.json` — short curated Amharic answers
4. `data/kb/wecare_articles_am.json` — scraped articles (if present)

Rebuild the Docs KB after adding files under `Docs/blogs` or `Docs/guidelines_am`:

```bash
cd nlu && PYTHONPATH=src python scripts/build_docs_kb.py
```

Replies are **composed** from the retrieved span (cleaned + rendered via `we-care-mt` on `:8090` when available), not free-form generation. Unreviewed alignments stay `review_status=unreviewed`.

## Install

```bash
cd nlu
pip install -e ".[dev]"
pytest
```

## Run

```bash
# Terminal A — span MT (Google)
cd ../mt && we-care-mt serve --port 8090 --model google

# Terminal B — chat
cd ../nlu
WE_CARE_TRANSLATION=mt WE_CARE_MT_URL=http://127.0.0.1:8090 we-care-nlu serve --port 8088
```

UI: http://127.0.0.1:8088/

```bash
curl -X POST localhost:8088/nlp/reply \
  -H 'Content-Type: application/json' \
  -d '{"text":"ኮንዶም ምንድን ነው?"}'

curl -X POST localhost:8088/documents/ingest \
  -H 'Content-Type: application/json' \
  -d '{"path":"data/kb/fp_guideline_bilingual.json","review_status":"unreviewed","document_version":"2021-06"}'
```

## Embeddings

Serve defaults to **hybrid BM25 + dense** via `st_multilingual_minilm` when `sentence-transformers` is installed (`pip install 'we-care-nlu[ml]'`). Tests/CI stay on **`hash`** (BM25 only).

```bash
# optional override
WE_CARE_NLU_MODEL=st_multilingual_minilm we-care-nlu serve --port 8088
# stronger Amharic (downloads weights):
WE_CARE_NLU_MODEL=amharic_medium we-care-nlu serve --port 8088
```

| Key | Use |
|---|---|
| `st_multilingual_minilm` | Default serve dense model (fast CPU, multilingual paraphrase) |
| `hash` | Tests / CI (no semantic similarity) |
| `labse` | Cross-lingual dense retrieval |
| `amharic_base` / `amharic_medium` | Amharic passage embedders |

Chunk vectors are cached under `data/cache/embeddings/` so restarts skip re-encoding. Fusion uses reciprocal rank fusion (RRF) of BM25 + dense, plus medical feature rerank.

Intent/NER are lexicon/gazetteer prototypes unless you train with `scripts/train_*.py`.
