# We Care

Non-generative Amharic assistant for reproductive-health guidelines.

**Flow:** translate English PDFs → bilingual knowledge base → retrieve evidence → compose a short Amharic reply grounded in that evidence. No free-form clinical generation. Not a clinically validated medical device.

| Package | Role |
|---|---|
| [`mt/`](mt/README.md) | English→Amharic PDF pipeline (Google + glossary by default) |
| [`nlu/`](nlu/README.md) | Amharic Q&A over the bilingual / curated KB |

## Runbook

```bash
# 1) PDF translation (resumes from mt/checkpoints/)
cd mt
we-care-mt translate data/source_pdfs --model google -o output
# or one file:
we-care-mt translate "data/source_pdfs/FAMILY PLANNING SERVICE INTEGRATION NATIONAL IMPLEMENTATION GUIDELINE.pdf" --model google -o output

# 2) Span translator for chat compose (Google, no local GPU weights)
we-care-mt serve --port 8090 --model google

# 3) Chat API + UI
cd ../nlu
WE_CARE_TRANSLATION=mt WE_CARE_MT_URL=http://127.0.0.1:8090 we-care-nlu serve --port 8088
# open http://127.0.0.1:8088/
```

Amharic blogs + translated guideline DOCX live in [`Docs/`](Docs/). Build the chat KB with:

```bash
cd nlu && PYTHONPATH=src python scripts/build_docs_kb.py
```

That writes `nlu/data/kb/docs_amharic_kb.json` (loaded first on NLU startup). Optional fillers: `fp_guideline_bilingual.json`, `fp_user_kb.json`.

Local chat defaults to the **hash** embedder (no GPU). Optional dense models: set `WE_CARE_NLU_MODEL=labse` (or similar) after installing weights.

```bash
docker compose up --build   # see mt/ and nlu/ compose files
```
