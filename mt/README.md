# We Care MT — English→Amharic PDF translation

**Component 1.** Extract → segment → translate → glossary/QA → alignment + review PDFs.

**Default backend: Google Translate** (web endpoint for offline quality work) plus the medical glossary. Optional local NLLB / DSFSI models remain available when disk/GPU allow. Not clinically validated.

PDFs live in repo-root [`data/source_pdfs/`](../data/source_pdfs/) (symlinked as `data/source_pdfs` from this package).

## Install

```bash
cd mt
pip install -e ".[dev]"
pytest
```

## Translate

```bash
# Preferred (no local neural weights)
we-care-mt translate data/source_pdfs --model google -o output

# One PDF
we-care-mt translate \
  "data/source_pdfs/FAMILY PLANNING SERVICE INTEGRATION NATIONAL IMPLEMENTATION GUIDELINE.pdf" \
  --model google -o output

# Tests / CI
we-care-mt translate data/source_pdfs/Access\ to\ Contraception.pdf --model mock --backend mock -o output
```

Interrupted jobs resume from `checkpoints/<stem>.google.jsonl`. Segment cache: `cache/translations.sqlite`.

## Span server (for NLU compose)

```bash
we-care-mt serve --port 8090 --model google
```

NLU calls this to turn a short grounded English span into Amharic for the chat reply.

## Optional local models

```bash
we-care-mt download-models --model dsfsi_600m
we-care-mt translate data/source_pdfs --model dsfsi_600m --device auto -o output
```

Weights go under `.hf_cache/`. Do not treat neural or Google output as patient-safe without human review.
