# We Care

Non-generative Amharic medical assistant for reproductive-health guidelines. It retrieves an English sentence from the source documents and can translate that sentence into Amharic. It does not write a free-form clinical answer.

Two packages sit side by side:

| Directory | Role | Run |
|---|---|---|
| [`nlu/`](nlu/README.md) | Understand an Amharic question, search the English guidelines, return one reply | `cd nlu && we-care-nlu serve` |
| [`mt/`](mt/README.md) | Translate a whole English PDF into Amharic for review | `cd mt && we-care-mt --help` |

The knowledge base is the original English text. Source PDFs live in [`data/guidelines/`](data/guidelines). They are not translated ahead of time. `nlu` reads them, and `mt` is the separate document translator for when a full Amharic PDF is needed.

```bash
docker compose up nlu
docker compose run --rm we-care-mt --help
```

Build the English index from the guidelines:

```bash
cd nlu
PYTHONPATH=src python scripts/ingest_guidelines.py --pdf-dir ../data/guidelines
```

`POST /nlp/reply` is the user-facing route. Set `WE_CARE_TRANSLATION=nllb` to translate the retrieved sentence into Amharic. Without that model, the reply shows the English guideline sentence and the page it came from.

This is not a clinically validated medical device. A passing check does not make a translation safe for patients.
