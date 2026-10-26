# We Care

Non-generative Amharic medical assistant for reproductive-health guidelines. It retrieves an English sentence from the source documents and can translate that sentence into Amharic. It does not write a free-form clinical answer.

Two packages sit side by side:

| Directory | Role | Run |
|---|---|---|
| [`nlu/`](nlu/README.md) | Understand an Amharic question, search the English guidelines, return one reply | `cd nlu && we-care-nlu serve` |
| [`mt/`](mt/README.md) | Translate a whole English PDF into Amharic for review | `cd mt && we-care-mt --help` |

Default chat KB is Amharic WeCare articles (`nlu/data/kb/wecare_articles_am.json`). `mt` translates one retrieved span for English asks (and remains available for full PDF jobs).

```bash
# Chat stack (slim MT span image + NLU/LaBSE)
docker compose up --build

# Full PDF MT image (PyMuPDF/fonts) when you need document translation:
docker compose build --build-arg WE_CARE_MT_PROFILE=full we-care-mt
```

This is not a clinically validated medical device. A passing check does not make a translation safe for patients.
