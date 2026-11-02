# Component 2 — Amharic NLU and Medical Knowledge Retrieval

Independent FastAPI service. It does **not** generate chatbot replies and does **not** call GPT/Gemini/Claude/Llama/Qwen or any generative SLM.

```
User message (Amharic)
        │
        ▼
 Preprocess (NFC, Ethiopic punct, light spelling, keep negation)
        │
        ├─► Intent (lexicon + prototypes; optional trained encoder)
        ├─► Entities (gazetteer + regex; optional trained NER)
        ├─► Context resolver (uses dialogue-manager state only)
        └─► Hybrid retrieval
              BM25  +  dense cosine  →  medical-aware rerank
        │
        ▼
 JSON for Component 3 (rule-based dialogue manager)
```

Components 1 / 3 / 4 / 5 talk to this service only through the contracts in `src/we_care_nlu/contracts/`.

## Model selection (verified)

| Role | Checkpoint | Why | Limitation |
|---|---|---|---|
| Default Amharic dense retriever | `rasyosef/roberta-amharic-text-embedding-base` | Sentence-Transformer trained for **Amharic passage retrieval** (ACL 2025 Findings). Mean-pool + cosine. 768-d. | News/web retrieval, not clinical. Must be re-benchmarked on this KB. |
| Lighter Amharic embedder | `rasyosef/roberta-amharic-text-embedding-medium` | Same paper, 42M params, 512-d. | Same domain gap. |
| Cross-lingual EN↔AM | `sentence-transformers/LaBSE` | Explicit Amharic (`am`) among 109 languages. Good when a query is Amharic and a chunk is still English. | Weaker for paraphrase-not-translation similarity. |
| African LM features | `Davlan/afro-xlmr-base` | MLM-adapted XLM-R including Amharic. Used with **mean pooling**; it is **not** a pretrained sentence encoder. | Classification/feature model, not STS-trained. |
| Multilingual ST | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | XLM-R family; Amharic present in XLM-R pretraining. | Not Amharic-specialized; evaluate before preferring it. |
| Optional reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Encoder pairwise scorer. | **Not Amharic-validated.** Default reranker is a medical feature model (negation / contraindication / entity overlap). |
| Tests / CI | `hash` | Deterministic bag-of-character hashes. | Not semantic. Never used as a quality claim. |

AfroXLMR is wired as an encoder + mean-pool adapter, not treated as a sentence-transformer.

No intent or NER checkpoint in this repo is pretrained on the project intent/entity schema. Production intent/NER therefore uses **lexicons + patterns** (working) plus **training scripts** for when labeled data exists. An untrained XLM-R classifier is not loaded as if it were ready.

## Retrieval polarity

Cosine similarity is not allowed to collapse:

- “usable while breastfeeding” vs “contraindicated while breastfeeding”
- presence vs absence of negation

The reranker applies a large penalty when query polarity (negation / contraindication / eligibility) disagrees with the chunk.

## Storage

- Dev/tests: in-memory BM25 + numpy vectors
- Docker: PostgreSQL + pgvector for dense vectors; BM25 remains in-process (rank-bm25)

Unreviewed machine translations are stored with `review_status=unreviewed` and never labeled clinically verified.
