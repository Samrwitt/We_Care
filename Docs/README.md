# Docs — Amharic source materials

| Folder | Contents |
|---|---|
| `blogs/` | Amharic health articles (~90 DOCX) |
| `guidelines_am/` | Amharic translations of RH guideline DOCX files |

Build / refresh the NLU knowledge base:

```bash
cd nlu && PYTHONPATH=src python scripts/build_docs_kb.py
# writes nlu/data/kb/docs_amharic_kb.json — loaded automatically on we-care-nlu serve
```
