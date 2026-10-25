English source PDFs for the knowledge base. The NLU service reads these files and does not translate them in advance.

```bash
cd nlu
PYTHONPATH=src python scripts/ingest_guidelines.py --pdf-dir ../data/guidelines
```
