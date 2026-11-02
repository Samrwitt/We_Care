"""Hybrid BM25 + dense retrieval, expansion, and relevance scoring.

Start with engine.py (HybridRetriever). Scoring helpers are re-exported from
relevance.py (implemented in query_facets / chunk_signals / scoring).
"""

from we_care_nlu.retrieval.engine import HybridRetriever

__all__ = ["HybridRetriever"]
