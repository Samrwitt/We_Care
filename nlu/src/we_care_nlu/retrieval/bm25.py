"""In-process BM25 (Okapi). Avoids extra native dependencies."""

from __future__ import annotations

import math
from collections import Counter


class BM25Okapi:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.n_docs = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / max(self.n_docs, 1)
        df: Counter[str] = Counter()
        for doc in corpus:
            df.update(set(doc))
        self.idf = {
            t: math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1.0) for t, n in df.items()
        }
        self.doc_tf = [Counter(doc) for doc in corpus]
        inverted: dict[str, list[int]] = {}
        for i, tf_map in enumerate(self.doc_tf):
            for term in tf_map:
                inverted.setdefault(term, []).append(i)
        self.inverted = inverted

    def get_scores(self, query: list[str]) -> list[float]:
        scores = [0.0] * self.n_docs
        if not self.n_docs:
            return scores
        for q in query:
            idf = self.idf.get(q)
            if idf is None:
                continue
            for i in self.inverted.get(q, ()):
                tf = self.doc_tf[i][q]
                dl = self.doc_len[i]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[i] += idf * (tf * (self.k1 + 1) / denom)
        return scores
