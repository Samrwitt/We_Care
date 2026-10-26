"""Hybrid BM25 + dense retrieval with medical polarity reranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from we_care_nlu.retrieval.bm25 import BM25Okapi

from we_care_nlu.contracts.schemas import ChunkRecord, RetrievalHit
from we_care_nlu.embeddings.registry import Embedder, cosine
from we_care_nlu.kb.ingest import tokenize
from we_care_nlu.preprocess.normalize import has_negation, preprocess
from we_care_nlu.retrieval.expand import INTENT_NEEDLES, expand_query, method_terms

CONTRA_CUES = ("ተቃራኒ", "የተከለከለ", "አይመከርም", "contraindicat", "must not", "do not use")
ALLOW_CUES = ("ይቻላል", "ይመከራል", "can be used", "is indicated", "may be used")
BF_CUES = ("ጡት ማጥባት", "breastfeed", "lactat")


def _polarity(text: str) -> str:
    low = text.lower()
    contra = any(c.lower() in low for c in CONTRA_CUES) or has_negation(text)
    allow = any(c.lower() in low for c in ALLOW_CUES)
    if contra and not allow:
        return "contra"
    if allow and not contra:
        return "allow"
    if contra and allow:
        return "mixed"
    return "neutral"


@dataclass
class KnowledgeIndex:
    chunks: list[ChunkRecord] = field(default_factory=list)
    embeddings: np.ndarray | None = None
    bm25: BM25Okapi | None = None
    tokens: list[list[str]] = field(default_factory=list)
    model_key: str = "hash"

    def rebuild(self, embedder: Embedder | None) -> None:
        corpus = []
        self.tokens = []
        for c in self.chunks:
            blob = f"{c.amharic_text} {c.english_text} {c.section}"
            corpus.append(blob)
            self.tokens.append(tokenize(blob))
        self.bm25 = BM25Okapi(self.tokens) if self.tokens else None
        # The hash embedder is a test stand-in. Scoring it over every sentence
        # only slows search, so hash mode stays on BM25.
        use_dense = embedder is not None and corpus and getattr(embedder, "model_key", "") != "hash"
        if use_dense:
            self.embeddings = embedder.encode(corpus)
            self.model_key = embedder.model_key
        else:
            self.embeddings = None
            if embedder is not None:
                self.model_key = embedder.model_key


class HybridRetriever:
    def __init__(self, cfg: dict[str, Any], embedder: Embedder):
        self.cfg = cfg
        self.embedder = embedder
        self.index = KnowledgeIndex()

    def add_chunks(self, chunks: Iterable[ChunkRecord]) -> int:
        existing = {c.content_hash for c in self.index.chunks}
        added = 0
        for c in chunks:
            if c.content_hash in existing:
                continue
            self.index.chunks.append(c)
            existing.add(c.content_hash)
            added += 1
        if added:
            self.index.rebuild(self.embedder)
        return added

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        intent: str | None = None,
        entities: list[Any] | None = None,
    ) -> list[RetrievalHit]:
        if not self.index.chunks or self.index.bm25 is None:
            return []
        cfg = self.cfg
        k = int(top_k or cfg.get("top_k") or 8)
        cand = int(cfg.get("candidate_k") or 40)
        semantic = getattr(self.embedder, "model_key", "") != "hash" and self.index.embeddings is not None
        dw = float(cfg.get("dense_weight") or 0.55)
        sw = float(cfg.get("sparse_weight") or 0.45)
        if not semantic:
            dw, sw = 0.0, 1.0
        else:
            dw, sw = 0.75, 0.25
        query = expand_query(query)
        q_tokens = tokenize(query)
        bm25_raw = np.array(self.index.bm25.get_scores(q_tokens), dtype=np.float32)
        if semantic and bm25_raw.max() > 0:
            # Do not stretch a weak keyword hit up to 1.0 when a meaning score exists.
            scale = float(np.percentile(bm25_raw, 90) or bm25_raw.max())
            bm25_n = np.clip(bm25_raw / (scale + 1e-9), 0.0, 1.0)
        elif bm25_raw.max() > 0:
            bm25_n = bm25_raw / (bm25_raw.max() + 1e-9)
        else:
            bm25_n = bm25_raw
        dense_n = np.zeros(len(self.index.chunks), dtype=np.float32)
        q_vec = None
        if self.index.embeddings is not None:
            q_vec = self.embedder.encode([preprocess(query).normalized])[0]
            sims = cosine(q_vec, self.index.embeddings)[0]
            dense_n = ((sims + 1.0) / 2.0).astype(np.float32)
        fused = dw * dense_n + sw * bm25_n
        order = np.argsort(-fused)[:cand]
        q_pol = _polarity(query)
        q_bf = any(c.lower() in query.lower() for c in BF_CUES)
        hits: list[RetrievalHit] = []
        for idx in order:
            chunk = self.index.chunks[int(idx)]
            if filters:
                if filters.get("document_id") and chunk.document_id != filters["document_id"]:
                    continue
                if filters.get("review_status") and chunk.review_status != filters["review_status"]:
                    continue
            c_pol = _polarity(f"{chunk.amharic_text} {chunk.english_text}")
            polarity_conflict = q_pol in {"contra", "allow"} and c_pol in {"contra", "allow"} and q_pol != c_pol
            c_bf = any(c.lower() in (chunk.amharic_text + chunk.english_text).lower() for c in BF_CUES)
            score = float(fused[int(idx)])
            blob = f"{chunk.amharic_text} {chunk.english_text}".lower()
            if intent:
                if any(n.lower() in blob for n in INTENT_NEEDLES.get(intent) or ()):
                    score += 0.45 if intent == "missed_dose" else 0.12
            for term in method_terms(entities or []):
                if term.lower() in blob:
                    score += 0.18
                    break
            if polarity_conflict:
                score -= float(cfg.get("polarity_penalty") or 0.55)
            if q_bf and c_bf and polarity_conflict:
                score -= 0.25
            if chunk.review_status == "unreviewed":
                score -= float(cfg.get("unreviewed_penalty") or 0.08)
            hits.append(
                RetrievalHit(
                    evidence_id=chunk.evidence_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    section=chunk.section,
                    page=chunk.page,
                    amharic_text=chunk.amharic_text,
                    english_text=chunk.english_text,
                    review_status=chunk.review_status,
                    medical_subject=chunk.medical_subject,
                    document_version=chunk.document_version,
                    dense_score=float(dense_n[int(idx)]),
                    bm25_score=float(bm25_n[int(idx)]),
                    fused_score=max(0.0, score),
                    rerank_score=max(0.0, score),
                    polarity_conflict=polarity_conflict,
                    source_kind=getattr(chunk, "source_kind", "") or "document_evidence",
                    source_url=getattr(chunk, "source_url", "") or "",
                )
            )
        hits.sort(key=lambda h: h.fused_score, reverse=True)
        min_s = float(cfg.get("min_fused_score") or 0.0)
        hits = [h for h in hits if h.fused_score >= min_s]
        return hits[:k]
