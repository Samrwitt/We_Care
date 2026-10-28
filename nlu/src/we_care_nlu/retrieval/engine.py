"""Hybrid BM25 + dense retrieval with medical polarity reranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from we_care_nlu.retrieval.bm25 import BM25Okapi

from we_care_nlu.contracts.schemas import ChunkRecord, RetrievalHit
from we_care_nlu.embeddings.cache import embed_texts
from we_care_nlu.embeddings.registry import Embedder, cosine
from we_care_nlu.kb.ingest import tokenize
from we_care_nlu.preprocess.normalize import has_negation, preprocess
from we_care_nlu.retrieval.expand import expand_query, method_terms
from we_care_nlu.retrieval.relevance import (
    BF_CUES,
    blob_of,
    is_demo_chunk,
    relevance_boost,
    should_force_include,
)

CONTRA_CUES = ("ተቃራኒ", "የተከለከለ", "አይመከርም", "contraindicat", "must not", "do not use")
ALLOW_CUES = ("ይቻላል", "ይመከራል", "can be used", "is indicated", "may be used")


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

    def rebuild(self, embedder: Embedder | None, *, embed_cache: bool = True) -> None:
        corpus = []
        dense_corpus = []
        self.tokens = []
        for c in self.chunks:
            # Sparse index keeps bilingual + section text.
            blob = f"{c.amharic_text} {c.english_text} {c.section}"
            corpus.append(blob)
            self.tokens.append(tokenize(blob))
            # Dense index prefers native Amharic passage text (fallback English).
            am = (c.amharic_text or "").strip()
            en = (c.english_text or "").strip()
            dense_corpus.append(am if am else en)
        self.bm25 = BM25Okapi(self.tokens) if self.tokens else None
        # Hash is a test stand-in — skip dense encode (not semantic).
        use_dense = embedder is not None and dense_corpus and getattr(embedder, "model_key", "") != "hash"
        if use_dense:
            print(
                f"we-care-nlu: encoding {len(dense_corpus)} chunks with {embedder.model_key} "
                f"(cache={'on' if embed_cache else 'off'})",
                flush=True,
            )
            self.embeddings = embed_texts(embedder, dense_corpus, use_cache=embed_cache)
            self.model_key = embedder.model_key
            print(
                f"we-care-nlu: dense index ready shape={self.embeddings.shape}",
                flush=True,
            )
        else:
            self.embeddings = None
            if embedder is not None:
                self.model_key = embedder.model_key


class HybridRetriever:
    def __init__(self, cfg: dict[str, Any], embedder: Embedder):
        self.cfg = cfg
        self.embedder = embedder
        self.index = KnowledgeIndex()

    def add_chunks(self, chunks: Iterable[ChunkRecord], *, rebuild: bool = True) -> int:
        existing = {c.content_hash for c in self.index.chunks}
        added = 0
        for c in chunks:
            if c.content_hash in existing:
                continue
            self.index.chunks.append(c)
            existing.add(c.content_hash)
            added += 1
        if added and rebuild:
            self.rebuild_index()
        return added

    def rebuild_index(self) -> None:
        self.index.rebuild(
            self.embedder,
            embed_cache=bool((self.cfg or {}).get("embed_cache", True)),
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        intent: str | None = None,
        entities: list[Any] | None = None,
        extra_terms: list[str] | None = None,
        concept_terms: list[str] | None = None,
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
        # Keep the user ask for lexical alignment; expand only for BM25/dense.
        original_query = query
        query = expand_query(query, extra_terms, concept_terms=concept_terms)
        # Broader candidate pool when entities/intent need curated recall.
        if entities or intent in {"family_planning_methods", "eligibility", "missed_dose"}:
            cand = max(cand, 80)
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
        if self.index.embeddings is not None:
            q_texts = [preprocess(original_query).normalized]
            if bool(cfg.get("dual_query_encode", True)):
                expanded_norm = preprocess(query).normalized
                if expanded_norm and expanded_norm != q_texts[0]:
                    q_texts.append(expanded_norm)
            q_vecs = self.embedder.encode(q_texts)
            sims = cosine(q_vecs, self.index.embeddings)
            # Max over original + expanded query vectors.
            best = sims.max(axis=0) if sims.ndim == 2 else sims
            dense_n = ((best + 1.0) / 2.0).astype(np.float32)
        fusion = str(cfg.get("fusion") or "weighted").lower()
        if semantic and fusion == "rrf":
            rrf_k = float(cfg.get("rrf_k") or 60)
            bm25_rank = np.argsort(np.argsort(-bm25_raw))
            dense_rank = np.argsort(np.argsort(-dense_n))
            fused = (sw / (rrf_k + bm25_rank + 1.0)) + (dw / (rrf_k + dense_rank + 1.0))
            fused = fused.astype(np.float32)
        else:
            fused = (dw * dense_n + sw * bm25_n).astype(np.float32)
        # Hybrid pool: union of top BM25 + top dense, then rank by fused.
        # Dense-only top-k buries strong Amharic lexical hits when MiniLM ranks are noisy.
        pool: set[int] = set(int(i) for i in np.argsort(-bm25_raw)[:cand])
        if semantic:
            pool.update(int(i) for i in np.argsort(-dense_n)[:cand])
        order = sorted(pool, key=lambda i: float(-fused[i]))
        seen = set(order)
        # Force-include curated / high-signal chunks BM25 may bury under large KBs.
        # Use original_query for alignment so expand() synonym dumps don't fake distinctive tokens.
        extras: list[tuple[float, int]] = []
        for i, chunk in enumerate(self.index.chunks):
            if i in seen:
                continue
            pri = should_force_include(
                query=original_query, chunk=chunk, intent=intent, entities=entities
            )
            if pri > 0:
                extras.append((pri, i))
        extras.sort(reverse=True)
        for _, i in extras[:20]:
            if i not in seen:
                order.append(i)
                seen.add(i)

        q_pol = _polarity(original_query)
        q_bf = any(c.lower() in (original_query or "").lower() for c in BF_CUES)
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
            blob = blob_of(chunk)
            eid = str(chunk.evidence_id or "")
            demo = is_demo_chunk(chunk)
            score = float(fused[int(idx)])
            score += relevance_boost(
                query=original_query,
                blob=blob,
                intent=intent,
                entities=entities,
                evidence_id=eid,
                demo=demo,
            )
            # Keep a light method-term bump for non-entity lexical hits.
            for term in method_terms(entities or []):
                if term.lower() in blob:
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


# Re-export cues used by pipeline/tests.
__all__ = [
    "HybridRetriever",
    "KnowledgeIndex",
    "_polarity",
    "BF_CUES",
    "LAM_CUES",
    "METHOD_LIST_CUES",
    "ALLOW_CUES",
    "CONTRA_CUES",
]
