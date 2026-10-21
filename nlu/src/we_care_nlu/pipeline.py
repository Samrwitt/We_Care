"""Orchestrates preprocess → intent → NER → context → retrieval → safety JSON."""

from __future__ import annotations

from typing import Any

from we_care_nlu.config import load_config
from we_care_nlu.context.resolve import resolve_context
from we_care_nlu.contracts.schemas import AnalyzeRequest, AnalyzeResponse, ContextState
from we_care_nlu.embeddings.registry import load_embedder
from we_care_nlu.intent.classifier import IntentClassifier
from we_care_nlu.kb.ingest import ingest_alignment, ingest_path
from we_care_nlu.ner.extract import EntityExtractor
from we_care_nlu.preprocess.normalize import preprocess
from we_care_nlu.retrieval.engine import HybridRetriever
from we_care_nlu.safety.rules import emergency_flags, evidence_conflicts, review_warnings


class NLUEngine:
    def __init__(self, cfg: dict[str, Any] | None = None, model_key: str | None = None):
        self.cfg = cfg or load_config()
        models = self.cfg.get("embeddings", {}).get("models", {})
        key = model_key or self.cfg.get("embeddings", {}).get("test_model") or "hash"
        spec = models.get(key) or {"kind": "hash", "dim": 256}
        # If a neural model is requested but unavailable, fall back to hash and record it.
        try:
            self.embedder = load_embedder(key, spec)
            self.embedder_fallback = False
        except Exception:
            self.embedder = load_embedder("hash", models.get("hash") or {"kind": "hash"})
            self.embedder_fallback = True
        self.retriever = HybridRetriever(self.cfg.get("retrieval") or {}, self.embedder)
        self.intents = IntentClassifier()
        self.ner = EntityExtractor()
        self.model_key = self.embedder.model_key

    def ingest_alignment_dict(self, payload: dict[str, Any], **kwargs) -> int:
        chunks = ingest_alignment(payload, **kwargs)
        return self.retriever.add_chunks(chunks)

    def ingest_file(self, path: str, **kwargs) -> int:
        return self.retriever.add_chunks(ingest_path(path, **kwargs))

    def analyze(self, req: AnalyzeRequest) -> AnalyzeResponse:
        pre = preprocess(req.text)
        intent_out = self.intents.predict(pre.normalized)
        entities = self.ner.extract(pre.normalized)
        ctx_info = {}
        entities, ctx_info = resolve_context(pre.normalized, entities, req.context, intent_out["intent"])
        emergency = emergency_flags(pre.original + " " + pre.normalized)
        if emergency["is_emergency"]:
            intent_out["intent"] = "emergency"
            if "emergency" not in intent_out["intent_labels"]:
                intent_out["intent_labels"].insert(0, "emergency")

        filters = dict(req.filters or {})
        hits = self.retriever.search(pre.normalized, top_k=req.top_k, filters=filters)
        conflicts = evidence_conflicts(hits, pre.normalized)
        review = review_warnings(hits)
        abstain_thr = float((self.cfg.get("retrieval") or {}).get("abstain_below") or 0.22)
        top_score = hits[0].fused_score if hits else 0.0
        unsupported = not hits or top_score < abstain_thr
        requires_clarification = bool(ctx_info.get("needs_clarification"))
        if unsupported and not emergency["is_emergency"]:
            conf_status = "abstain"
        elif self.embedder.model_key == "hash":
            conf_status = "requires_calibration"
        else:
            conf_status = intent_out.get("confidence_status") or "requires_calibration"
        if self.embedder_fallback:
            conf_status = "requires_calibration"

        safety = {
            "emergency": emergency,
            "negation_detected": pre.has_negation,
            "unsupported_question": unsupported,
            "evidence_conflict": conflicts,
            "review_notes": review,
            "diagnoses_patient": False,
            "guideline_currency": "Retrieved passages keep document_version; recency is not inferred.",
            "embedder": {
                "key": self.embedder.model_key,
                "hf_id": getattr(self.embedder, "hf_id", None),
                "amharic_sentence_encoder": getattr(self.embedder, "amharic_sentence_encoder", False),
                "fallback_to_hash": self.embedder_fallback,
            },
        }
        return AnalyzeResponse(
            language=pre.language,
            normalized_text=pre.normalized,
            intent=intent_out["intent"],
            intent_labels=intent_out["intent_labels"],
            intent_confidence=intent_out["intent_confidence"],
            intent_backend=intent_out["intent_backend"],
            entities=entities,
            context_resolution=ctx_info,
            retrieval_results=hits,
            requires_clarification=requires_clarification,
            clarification_question_key="which_method" if requires_clarification else None,
            confidence_status=conf_status,
            safety=safety,
        )

    def resolve_only(self, text: str, context: ContextState | None) -> dict:
        pre = preprocess(text)
        intent_out = self.intents.predict(pre.normalized)
        entities = self.ner.extract(pre.normalized)
        entities, ctx_info = resolve_context(pre.normalized, entities, context, intent_out["intent"])
        return {
            "normalized_text": pre.normalized,
            "intent": intent_out["intent"],
            "entities": [e.model_dump() for e in entities],
            "context_resolution": ctx_info,
        }
