"""NLUEngine: analyze a question, then build a reply.

Flow for newcomers:
  1. analyze() — preprocess, intent, entities, context, retrieve chunks
  2. reply()   — implemented in reply_pipeline.ReplyMixin

See docs/BACKEND.md for the full package map.
"""

from __future__ import annotations

from typing import Any

from we_care_nlu.config import load_config
from we_care_nlu.context.resolve import resolve_context
from we_care_nlu.contracts.schemas import AnalyzeRequest, AnalyzeResponse, ContextState, Entity
from we_care_nlu.embeddings.registry import load_embedder
from we_care_nlu.intent.classifier import IntentClassifier
from we_care_nlu.kb.ingest import ingest_alignment, ingest_path
from we_care_nlu.ner.extract import EntityExtractor
from we_care_nlu.preprocess.normalize import preprocess
from we_care_nlu.retrieval.engine import HybridRetriever
from we_care_nlu.retrieval.expand import INTENT_NEEDLES, TOPIC_SURFACES, method_terms
from we_care_nlu.retrieval.relevance import (
    CLINICAL_CONDITION_INTENTS,
    FP_SCOPED_INTENTS,
    _distinctive_aliases,
    distinctive_query_tokens,
    soft_topic_groups,
)
from we_care_nlu.safety.rules import emergency_flags, evidence_conflicts, review_warnings
from we_care_nlu.terminology.index import ConceptIndex
from we_care_nlu.translation import build_translator

from we_care_nlu.reply_pipeline import ReplyMixin


class NLUEngine(ReplyMixin):
    def __init__(self, cfg: dict[str, Any] | None = None, model_key: str | None = None):
        self.cfg = cfg or load_config()
        models = self.cfg.get("embeddings", {}).get("models", {})
        key = model_key or self.cfg.get("embeddings", {}).get("test_model") or "hash"
        spec = models.get(key) or {"kind": "hash", "dim": 256}
        # If a neural model is requested but unavailable, fall back to hash and record it.
        try:
            self.embedder = load_embedder(key, spec)
            self.embedder_fallback = False
        except Exception as load_exc:
            print(
                f"we-care-nlu: embedder={key} failed ({load_exc.__class__.__name__}: {load_exc}); "
                "falling back to hash (BM25-only — not semantic)",
                flush=True,
            )
            self.embedder = load_embedder("hash", models.get("hash") or {"kind": "hash"})
            self.embedder_fallback = True
        self.retriever = HybridRetriever(self.cfg.get("retrieval") or {}, self.embedder)
        self.intents = IntentClassifier()
        self.ner = EntityExtractor()
        self.concepts = ConceptIndex().load()
        if not self.concepts.concepts:
            self.concepts.ensure_seeded_from_gazetteer()
        self.model_key = self.embedder.model_key
        self.translator = build_translator(self.cfg)

    def ingest_alignment_dict(self, payload: dict[str, Any], *, rebuild: bool = True, **kwargs) -> int:
        chunks = ingest_alignment(payload, **kwargs)
        n = self.retriever.add_chunks(chunks, rebuild=rebuild)
        self._refresh_terminology_candidates()
        return n

    def ingest_file(self, path: str, *, rebuild: bool = True, **kwargs) -> int:
        n = self.retriever.add_chunks(ingest_path(path, **kwargs), rebuild=rebuild)
        self._refresh_terminology_candidates()
        return n

    def rebuild_index(self) -> None:
        self.retriever.rebuild_index()
        self._refresh_terminology_candidates()

    def _refresh_terminology_candidates(self) -> None:
        """Extract pending candidates from the live index; never auto-approve."""
        try:
            chunks = self.retriever.index.chunks
            if chunks:
                self.concepts.rebuild_candidates_from_chunks(chunks)
        except Exception as exc:  # noqa: BLE001
            print(f"we-care-nlu: terminology extract skipped ({exc.__class__.__name__}: {exc})", flush=True)

    def analyze(self, req: AnalyzeRequest, *, retrieval_text: str | None = None) -> AnalyzeResponse:
        # Intent/NER always use the ask as written. Retrieval may use an expanded string
        # (e.g. English ask + Amharic BM25 boost) without flipping the classifier.
        pre = preprocess(req.text)
        intent_out = self.intents.predict(pre.normalized)
        entities = self.ner.extract(pre.normalized)
        pre_retr = preprocess(retrieval_text) if (retrieval_text and retrieval_text != req.text) else pre

        # Layer B: KB/gazetteer concepts (approved+seeded only).
        matched_concepts = self.concepts.match(pre.normalized, limit=6)
        concept_entities: list[Entity] = []
        for c in matched_concepts:
            surface = c.canonical_amharic or c.canonical_english or c.concept_id
            concept_entities.append(
                Entity(
                    type="medical_concept",
                    value=c.concept_id,
                    surface=surface,
                    start=-1,
                    end=-1,
                    confidence=0.75,
                    source="concept_index",
                )
            )
        # Prefer gazetteer spans; append concept hits that are not already covered.
        existing_vals = {e.value for e in entities}
        for ce in concept_entities:
            if ce.value not in existing_vals:
                entities.append(ce)

        ctx_info = {}
        entities, ctx_info = resolve_context(pre.normalized, entities, req.context, intent_out["intent"])
        inherited = ctx_info.get("inherited_intent")
        if inherited and intent_out["intent"] in {
            "general_medical_info",
            "definition",
            "family_planning_methods",
            "follow_up",
        }:
            intent_out["intent"] = str(inherited)
            labels = list(intent_out.get("intent_labels") or [])
            if inherited not in labels:
                labels.insert(0, str(inherited))
            intent_out["intent_labels"] = labels

        # Topic: lexicon topics first, else primary concept, else prior context topic on follow-ups.
        topic = intent_out.get("topic")
        topics = list(intent_out.get("topics") or [])
        if not topic and matched_concepts:
            primary_c = matched_concepts[0]
            topic = primary_c.canonical_english or primary_c.canonical_amharic or primary_c.concept_id
            topics = [
                (c.canonical_english or c.canonical_amharic or c.concept_id) for c in matched_concepts
            ]
        if not topic and ctx_info.get("inherited_topic"):
            topic = str(ctx_info["inherited_topic"])
        if (
            not topic
            and req.context
            and req.context.last_topic
            and ctx_info.get("resolved_from_previous_turn")
        ):
            topic = req.context.last_topic
            ctx_info["inherited_topic"] = topic
        # Keep sibling menses labels so retrieval expands both irregular + period-pain surfaces.
        if str(topic or "").lower() in {"irregular_menses", "menstruation", "endometriosis"}:
            for label in ("menstruation", "irregular_menses"):
                if label not in topics:
                    topics.append(label)

        emergency = emergency_flags(pre.original + " " + pre.normalized)
        if emergency["is_emergency"]:
            intent_out["intent"] = "emergency"
            if "emergency" not in intent_out["intent_labels"]:
                intent_out["intent_labels"].insert(0, "emergency")

        filters = dict(req.filters or {})
        intent_label = str(intent_out["intent"] or "")
        extra_terms = list(method_terms(entities))
        concept_terms = self.concepts.expansion_terms(pre.normalized, limit=6)
        if topic and topic not in concept_terms:
            concept_terms = list(concept_terms) + [str(topic)]
        topic_key = str(topic or "").strip().lower()
        for surf in TOPIC_SURFACES.get(topic_key, ()):
            if surf and surf not in concept_terms:
                concept_terms.append(surf)
        # Period-pain follow-ups after መዛባት: pull both irregular + menstruation surfaces.
        if topic_key in {"irregular_menses", "menstruation", "endometriosis"}:
            for key in ("menstruation", "irregular_menses"):
                for surf in TOPIC_SURFACES.get(key, ()):
                    if surf and surf not in concept_terms:
                        concept_terms.append(surf)
        conditions = [
            str(getattr(e, "value", "") or "")
            for e in entities
            if getattr(e, "type", "") == "condition"
        ]
        for cond in conditions:
            for surf in TOPIC_SURFACES.get(cond, ()):
                if surf and surf not in concept_terms:
                    concept_terms.append(surf)
        # HIV: FP cues only for eligibility/methods; clinical asks need disease + intent surfaces.
        if "hiv" in conditions or topic_key == "hiv":
            extra_terms.extend(["ኤች.አይ.ቪ", "ኤችአይቪ", "HIV", "ኤድስ"])
            if intent_label in FP_SCOPED_INTENTS:
                extra_terms.extend(["የእርግዝና መከላከያ", "family planning", "contraceptive"])
            elif intent_label in CLINICAL_CONDITION_INTENTS:
                extra_terms.extend(list(INTENT_NEEDLES.get(intent_label) or ()))
        elif intent_label in CLINICAL_CONDITION_INTENTS:
            extra_terms.extend(list(INTENT_NEEDLES.get(intent_label) or ()))
        # Ground distinctive ask facets + soft domain into BM25 (general, not per-article rules).
        for tok in distinctive_query_tokens(pre.normalized):
            extra_terms.extend(_distinctive_aliases(tok)[:8])
        for group in soft_topic_groups(pre.normalized):
            extra_terms.extend(list(group[:6]))
        hits = self.retriever.search(
            pre_retr.normalized,
            top_k=req.top_k,
            filters=filters,
            intent=intent_label,
            entities=entities,
            extra_terms=extra_terms,
            concept_terms=concept_terms,
        )
        conflicts = evidence_conflicts(hits, pre_retr.normalized)
        review = review_warnings(hits)
        abstain_thr = float((self.cfg.get("retrieval") or {}).get("abstain_below") or 0.22)
        top_score = hits[0].fused_score if hits else 0.0
        unsupported = not hits or top_score < abstain_thr
        if hits and self.embedder.model_key != "hash" and (hits[0].dense_score or 0) < 0.70:
            # dense_score is (cosine + 1) / 2. Below 0.70 the chunk is not about the question.
            unsupported = True
        requires_clarification = bool(ctx_info.get("needs_clarification"))
        if unsupported and not emergency["is_emergency"]:
            conf_status = "abstain"
        elif self.embedder.model_key == "hash":
            conf_status = "requires_calibration"
        else:
            conf_status = intent_out.get("confidence_status") or "requires_calibration"
        if self.embedder_fallback:
            conf_status = "requires_calibration"

        interpretation = {
            "intent": intent_out["intent"],
            "topic": topic,
            "topics": topics,
            "requires_context": bool(ctx_info.get("resolved_from_previous_turn")),
            "context_resolved": bool(ctx_info.get("resolved_from_previous_turn")),
            "query_expansion_terms": concept_terms[:16],
            "concept_ids": [c.concept_id for c in matched_concepts],
        }
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
            "terminology": self.concepts.stats(),
        }
        return AnalyzeResponse(
            language=pre.language,
            normalized_text=pre.normalized,
            intent=intent_out["intent"],
            intent_labels=intent_out["intent_labels"],
            intent_confidence=intent_out["intent_confidence"],
            intent_backend=intent_out["intent_backend"],
            topic=topic,
            topics=topics,
            medical_concepts=[c.to_public() for c in matched_concepts],
            interpretation=interpretation,
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
