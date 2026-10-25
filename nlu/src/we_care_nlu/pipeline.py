"""Orchestrates preprocess → intent → NER → context → retrieval → safety JSON."""

from __future__ import annotations

from typing import Any

from we_care_nlu.config import load_config
from we_care_nlu.context.resolve import resolve_context
from we_care_nlu.contracts.schemas import AnalyzeRequest, AnalyzeResponse, ContextState, ReplyResponse
from we_care_nlu.embeddings.registry import load_embedder
from we_care_nlu.intent.classifier import IntentClassifier
from we_care_nlu.kb.ingest import ingest_alignment, ingest_path
from we_care_nlu.ner.extract import EntityExtractor
from we_care_nlu.preprocess.normalize import detect_language, preprocess
from we_care_nlu.retrieval.engine import HybridRetriever
from we_care_nlu.retrieval.expand import expand_query, method_terms
from we_care_nlu.safety.rules import emergency_flags, evidence_conflicts, review_warnings
from we_care_nlu.translation import build_translator
from we_care_nlu.translation.mt_client import TranslatorNotReady
from we_care_nlu.translation.phrase_map import covered_question, map_to_english
from we_care_nlu.translation.advice import intent_advice
from we_care_nlu.translation.reply import TEMPLATES, citation, reply_qa, select_span


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
        self.translator = build_translator(self.cfg)

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
        search_q = expand_query(pre.normalized, method_terms(entities))
        hits = self.retriever.search(
            search_q,
            top_k=req.top_k,
            filters=filters,
            intent=intent_out["intent"],
            entities=entities,
        )
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

    def reply(self, req: AnalyzeRequest) -> ReplyResponse:
        original = req.text or ""
        translator_loading = False
        if detect_language(original) == "en":
            english_query = original
        else:
            covered = covered_question(original)
            if covered is not None:
                english_query = covered
            else:
                try:
                    english_query = self.translator.to_english(original)
                except TranslatorNotReady:
                    english_query = map_to_english(original)
                    translator_loading = True
        analysis = self.analyze(
            AnalyzeRequest(
                text=english_query,
                language="en",
                conversation_id=req.conversation_id,
                context=req.context,
                filters=req.filters,
                top_k=req.top_k or 20,
                model_key=req.model_key,
            )
        )
        emergency = emergency_flags(original + " " + english_query)
        if emergency["is_emergency"]:
            analysis.safety["emergency"] = emergency
            analysis.intent = "emergency"
            return self._reply_template("emergency", "template", original, english_query, analysis)

        if analysis.requires_clarification:
            return self._reply_template("clarify", "template", original, english_query, analysis)

        span, hit = select_span(english_query, analysis.retrieval_results, analysis.intent)
        if analysis.safety.get("unsupported_question") or not span:
            advice = intent_advice(analysis.intent)
            if advice:
                amharic, english = advice
                return self._reply_from_analysis(
                    analysis,
                    amharic_reply=amharic,
                    english_query=english_query,
                    english_reply=english,
                    status="template",
                    cited={},
                )
            return self._reply_template("abstain", "abstained", original, english_query, analysis)

        cited = citation(hit)
        if translator_loading:
            return self._reply_from_analysis(
                analysis,
                amharic_reply=TEMPLATES["loading"]["am"],
                english_query=english_query,
                english_reply=span,
                status="source_only",
                cited=cited,
            )
        if self.translator.can_translate_reply:
            try:
                amharic = self.translator.to_amharic(span)
            except TranslatorNotReady:
                return self._reply_from_analysis(
                    analysis,
                    amharic_reply=TEMPLATES["loading"]["am"],
                    english_query=english_query,
                    english_reply=span,
                    status="source_only",
                    cited=cited,
                )
            except Exception as exc:
                failed = self._reply_template("qa_fail", "qa_failed", original, english_query, analysis, cited)
                failed.qa_ok = False
                failed.qa_notes = [f"translation_error:{exc.__class__.__name__}"]
                failed.english_reply = span
                return failed
            notes = reply_qa(span, amharic)
            if notes:
                failed = self._reply_template("qa_fail", "qa_failed", original, english_query, analysis, cited)
                failed.qa_ok = False
                failed.qa_notes = notes
                failed.english_reply = span
                return failed
            return self._reply_from_analysis(
                analysis,
                amharic_reply=amharic,
                english_query=english_query,
                english_reply=span,
                status="translated",
                cited=cited,
            )

        notice = TEMPLATES["source_only"]["am"]
        return self._reply_from_analysis(
            analysis,
            amharic_reply=notice,
            english_query=english_query,
            english_reply=span,
            status="source_only",
            cited=cited,
        )

    def _reply_template(
        self,
        key: str,
        status: str,
        original: str,
        english_query: str,
        analysis: AnalyzeResponse,
        cited: dict | None = None,
    ) -> ReplyResponse:
        del original
        template = TEMPLATES[key]
        return self._reply_from_analysis(
            analysis,
            amharic_reply=template["am"],
            english_query=english_query,
            english_reply=template["en"],
            status=status,
            cited=cited or {},
        )

    def _reply_from_analysis(
        self,
        analysis: AnalyzeResponse,
        *,
        amharic_reply: str,
        english_query: str,
        english_reply: str,
        status: str,
        cited: dict,
    ) -> ReplyResponse:
        return ReplyResponse(
            amharic_reply=amharic_reply,
            english_query=english_query,
            english_reply=english_reply,
            translation_backend=getattr(self.translator, "backend", "map"),
            translation_status=status,  # type: ignore[arg-type]
            evidence_id=cited.get("evidence_id"),
            document_id=cited.get("document_id") or "",
            document_title=cited.get("document_title") or "",
            page=cited.get("page"),
            section=cited.get("section") or "",
            review_status=cited.get("review_status") or "",
            intent=analysis.intent,
            intent_labels=analysis.intent_labels,
            entities=analysis.entities,
            requires_clarification=analysis.requires_clarification,
            clarification_question_key=analysis.clarification_question_key,
            confidence_status=analysis.confidence_status,
            safety=analysis.safety,
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
