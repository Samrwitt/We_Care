"""Orchestrates preprocess → intent → NER → context → retrieval → safety JSON."""

from __future__ import annotations

import os
import re
from typing import Any

from we_care_nlu.config import load_config
from we_care_nlu.context.resolve import resolve_context
from we_care_nlu.contracts.schemas import AnalyzeRequest, AnalyzeResponse, ContextState, Entity, ReplyResponse
from we_care_nlu.embeddings.registry import load_embedder
from we_care_nlu.intent.classifier import IntentClassifier
from we_care_nlu.kb.ingest import ingest_alignment, ingest_path
from we_care_nlu.ner.extract import EntityExtractor
from we_care_nlu.preprocess.normalize import detect_language, preprocess
from we_care_nlu.retrieval.engine import HybridRetriever
from we_care_nlu.retrieval.expand import INTENT_NEEDLES, TOPIC_SURFACES, expand_query, method_terms
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
from we_care_nlu.translation.mt_client import TranslatorNotReady
from we_care_nlu.translation.phrase_map import covered_question, map_to_english
from we_care_nlu.translation.advice import intent_advice
from we_care_nlu.translation.compose import compose_amharic_reply, shape_english_answer
from we_care_nlu.translation.reply import (
    TEMPLATES,
    citation,
    reply_qa,
    focus_span_for_query,
    select_span,
    with_article_cite,
)

_ETHIOPIC = re.compile(r"[\u1200-\u137F]")


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

    def analyze(self, req: AnalyzeRequest) -> AnalyzeResponse:
        pre = preprocess(req.text)
        intent_out = self.intents.predict(pre.normalized)
        entities = self.ner.extract(pre.normalized)

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
        if (
            not topic
            and req.context
            and req.context.last_topic
            and ctx_info.get("resolved_from_previous_turn")
        ):
            topic = req.context.last_topic
            ctx_info["inherited_topic"] = topic

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
            pre.normalized,
            top_k=req.top_k,
            filters=filters,
            intent=intent_label,
            entities=entities,
            extra_terms=extra_terms,
            concept_terms=concept_terms,
        )
        conflicts = evidence_conflicts(hits, pre.normalized)
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

    def reply(self, req: AnalyzeRequest) -> ReplyResponse:
        original = req.text or ""
        translator_loading = False
        ask_amharic = detect_language(original) != "en"
        if ask_amharic:
            # Native Amharic KB: search as written. Phrase-map only labels english_query (no Am→En MT).
            covered = covered_question(original)
            if covered is not None:
                english_query = covered
            else:
                mapped = map_to_english(original)
                english_query = mapped if mapped and not _ETHIOPIC.search(mapped) else original
            search_text = original
            search_lang = "am"
        else:
            english_query = original
            search_text = original
            search_lang = "en"
        analysis = self.analyze(
            AnalyzeRequest(
                text=search_text,
                language=search_lang,
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

        span_query = original if search_lang == "am" else english_query
        span, hit = select_span(span_query, analysis.retrieval_results, analysis.intent)
        # General entity/intent re-rank over retrieval hits (not TOC / wrong-doc dumps).
        from we_care_nlu.retrieval.relevance import answer_quality, relevant_enough
        from we_care_nlu.translation.reply import _content_tokens

        need = _content_tokens(span_query)
        from we_care_nlu.retrieval.relevance import (
            distinctive_overlap,
            distinctive_query_tokens,
            is_demo_chunk,
        )

        dist = distinctive_query_tokens(span_query)
        ranked = []
        for candidate in analysis.retrieval_results:
            if candidate.polarity_conflict:
                continue
            stored = (candidate.amharic_text or "").strip()
            english = (candidate.english_text or "").strip()
            text = stored if (stored and _ETHIOPIC.search(stored)) else english
            if not text or len(text) < 25:
                continue
            if not relevant_enough(
                text,
                query=span_query,
                intent=analysis.intent,
                entities=analysis.entities,
                need_tokens=need,
            ):
                continue
            quality = answer_quality(
                text,
                intent=analysis.intent,
                query=span_query,
                entities=analysis.entities,
                evidence_id=str(candidate.evidence_id or ""),
                fused=float(candidate.fused_score or 0),
            )
            if quality < 0:
                continue
            ranked.append((quality, float(candidate.fused_score or 0), len(text), candidate, text))
        # Avoid comparing RetrievalHit when scores tie.
        ranked.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
        # General answer shaping (any topic):
        # - definition → keep definition-shaped spans only (not FAQ titles / treatment dumps)
        # - no method entity → prefer docs that share distinctive terms over curated fillers
        _DEF_SHAPE = ("ነው።", "ነው፡", "is a", "is an", "ማለት", "ይባላል", "የሚታወቅ", "refers to", "means", "የሚከሠተው")
        from we_care_nlu.retrieval.relevance import (
            CLINICAL_CONDITION_INTENTS,
            FP_SCOPED_INTENTS,
            entity_condition_values,
            entity_method_values,
        )

        methods = entity_method_values(analysis.entities)
        conditions = entity_condition_values(analysis.entities)
        if analysis.intent == "definition" and ranked:
            shaped = [
                row
                for row in ranked
                if any(c in row[4] for c in _DEF_SHAPE)
                and not row[4].strip().endswith(("?", "؟"))
            ]
            if shaped:
                ranked = shaped
        # Prefer docs-grounded spans for open distinctive topics (e.g. cryptic pregnancy).
        # Do not drop curated condition answers (HIV eligibility) or clinical condition hits.
        if (
            dist
            and ranked
            and not methods
            and not (
                conditions
                and (
                    analysis.intent in FP_SCOPED_INTENTS
                    or analysis.intent in CLINICAL_CONDITION_INTENTS
                )
            )
        ):
            grounded = [
                row
                for row in ranked
                if distinctive_overlap(row[4], dist) > 0 and not is_demo_chunk(row[3])
            ]
            if grounded:
                ranked = grounded
        if ranked:
            best_q, _, _, best_hit, best_text = ranked[0]
            span_q = (
                answer_quality(
                    span or "",
                    intent=analysis.intent,
                    query=span_query,
                    entities=analysis.entities,
                    evidence_id=str(getattr(hit, "evidence_id", "") or ""),
                    fused=float(getattr(hit, "fused_score", 0) or 0),
                )
                if span
                else -99.0
            )
            if (
                not span
                or not hit
                or len((span or "").strip()) < 50
                or span_q + 0.8 < best_q
                or best_q >= 8.0
            ):
                hit = best_hit
                span = best_text[:900]
        if span:
            span = focus_span_for_query(span, span_query)[:900]
        # Methods-*list* asks with TOC/category teasers → curated overview advice.
        list_ask = any(
            cue in (original or "").lower()
            for cue in (
                "ምንድን ናቸው",
                "አማራጮች",
                "ዘዴዎች ምን",
                "what methods",
                "which methods",
                "method options",
                "contraceptive options",
            )
        )
        if span and hit and analysis.intent == "family_planning_methods" and list_ask:
            from we_care_nlu.translation.reply import method_cue_count

            body = span or ""
            eid = str(hit.evidence_id or "")
            if not eid.startswith("chunk_") and method_cue_count(body) < 3:
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
        if (analysis.safety.get("unsupported_question") and not hit) or not span:
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

        cited = citation(hit, query=original)
        source_url = str(cited.get("source_url") or "")
        article_hit = (cited.get("source_kind") or getattr(hit, "source_kind", "") or "") == "wecare_article"
        span_is_amharic = bool(_ETHIOPIC.search(span or ""))

        # WeCare Amharic articles: return the span as-is for Amharic asks; MT only for English asks.
        if article_hit and span_is_amharic:
            amharic = with_article_cite(span[:700], source_url, amharic=True)
            if ask_amharic:
                return self._reply_from_analysis(
                    analysis,
                    amharic_reply=amharic,
                    english_query=english_query,
                    english_reply="",
                    status="translated",
                    cited=cited,
                    translation_backend="article",
                )
            if translator_loading:
                return self._reply_from_analysis(
                    analysis,
                    amharic_reply=amharic,
                    english_query=english_query,
                    english_reply=TEMPLATES["loading"]["am"],
                    status="source_only",
                    cited=cited,
                    translation_backend="article",
                )
            try:
                english = self.translator.to_english(span)
            except TranslatorNotReady:
                return self._reply_from_analysis(
                    analysis,
                    amharic_reply=amharic,
                    english_query=english_query,
                    english_reply=TEMPLATES["loading"]["en"] or span[:700],
                    status="source_only",
                    cited=cited,
                    translation_backend="article",
                )
            except Exception as exc:
                failed = self._reply_template("qa_fail", "qa_failed", original, english_query, analysis, cited)
                failed.qa_ok = False
                failed.qa_notes = [f"translation_error:{exc.__class__.__name__}"]
                failed.amharic_reply = amharic
                return failed
            if not english or _ETHIOPIC.search(english):
                english = span
            english = with_article_cite(english[:700], source_url, amharic=False)
            return self._reply_from_analysis(
                analysis,
                amharic_reply=amharic,
                english_query=english_query,
                english_reply=english,
                status="translated",
                cited=cited,
                translation_backend="mt",
            )

        # Legacy English-guideline path: compose Amharic from an English span.
        stored_am = (hit.amharic_text or "").strip() if hit else ""
        amharic, compose_backend, compose_notes = compose_amharic_reply(
            english_span=span if not span_is_amharic else (hit.english_text if hit else ""),
            document_amharic=stored_am if not article_hit else span,
            translator=None if translator_loading else self.translator,
        )
        if span_is_amharic and not amharic:
            amharic = span
            compose_backend = "article"
            compose_notes = ["native_amharic_span"]
        if amharic and _ETHIOPIC.search(amharic):
            notes = list(compose_notes)
            qa_flags: list[str] = []
            if self.translator.can_translate_reply and "mt_rendered" in compose_notes:
                qa_flags = reply_qa(span, amharic, getattr(self.translator, "last_qa_notes", None)) or []
                notes.extend(qa_flags)
            skip_qa_block = os.environ.get("WE_CARE_SKIP_REPLY_QA", "").strip().lower() in {
                "1",
                "true",
                "yes",
            }
            if qa_flags and not skip_qa_block:
                failed = self._reply_template("qa_fail", "qa_failed", original, english_query, analysis, cited)
                failed.qa_ok = False
                failed.qa_notes = qa_flags
                failed.english_reply = span if not span_is_amharic else ""
                return failed
            english_shown = shape_english_answer(span) if not span_is_amharic else ""
            if not english_shown and not span_is_amharic:
                english_shown = span
            shown = self._reply_from_analysis(
                analysis,
                amharic_reply=amharic[:700],
                english_query=english_query,
                english_reply=(english_shown or "")[:700],
                status="translated",
                cited=cited,
                translation_backend=compose_backend,
            )
            if qa_flags:
                shown.qa_ok = False
            shown.qa_notes = notes
            return shown

        mt_loading = translator_loading or any(
            note.startswith("mt_fallback:TranslatorNotReady") for note in compose_notes
        )
        if mt_loading:
            return self._reply_from_analysis(
                analysis,
                amharic_reply=TEMPLATES["loading"]["am"],
                english_query=english_query,
                english_reply=span,
                status="source_only",
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
        translation_backend: str | None = None,
    ) -> ReplyResponse:
        return ReplyResponse(
            amharic_reply=amharic_reply,
            english_query=english_query,
            english_reply=english_reply,
            translation_backend=translation_backend or getattr(self.translator, "backend", "map"),
            translation_status=status,  # type: ignore[arg-type]
            evidence_id=cited.get("evidence_id"),
            document_id=cited.get("document_id") or "",
            document_title=cited.get("document_title") or "",
            page=cited.get("page"),
            section=cited.get("section") or "",
            review_status=cited.get("review_status") or "",
            source_url=cited.get("source_url") or "",
            intent=analysis.intent,
            intent_labels=analysis.intent_labels,
            topic=analysis.topic,
            topics=list(analysis.topics or []),
            medical_concepts=list(analysis.medical_concepts or []),
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
