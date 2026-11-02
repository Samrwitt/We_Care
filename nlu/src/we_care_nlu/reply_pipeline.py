"""Build user-facing replies from analysis + retrieved spans.

Called by NLUEngine.reply(). Keep pipeline.py focused on analyze() + wiring.
"""

from __future__ import annotations

import os
import re
from typing import Any

from we_care_nlu.contracts.schemas import AnalyzeRequest, AnalyzeResponse, ReplyResponse
from we_care_nlu.preprocess.normalize import detect_language
from we_care_nlu.safety.rules import emergency_flags
from we_care_nlu.translation.advice import advice_key, intent_advice
from we_care_nlu.translation.compose import compose_amharic_reply, shape_english_answer
from we_care_nlu.translation.mt_client import TranslatorNotReady
from we_care_nlu.translation.phrase_map import covered_question, map_to_amharic_terms, map_to_english
from we_care_nlu.translation.reply import (
    TEMPLATES,
    citation,
    focus_span_for_query,
    reply_qa,
    select_span,
    tidy_reply_span,
    with_article_cite,
)

_ETHIOPIC = re.compile(r"[\u1200-\u137F]")


class ReplyMixin:
    """Mixin mixed into NLUEngine: turn an AnalyzeResponse into ReplyResponse text.

    Steps inside reply():
      1. Detect ask language and build search/english_query strings
      2. Call self.analyze(...)
      3. Rank retrieval hits → pick a span
      4. Article / compose / advice → ReplyResponse
    """

    def _advice_for(self, analysis: AnalyzeResponse) -> tuple[str, str] | None:
        """Template advice keyed by clinical topic when the question-type intent is generic."""
        key = advice_key(analysis.intent, analysis.topic)
        advice = intent_advice(analysis.intent, analysis.topic)
        if advice and key and key != analysis.intent:
            analysis.intent = key
            labels = list(analysis.intent_labels or [])
            if key not in labels:
                labels = [key] + labels
            analysis.intent_labels = labels
        return advice

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
            # Boost Amharic KB recall for English asks (methods, conditions, FP terms).
            am_terms = map_to_amharic_terms(original, limit=16)
            search_text = " ".join([original] + am_terms) if am_terms else original
            search_lang = "en"
        analysis = self.analyze(
            AnalyzeRequest(
                text=original,
                language=search_lang,
                conversation_id=req.conversation_id,
                context=req.context,
                filters=req.filters,
                top_k=req.top_k or 20,
                model_key=req.model_key,
            ),
            retrieval_text=search_text if search_text != original else None,
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
            # Match reply language to the ask: English asks prefer English evidence.
            if ask_amharic:
                text = stored if (stored and _ETHIOPIC.search(stored)) else english
            else:
                text = english if english else stored
            if not text or len(text) < 25:
                continue
            # Relevance uses Amharic body when available so soft-topic cues still fire.
            check_blob = stored if (stored and _ETHIOPIC.search(stored)) else text
            if not relevant_enough(
                check_blob,
                query=span_query if ask_amharic else (search_text or span_query),
                intent=analysis.intent,
                entities=analysis.entities,
                need_tokens=need,
            ):
                continue
            quality = answer_quality(
                check_blob,
                intent=analysis.intent,
                query=span_query if ask_amharic else (search_text or span_query),
                entities=analysis.entities,
                evidence_id=str(candidate.evidence_id or ""),
                fused=float(candidate.fused_score or 0),
            )
            if quality < 0:
                continue
            # English asks: strongly prefer passages that already have English text.
            if not ask_amharic and english:
                quality += 2.5
            elif not ask_amharic and text and _ETHIOPIC.search(text):
                quality -= 4.0
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
                if (
                    distinctive_overlap(row[4], dist) > 0
                    or distinctive_overlap((row[3].amharic_text or ""), dist) > 0
                    or distinctive_overlap((row[3].english_text or ""), dist) > 0
                )
                and not is_demo_chunk(row[3])
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
            span = tidy_reply_span(focus_span_for_query(span, span_query)[:900])
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
                advice = self._advice_for(analysis)
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
            advice = self._advice_for(analysis)
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
                advice = self._advice_for(analysis)
                english = advice[1] if advice else (TEMPLATES.get("abstain", {}).get("en") or "")
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
                failed.english_reply = self._english_reply_body(hit, span, ask_amharic=ask_amharic)
                return failed
            english_shown = self._english_reply_body(hit, span, ask_amharic=ask_amharic)
            if not ask_amharic and (not english_shown or _ETHIOPIC.search(english_shown)):
                # Never show Amharic as the English answer — fall back to intent advice.
                advice = self._advice_for(analysis)
                if advice:
                    return self._reply_from_analysis(
                        analysis,
                        amharic_reply=advice[0],
                        english_query=english_query,
                        english_reply=advice[1],
                        status="template",
                        cited=cited,
                    )
                english_shown = TEMPLATES.get("abstain", {}).get("en") or ""
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
        english_fallback = self._english_reply_body(hit, span, ask_amharic=ask_amharic)
        if not ask_amharic and (not english_fallback or _ETHIOPIC.search(english_fallback)):
            advice = self._advice_for(analysis)
            if advice:
                return self._reply_from_analysis(
                    analysis,
                    amharic_reply=advice[0],
                    english_query=english_query,
                    english_reply=advice[1],
                    status="template",
                    cited=cited,
                )
            english_fallback = TEMPLATES.get("abstain", {}).get("en") or ""
        if mt_loading:
            return self._reply_from_analysis(
                analysis,
                amharic_reply=TEMPLATES["loading"]["am"],
                english_query=english_query,
                english_reply=english_fallback if not ask_amharic else span,
                status="source_only",
                cited=cited,
            )

        notice = TEMPLATES["source_only"]["am"]
        return self._reply_from_analysis(
            analysis,
            amharic_reply=notice if ask_amharic else (amharic or notice)[:700],
            english_query=english_query,
            english_reply=english_fallback if not ask_amharic else (span if not span_is_amharic else ""),
            status="source_only",
            cited=cited,
        )

    def _english_reply_body(self, hit: Any, span: str, *, ask_amharic: bool) -> str:
        """Prefer curated English evidence for English asks; never echo Amharic as English."""
        span = (span or "").strip()
        en_src = ((getattr(hit, "english_text", None) or "") if hit else "").strip()
        if not ask_amharic:
            if en_src:
                shaped = shape_english_answer(en_src) or en_src
                return shaped[:700]
            if span and not _ETHIOPIC.search(span):
                return (shape_english_answer(span) or span)[:700]
            if span and getattr(self.translator, "can_translate_reply", False):
                try:
                    translated = self.translator.to_english(span)
                    if translated and not _ETHIOPIC.search(translated):
                        return translated[:700]
                except Exception:
                    pass
            return ""
        if span and not _ETHIOPIC.search(span):
            return (shape_english_answer(span) or span)[:700]
        if en_src:
            return (shape_english_answer(en_src) or en_src)[:700]
        return ""

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
        # Hard guard: never put Ethiopic text in english_reply (UI shows it for EN asks).
        if english_reply and _ETHIOPIC.search(english_reply):
            advice = self._advice_for(analysis)
            english_reply = (advice[1] if advice else "") or TEMPLATES.get("abstain", {}).get("en") or ""
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

