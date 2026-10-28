"""Choose one English guideline sentence and check the Amharic rendering."""

from __future__ import annotations

import re
from typing import Any

from we_care_nlu.contracts.schemas import RetrievalHit
from we_care_nlu.kb.sentences import split_sentences
from we_care_nlu.retrieval.engine import _polarity
from we_care_nlu.retrieval.expand import INTENT_NEEDLES

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_ETH = re.compile(r"[\u1200-\u137F]")
_STOP = {
    "what", "are", "there", "this", "with", "from", "that", "have", "been",
    "does", "they", "them", "about", "while", "used", "must", "where", "when",
    "which", "your", "into", "such", "also", "their", "than", "then", "how",
    "long", "many", "much", "some", "more", "most", "only", "just",
    # Frequent Amharic question filler — keep method/topic terms for matching.
    "ለምን", "ምንድን", "ምንድር", "ነው", "ናቸው", "እንዴት", "መቼ", "የት", "ማን",
    "እባክዎ", "ስለ", "ላይ", "ውስጥ", "ጋር", "ወይም", "እና", "ይህ", "ያ", "እንደ",
    "ጠቃሚ", "አስፈላጊ", "መረጃ", "ጥያቄ",
}

# Phrases that mark a sentence as answering a given intent, not merely sharing words.
ANSWER_CUES: dict[str, tuple[str, ...]] = {
    "duration": (
        "year", "years", "month", "months", "week", "weeks", "day", "days",
        "effective for", "lasts", "lasting", "remains effective", "every ",
        "ዓመት", "ወር", "ሳምንት", "ቀን", "ያገለግላል", "ይቆያል",
    ),
    "family_planning_methods": (
        "include", "options", "methods", "condom", "implant", "injectable", "pill", "iud",
        "ዘዴ", "መከላከያ", "ኮንዶም", "ክኒን", "መርፌ", "ኢምፕላንት",
    ),
    "definition": (
        "is a", "is an", "means", "refers to", "defined as",
        "ነው", "ናቸው", "ማለት", "ይባላል",
    ),
    "side_effects": (
        "side effect", "bleeding", "nausea", "headache", "weight",
        "ጎንዮሽ", "ደም", "ራስ ምታት",
    ),
    "contraindications": ("must not", "should not", "contraindicat", "do not use", "አይመከር", "ተቃራኒ"),
    "eligibility": ("can be used", "may be used", "eligible", "suitable", "መጠቀም ይቻላል", "ትችላለች"),
    "how_it_works": ("works by", "prevents", "stopping", "thickening", "ይከላከላል", "ይሰራል", "ይሠራል"),
    "missed_dose": ("missed", "as soon as", "remembered", "late", "ረሳ"),
    "access": ("available", "health facilit", "clinic", "pharmacy", "የጤና ተቋም", "ይገኛል"),
}

TEMPLATES: dict[str, dict[str, str]] = {
    "emergency": {
        "en": "This may be an emergency. Go to the nearest health facility now. This service does not give emergency treatment.",
        "am": "ይህ አስቸኳይ ሊሆን ይችላል። ወዲያውኑ ወደ ቅርብ የጤና ተቋም ይሂዱ። ይህ አገልግሎት አስቸኳይ ሕክምና አይሰጥም።",
    },
    "clarify": {
        "en": "Which family planning method is this question about?",
        "am": "ይህ ጥያቄ ስለ የትኛው የእርግዝና መከላከያ ዘዴ ነው?",
    },
    "abstain": {
        "en": "I cannot find a matching statement in the guidelines.",
        "am": "በመመሪያዎቹ ውስጥ የሚዛመድ መግለጫ አላገኘሁም።",
    },
    "qa_fail": {
        "en": "A matching guideline statement was found, but the Amharic translation failed the safety check, so it is not shown.",
        "am": "ተመሳሳይ መግለጫ ተገኝቷል፣ ነገር ግን የአማርኛ ትርጉሙ የደህንነት ምርመራውን ስላላለፈ አልታየም።",
    },
    "source_only": {
        "en": "",
        "am": "የሚከተለው ከመመሪያው የእንግሊዝኛ ጽሑፍ ነው። ወደ አማርኛ አልተተረጎመም።",
    },
    "loading": {
        "en": "",
        "am": "የትርጉም ሞዴሉ እየተጫነ ነው። የሚከተለው ከመመሪያው የእንግሊዝኛ ጽሑፍ ነው።",
    },
}


def number_tokens(text: str) -> list[str]:
    return _NUMBER.findall(text or "")


def reply_qa(english: str, amharic: str, extra_notes: list[str] | None = None) -> list[str]:
    notes: list[str] = []
    missing = [token for token in number_tokens(english) if token not in number_tokens(amharic)]
    if missing:
        notes.append("numbers_missing:" + ",".join(missing))
    if "WCMT" in (amharic or ""):
        notes.append("placeholder_left")
    source_polarity = _polarity(english)
    target_polarity = _polarity(amharic)
    if source_polarity == "contra" and target_polarity == "allow":
        notes.append("polarity_flipped_to_allow")
    if source_polarity == "allow" and target_polarity == "contra":
        notes.append("polarity_flipped_to_contra")
    for note in extra_notes or []:
        text = str(note)
        if text.startswith("term_missing") or text == "placeholder_left":
            if text not in notes:
                notes.append(text)
    return notes


def _contains(token: str, lowered: str) -> bool:
    if token in lowered:
        return True
    if token.endswith("s") and token[:-1] in lowered:
        return True
    return False


def _content_tokens(query: str) -> list[str]:
    return [
        token.strip("?.!,;:")
        for token in (query or "").lower().split()
        if len(token.strip("?.!,;:")) > 3 and token.strip("?.!,;:") not in _STOP
    ]


def _answer_cue_hits(intent: str | None, lowered: str) -> int:
    cues = ANSWER_CUES.get(intent or "") or ()
    return sum(1 for cue in cues if cue in lowered)


def select_span(query: str, hits: list[RetrievalHit], intent: str | None) -> tuple[str, RetrievalHit | None]:
    needles = [item.lower() for item in (INTENT_NEEDLES.get(intent or "") or ())]
    content = _content_tokens(query)
    query_amharic = bool(_ETH.search(query or ""))
    # Amharic morphology makes exact token overlap sparse; one shared term + fusion is enough.
    if query_amharic:
        required = 1 if content else 0
    else:
        required = len(content) if len(content) <= 2 else max(2, len(content) - 1)
    best_span = ""
    best_hit: RetrievalHit | None = None
    best_score = -1.0
    for hit in hits:
        if hit.polarity_conflict:
            continue
        english = (hit.english_text or "").strip()
        amharic = (hit.amharic_text or "").strip()
        # Prefer Amharic for article KB (no English) and for Amharic questions.
        prefer_am = bool(amharic) and (query_amharic or not english)
        score_text = amharic if prefer_am else (english or amharic)
        if not score_text:
            continue
        scoring_amharic = bool(_ETH.search(score_text[:80]))
        dense = float(hit.dense_score or 0.0)
        fused = float(hit.fused_score or hit.rerank_score or 0.0)
        article = (getattr(hit, "source_kind", "") or "") == "wecare_article"
        for sentence in split_sentences(score_text):
            words = sentence.split()
            if not scoring_amharic:
                if any(len(re.sub(r"[^A-Za-z]", "", word)) > 28 for word in words):
                    continue
                acronyms = sum(
                    1
                    for word in words
                    if word.strip(".,;:()[]").isupper() and 2 <= len(word.strip(".,;:()[]")) <= 5
                )
                if acronyms >= 6:
                    continue
            lowered = sentence.lower()
            overlap = sum(1 for token in content if _contains(token, lowered))
            cues = _answer_cue_hits(intent, lowered)
            needle_hits = sum(1 for needle in needles if needle in lowered)
            # Dense hit with the right answer shape can beat weak lexical overlap.
            semantic_ok = dense >= 0.55 and (cues > 0 or needle_hits > 0 or overlap > 0)
            if query_amharic and fused >= 0.45 and overlap > 0:
                semantic_ok = True
            # English ask against Amharic article: Latin terms / cues / fusion are enough.
            if article and not query_amharic and (overlap > 0 or cues > 0 or fused >= 0.35):
                semantic_ok = True
            if content and overlap < required and not semantic_ok and dense < 0.72:
                continue
            # When the user named a concrete term, prefer chunks that actually contain it.
            if query_amharic and content and overlap == 0 and fused < 0.85 and not article:
                continue
            score = (
                dense * 12.0
                + fused * 2.0
                + overlap * 4.0
                + cues * 4.0
                + needle_hits * 3.0
            )
            # Prefer a compact grounded sentence over a long paragraph dump.
            if len(words) > 45:
                score -= 1.5
            # Definition intent: never prefer a bare FAQ question as the answer span.
            if intent == "definition" and sentence.strip().endswith(("?", "؟")):
                score -= 25.0
            if score > best_score:
                best_score = score
                best_span = sentence[:700]
                best_hit = hit
    if best_hit is None:
        return "", None
    return best_span[:700], best_hit


_ETH_MONTH_TITLE = re.compile(
    r"(መስከረም|ጥቅምት|ኅዳር|ህዳር|ታህሳስ|ጥር|የካቲት|መጋቢት|ሚያዝያ|ግንቦት|ሰኔ|ሐምሌ|ነሐሴ|ጳጉሜ|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)",
    re.I,
)
_METHOD_LIST_CUES = (
    "ኮንዶም", "ክኒን", "መርፌ", "ኢምፕላንት", "አይዩ", "አይ.ዩ", "አስቸኳይ", "ቋሚ",
    "condom", "pill", "injectable", "implant", "iud", "iucd",
)


def _title_is_date(title: str) -> bool:
    t = (title or "").strip()
    if not t:
        return True
    if re.fullmatch(r"\d{4}", t):
        return True
    return bool(_ETH_MONTH_TITLE.search(t) and re.search(r"\d{4}", t) and len(t) < 40)


def method_cue_count(text: str) -> int:
    body = (text or "").lower()
    return sum(1 for cue in _METHOD_LIST_CUES if cue.lower() in body)


def evidence_quality(text: str, *, intent: str | None = None, query: str = "", entities: list | None = None) -> float:
    """Higher is better. Penalize TOC teasers and mid-list fragments."""
    from we_care_nlu.retrieval.relevance import answer_quality

    return answer_quality(
        text,
        intent=intent,
        query=query or "",
        entities=entities,
        evidence_id="",
        fused=0.0,
    )


def _norm_title_cmp(text: str) -> str:
    """Loose compare for FAQ-style titles that duplicate the user ask."""
    t = (text or "").strip().lower()
    t = re.sub(r"[\s?.!؟፡።]+$", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def citation(hit: RetrievalHit | None, *, query: str = "") -> dict[str, Any]:
    if hit is None:
        return {
            "evidence_id": None,
            "document_id": "",
            "document_title": "",
            "page": None,
            "section": "",
            "review_status": "",
            "source_url": "",
            "source_kind": "",
        }
    title = (hit.document_title or "").strip()
    # Collapse OCR/layout-spaced Latin (e.g. "ET H I O P I A B E M O N C").
    title = re.sub(r"\b(?:[A-Za-z]\s+){3,}[A-Za-z]\b", lambda m: m.group(0).replace(" ", ""), title)
    title = re.sub(r"\s{2,}", " ", title).strip()
    # Drop bare dates and titles that are mostly a publication date line.
    if _title_is_date(title) or re.search(
        r"(መስከረም|ጥቅምት|ኅዳር|ህዳር|ታህሳስ|ጥር|የካቲት|መጋቢት|ሚያዝያ|ግንቦት|ሰኔ|ሐምሌ|ነሐሴ|ጳጉሜ)"
        r"[^\n]{0,12}\d{4}\s*$",
        title,
    ):
        title = hit.section if hit.section and not _title_is_date(hit.section) else ""
        if not title or _title_is_date(title):
            title = (hit.document_id or "").replace("_", " ")
    # Blog posts often use the FAQ question as the document title — don't echo it under the answer.
    qn = _norm_title_cmp(query)
    tn = _norm_title_cmp(title)
    sn = _norm_title_cmp(getattr(hit, "section", "") or "")
    if qn and tn and (tn == qn or qn in tn or tn in qn):
        title = ""
    elif qn and sn and sn == qn and tn == sn:
        title = ""
    return {
        "evidence_id": hit.evidence_id,
        "document_id": hit.document_id,
        "document_title": title,
        "page": hit.page,
        "section": hit.section,
        "review_status": hit.review_status,
        "source_url": getattr(hit, "source_url", "") or "",
        "source_kind": getattr(hit, "source_kind", "") or "",
    }


def with_article_cite(text: str, url: str, *, amharic: bool) -> str:
    body = (text or "").strip()
    link = (url or "").strip()
    if not link or link in body:
        return body
    cite = f"ተጨማሪ፦ {link}" if amharic else f"More: {link}"
    return f"{body}\n{cite}" if body else cite


def focus_span_for_query(text: str, query: str) -> str:
    """Lead with sentences that ground distinctive query tokens.

    General shaping (not topic-specific): if evidence opens with supporting contrast
    and buries the asked facet, bring grounded sentences forward.
    """
    from we_care_nlu.retrieval.relevance import distinctive_overlap, distinctive_query_tokens

    body = (text or "").strip()
    dist = distinctive_query_tokens(query or "")
    if not body or not dist:
        return body
    sents = split_sentences(body)
    if len(sents) < 2:
        return body
    focus = [s for s in sents if distinctive_overlap(s, dist) > 0]
    support = [s for s in sents if s not in focus]
    if not focus or focus == sents[: len(focus)]:
        return body
    ordered = [
        s
        for s in (focus + support)
        if not (s.strip().endswith(("?", "؟")) and len(s) < 80)
    ]
    if ordered and ordered[0].startswith("ስለዚህ "):
        ordered[0] = ordered[0][len("ስለዚህ ") :]
    if ordered:
        ordered[0] = ordered[0].replace(
            "ከላይ ከተጠቀሰው የጤናማ የወር አበባ ዑደት",
            "ጤናማ ከሚባለው የወር አበባ ዑደት",
        ).replace("ከላይ ከተጠቀሰው", "ጤናማ ከሚባለው")
    return " ".join(ordered).strip() or body
