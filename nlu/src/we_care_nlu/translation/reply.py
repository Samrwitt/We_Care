"""Choose one English guideline sentence and check the Amharic rendering."""

from __future__ import annotations

import re
from typing import Any

from we_care_nlu.contracts.schemas import RetrievalHit
from we_care_nlu.kb.sentences import split_sentences
from we_care_nlu.retrieval.engine import _polarity
from we_care_nlu.retrieval.expand import INTENT_NEEDLES

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_STOP = {
    "what", "are", "there", "this", "with", "from", "that", "have", "been",
    "does", "they", "them", "about", "while", "used", "must", "where", "when",
    "which", "your", "into", "such", "also", "their", "than", "then",
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
}


def number_tokens(text: str) -> list[str]:
    return _NUMBER.findall(text or "")


def reply_qa(english: str, amharic: str) -> list[str]:
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
    return notes


def _contains(token: str, lowered: str) -> bool:
    if token in lowered:
        return True
    if token.endswith("s") and token[:-1] in lowered:
        return True
    return False


def select_span(query: str, hits: list[RetrievalHit], intent: str | None) -> tuple[str, RetrievalHit | None]:
    needles = [item.lower() for item in (INTENT_NEEDLES.get(intent or "") or ())]
    content = [
        token.strip("?.!,;:")
        for token in (query or "").lower().split()
        if len(token.strip("?.!,;:")) > 3 and token.strip("?.!,;:") not in _STOP
    ]
    required = len(content) if len(content) <= 2 else max(2, len(content) - 1)
    best_span = ""
    best_hit: RetrievalHit | None = None
    best_score = 0
    for hit in hits:
        if hit.polarity_conflict:
            continue
        text = (hit.english_text or "").strip()
        if not text:
            continue
        for sentence in split_sentences(text):
            words = sentence.split()
            if any(len(re.sub(r"[^A-Za-z]", "", word)) > 28 for word in words):
                continue
            acronyms = sum(
                1 for word in words if word.strip(".,;:()[]").isupper() and 2 <= len(word.strip(".,;:()[]")) <= 5
            )
            if acronyms >= 6:
                continue
            lowered = sentence.lower()
            overlap = sum(1 for token in content if _contains(token, lowered))
            if content and overlap < required:
                continue
            score = overlap * 2 + sum(3 for needle in needles if needle in lowered)
            if score > best_score:
                best_score = score
                best_span = sentence
                best_hit = hit
    if best_hit is None:
        return "", None
    return best_span[:700], best_hit


def citation(hit: RetrievalHit | None) -> dict[str, Any]:
    if hit is None:
        return {
            "evidence_id": None,
            "document_id": "",
            "document_title": "",
            "page": None,
            "section": "",
            "review_status": "",
        }
    return {
        "evidence_id": hit.evidence_id,
        "document_id": hit.document_id,
        "document_title": hit.document_title,
        "page": hit.page,
        "section": hit.section,
        "review_status": hit.review_status,
    }
