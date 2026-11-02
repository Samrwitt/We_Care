"""What the user ask is 'about': distinctive tokens and soft topic groups.

Used by retrieval scoring and reply span shaping.
"""

from __future__ import annotations

import re
from typing import Any

_QUERY_STOP = {
    "what", "are", "there", "this", "with", "from", "that", "have", "been",
    "does", "they", "them", "about", "while", "used", "must", "where", "when",
    "which", "your", "into", "such", "also", "their", "than", "then", "how",
    "long", "many", "much", "some", "more", "most", "only", "just", "the", "and",
    "ለምን", "ምንድን", "ምንድር", "ምንድ", "ምን", "ነው", "ናቸው", "እንዴት", "መቼ", "የት", "ማን",
    "እባክዎ", "ስለ", "ላይ", "ውስጥ", "ጋር", "ወይም", "እና", "ይህ", "ያ", "እንደ",
    "ጠቃሚ", "አስፈላጊ", "መረጃ", "ጥያቄ", "አለብኝ", "እችላለሁ", "ይቻላል", "ይገኛል",
    "የሚባለው", "ሚባለው", "የሚባል", "ሚባል",
}

_GENERIC_TOPIC = {
    "እርግዝና", "እርግዝ", "መከላከያ", "መከላከል", "ዘዴ", "ዘዴዎች", "ሴት", "ሴቶች", "ቤተሰብ", "ምጣኔ",
    "አገልግሎት", "ጤና", "አማራጭ", "አማራጮች", "አሉ", "ይቻላል",
    # Broad body/topic words — not enough alone to pick an article (need ሲያመኝ / ተዛባ / …).
    "ወር", "አበባ", "ዑደት", "የወር", "ጤናማ", "ሰው", "ሰዎች", "ሕመምተኛ",
    "pregnancy", "contraception", "contraceptive", "method",
    "methods", "family", "planning", "woman", "women", "health", "service",
    "options", "option", "period", "menstrual", "menstruation", "cycle",
}

_DISTINCTIVE_ALIASES: tuple[tuple[str, ...], ...] = (
    ("ሲያመኝ", "ያመኛል", "ያመማል", "አመመ", "አመም", "ህመም", "ሕመም", "ቁርጠት", "pain", "cramp", "ache", "dysmenorrhea"),
    ("ላድርግ", "ላደርግ", "ማድረግ", "ማስታገስ", "ያስታግሳል", "መድኃኒት", "መድሃኒት", "ibuprofen", "paracetamol", "NSAID"),
    ("ተዛባ", "መዛባት", "ተዛቡ", "irregular", "oligomenorrhea", "amenorrhea"),
    ("ክሪፕቲክ", "ስውር", "ሚስጥራዊ", "cryptic", "denied pregnancy"),
    ("ኤች.አይ.ቪ", "ኤችአይቪ", "HIV", "ኤድስ", "ART", "antiretroviral"),
    ("አይዩሲዲ", "አይዩዲ", "IUD", "IUCD", "ሉፕ", "intrauterine"),
    ("ኢምፕላንት", "በትር", "implant", "jadelle"),
    ("ፖስት ፒል", "አስቸኳይ", "emergency contraception", "morning after"),
)

_SOFT_TOPIC_CUES: tuple[tuple[str, ...], ...] = (
    # Keep cues specific — short "ኢንዶሜትሪ" also matches unrelated "ኢንዶሜትሪተስ" (endometritis).
    (
        "ወር አበባ",
        "የወር አበባ",
        "menstrual",
        "dysmenorrhea",
        "ዲስሜኖሬያ",
        "endometriosis",
        "ኢንዶሜትሪዮሲስ",
        "መዛባት",
        "ተዛባ",
        "irregular period",
        "irregular cycle",
    ),
    ("ኤች.አይ.ቪ", "ኤችአይቪ", "hiv", "ኤድስ", "aids", "art", "antiretroviral"),
    ("እርግዝና", "pregnant", "pregnancy", "እርጉዝ", "ነፍሰ ጡር", "cryptic", "ክሪፕቲክ"),
    ("አባላዘር", "sti", "std", "ጨብጥ", "ቂጥኝ", "chlamydia", "gonorrhea"),
    ("ኮንዶም", "condom"),
    ("ኢምፕላንት", "implant", "በትር"),
    ("መርፌ", "injectable", "dmpa", "ዴፖ"),
    ("አይዩ", "iud", "iucd", "ሉፕ"),
)

_UNSPECIFIED_DISTRACTORS: tuple[tuple[str, ...], ...] = (
    ("ራስ ምታት", "migraine", "headache"),
    ("ቂጥኝ", "syphilis"),
    ("ጨብጥ", "gonorrhea", "gonorrh"),
    ("ክላሚዲያ", "chlamydia"),
    ("ኤችኤስቪ", "hsv", "herpes"),
    ("ፒአይዲ", "pid", "pelvic inflammatory", "ኢንዶሜትሪተስ", "endometritis"),
    ("ከማኅፀን ውጪ", "ectopic", "ሜቶትሪክሴት", "methotrexate"),
    # Period-pain asks must not latch onto labour analgesia / BF injectable chunks.
    ("ምጥ", "labour", "labor pain", "epidural", "በኢኮኖሚ በአደጉት"),
    ("ጡት ማጥባት", "breastfeeding", "lactational", "ፕሮጄስቲን ብቻ ያለው መርፌ"),
)

_AM_SUFFIXES = ("ዎች", "ዎት", "ዋ", "ው", "ን", "ም")

_AM_PREFIXES = ("የ", "በ", "ለ", "ከ", "ስለ", "ወደ", "እንደ")

def _strip_am_prefix(tok: str) -> str:
    for pref in _AM_PREFIXES:
        if tok.startswith(pref) and len(tok) - len(pref) >= 3:
            return tok[len(pref) :]
    return tok

def _normalize_am_token(tok: str) -> str:
    """Strip light clitics so እርግዝናን → እርግዝና (generic), not a false distinctive."""
    tok = _strip_am_prefix(tok)
    for suf in _AM_SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            stem = tok[: -len(suf)]
            # Only collapse when the stem is already a known filler/topic — never
            # invent short stems like የሚባለው → ሚባለ (matches every "የሚባለው" sentence).
            if stem in _GENERIC_TOPIC or stem in _QUERY_STOP:
                return stem
            if len(suf) >= 2:
                return stem
    return tok

def distinctive_query_tokens(query: str) -> list[str]:
    """Topic tokens that must appear in a grounded answer when present in the ask.

    Filters question fillers and broad words like እርግዝና/መከላከያ/አበባ so a period-pain
    ask cannot be answered by an irregular-cycle article that only shares 'ወር አበባ'.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in (query or "").lower().split():
        bare = raw.strip("?.!,;:፡።፣፤\"'()[]")
        if len(bare) < 3 or bare in _QUERY_STOP or bare in _GENERIC_TOPIC:
            continue
        tok = _normalize_am_token(bare)
        if len(tok) < 3 or tok in _QUERY_STOP or tok in _GENERIC_TOPIC:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out

def _distinctive_aliases(tok: str) -> list[str]:
    """Return tok plus stems/synonyms for soft Amharic paraphrase matching."""
    t = (tok or "").strip().lower()
    if not t:
        return []
    aliases = [t]
    for pref in ("ሲ", "ያ", "የ", "ት", "እንዲ", "በ"):
        if t.startswith(pref) and len(t) - len(pref) >= 3:
            aliases.append(t[len(pref) :])
    for group in _DISTINCTIVE_ALIASES:
        gl = tuple(x.lower() for x in group)
        if any(a in gl for a in aliases):
            aliases.extend(gl)
            break
    # unique, stable
    seen: set[str] = set()
    out: list[str] = []
    for a in aliases:
        if a and a not in seen:
            seen.add(a)
            out.append(a)
    return out

def distinctive_overlap(blob: str, tokens: list[str] | None) -> int:
    if not tokens:
        return 0
    low = (blob or "").lower()
    n = 0
    for t in tokens:
        if any(a and a in low for a in _distinctive_aliases(t)):
            n += 1
    return n

def distinctive_coverage(blob: str, tokens: list[str] | None) -> float:
    """Fraction of distinctive query tokens grounded in the candidate (0–1)."""
    if not tokens:
        return 1.0
    return distinctive_overlap(blob, tokens) / float(len(tokens))

def soft_topic_groups(query: str) -> list[tuple[str, ...]]:
    """Domain cue groups mentioned in the query (period, HIV, pregnancy, …)."""
    q = (query or "").lower()
    groups: list[tuple[str, ...]] = []
    for cues in _SOFT_TOPIC_CUES:
        if any(c.lower() in q for c in cues):
            groups.append(cues)
    return groups

def soft_topic_hit(
    blob: str,
    groups: list[tuple[str, ...]] | None,
    *,
    query: str = "",
) -> bool:
    """Require the domain the user actually named (e.g. ወር አበባ), not a cousin disease."""
    if not groups:
        return True
    low = (blob or "").lower()
    q = (query or "").lower()
    for group in groups:
        # Cues the user uttered are required in the answer when present.
        primaries = [c for c in group if c.lower() in q]
        if primaries:
            if any(p.lower() in low for p in primaries):
                return True
            continue
        if any(c.lower() in low for c in group):
            return True
    return False

def unspecified_distractor(query: str, blob: str) -> bool:
    """True when blob focuses on a specific illness the user did not name."""
    q = (query or "").lower()
    b = (blob or "").lower()
    for cues in _UNSPECIFIED_DISTRACTORS:
        if any(c.lower() in b for c in cues) and not any(c.lower() in q for c in cues):
            return True
    return False

