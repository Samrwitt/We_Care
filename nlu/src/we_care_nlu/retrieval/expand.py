"""Additive query expansion. Does not replace polarity checks."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import yaml

from we_care_nlu.config import package_root

METHOD_AM = {
    "injectable_contraceptive": "መርፌ",
    "implant": "ኢምፕላንት",
    "iud": "አይዩሲዲ",
    "condom": "ኮንዶም",
    "oral_pill": "ክኒን",
    "emergency_contraception": "የአስቸኳይ መከላከያ",
    "sterilization": "ቋሚ ዘዴ",
    "lam": "ጡት ማጥባት መከላከያ",
}

INTENT_NEEDLES = {
    "side_effects": ("የጎንዮሽ ጉዳት", "ጎንዮሽ", "side effect", "irregular bleeding", "መደበኛ ያልሆነ", "ክብደት"),
    "contraindications": ("የተከለከለ", "ተቃራኒ", "አይመከርም", "contraindicat", "must not", "should not use"),
    "eligibility": ("መጠቀም ይቻላል", "መጠቀም እችላለሁ", "can be used", "may be used", "eligible", "safe for"),
    "definition": ("is a", "is an", "ማለት", "ይባላል", "ነው።"),
    "access": ("ይገኛል", "እገኛለሁ", "available", "health facilit", "clinic", "ጤና ጣቢያ"),
    "duration": ("ዓመት", "አመት", "ሳምንት", "ወር", "years", "weeks", "months", "every", "ይቆያል"),
    "family_planning_methods": ("ዘዴዎች", "አማራጮች", "options", "include", "methods", "ዘመናዊ"),
    "how_it_works": ("ይሠራል", "ይሰራል", "works by", "stopping ovulation", "እንዴት"),
    "missed_dose": ("ከተረሳ", "ረሳሁ", "missed", "forgot", "as soon as remembered"),
    "treatment": ("ሕክምና", "ህክምና", "መድኃኒት", "መድሃኒት", "ይታከማል", "ላድርግ", "ART", "antiretroviral", "therapy", "treatment"),
    "medication": ("መድኃኒት", "መድሃኒት", "ክኒን", "ART", "antiretroviral", "medicine", "drug"),
    "symptoms": ("ምልክት", "ምልክቶች", "symptom", "signs", "ሕመም", "ህመም", "ይታወቃል"),
    "prevention": ("መከላከል", "prevention", "prevent", "እንዴት እከላከላለሁ"),
    "causes": ("ምክንያት", "cause", "caused", "ለምን ይከሰታል", "ያስከትላል"),
    "screening": ("ምርመራ", "screening", "test", "ስክሪን"),
    "effectiveness": ("ውጤታማ", "effectiveness", "how effective", "ይከላከላል"),
    "dosage": ("መጠን", "dosage", "dose", "ሚሊግራም", "mg"),
}

# Lexicon/topic labels → retrieval surfaces (for short follow-ups like "እንዴት ይታከማል?").
TOPIC_SURFACES: dict[str, tuple[str, ...]] = {
    "hiv": ("ኤች.አይ.ቪ", "ኤችአይቪ", "HIV", "ኤድስ", "AIDS", "ኤ.አር.ቲ", "ART", "antiretroviral"),
    "sti": ("አባላዘር", "STI", "STD", "ኤስቲአይ", "ጨብጥ", "ቂጥኝ", "ቻላሚዲያ"),
    "pregnancy": ("እርግዝና", "እርጉዝ", "ነፍሰ ጡር", "pregnancy", "pregnant"),
    "breastfeeding": ("ጡት ማጥባት", "ጡት እያጠባሁ", "breastfeeding", "ማጥባት"),
    "condom": ("ኮንዶም", "condom"),
    "family_planning": ("የቤተሰብ ምጣኔ", "የእርግዝና መከላከያ", "family planning", "birth control", "contraception"),
    "contraceptive_implant": ("ኢምፕላንት", "በትር", "implant", "ጀዴል"),
    "contraceptive_injection": ("መርፌ", "ዴፖ", "injectable", "DMPA"),
    "contraceptive_pill": ("ክኒን", "ኪኒን", "pill", "OCP"),
    "iud": ("አይዩሲዲ", "አይዩዲ", "IUD", "IUCD", "ሉፕ"),
    "emergency_contraception": ("የአስቸኳይ መከላከያ", "ፖስት ፒል", "emergency contraception", "morning after"),
    "sterilization": ("ቋሚ", "vasectomy", "tubal", "sterilization"),
    "menstruation": ("የወር አበባ", "ወር አበባ", "ሲያመኝ", "ቁርጠት", "period", "menstrual", "dysmenorrhea"),
    "irregular_menses": ("መዛባት", "ተዛባ", "irregular period", "irregular cycle", "ዑደት መዛባት"),
    "cryptic_pregnancy": ("ክሪፕቲክ", "ስውር እርግዝና", "cryptic pregnancy"),
    "endometriosis": ("ኢንዶሜትሪዮሲስ", "endometriosis"),
    "pcos": ("ፒሲኦኤስ", "PCOS", "polycystic"),
    "ectopic_pregnancy": ("ከማሕፀን ውጪ", "ከማህፀን ውጭ", "ectopic"),
    "antenatal_care": ("ቅድመ ወሊድ", "የእርግዝና ክትትል", "ANC", "antenatal"),
    "postpartum": ("ከወሊድ በኋላ", "ድህረ ወሊድ", "postpartum", "PPFP"),
    "vaginal_discharge": ("ፈሳሽ", "vaginal discharge", "yeast"),
    "genital_ulcer": ("ቁስለት", "genital ulcer", "herpes", "ሄርፒስ"),
    "uti": ("ሽንት", "UTI", "urinary"),
    "cervical_cancer": ("ማሕፀን ጫፍ", "cervical", "HPV", "ፓፕ"),
    "abortion_care": ("ውርጃ", "abortion", "miscarriage"),
    "fertility": ("መካንነት", "fertility", "infertility", "ማርገዝ"),
}


@lru_cache(maxsize=1)
def synonym_groups() -> list[list[str]]:
    path: Path = package_root() / "data" / "synonyms" / "amharic_medical.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    groups = []
    for terms in (data.get("groups") or {}).values():
        cleaned = [str(t).strip() for t in (terms or []) if str(t).strip()]
        if cleaned:
            groups.append(cleaned)
    return groups


def unique_join(parts: Iterable[str]) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        token = (part or "").strip()
        key = token.lower()
        if not token or key in seen:
            continue
        seen.add(key)
        out.append(token)
    return " ".join(out)


_QUERY_FILLER = {
    "ምንድን",
    "ምንድር",
    "ምንድረ",
    "ምንድ",
    "ነው",
    "ናቸው",
    "ለምን",
    "እንዴት",
    "መቼ",
    "የት",
    "ማን",
    "እባክዎ",
    "ስለ",
    "what",
    "is",
    "are",
    "how",
    "why",
    "when",
    "where",
    "which",
    "does",
    "do",
    "the",
    "a",
    "an",
    "please",
    "tell",
    "me",
}


def _normalize_am_acronyms(text: str) -> str:
    """Collapse dotted Ethiopic/Latin acronyms: አይ.ዩ.ዲ → አይዩዲ, I.U.D → IUD."""
    import re

    text = re.sub(r"(?<=[\u1200-\u137FA-Za-z])\.(?=[\u1200-\u137FA-Za-z])", "", text or "")
    return text


def expand_query(
    text: str,
    extra: Iterable[str] | None = None,
    *,
    concept_terms: Iterable[str] | None = None,
) -> str:
    text = _normalize_am_acronyms(text)
    # Drop bare question fillers so BM25 does not rank every FAQ title first.
    kept = [tok for tok in (text or "").split() if tok.strip("?.!፡።").lower() not in _QUERY_FILLER]
    core = " ".join(kept) if kept else text
    bits = [text, core]
    low = core.lower()
    compact = _normalize_am_acronyms(low.replace(" ", ""))
    for terms in synonym_groups():
        if any(term.lower() in low or _normalize_am_acronyms(term.lower()) in compact for term in terms):
            bits.extend(terms)
    # Approved concept synonyms only (never pending_review candidates).
    if concept_terms:
        bits.extend(concept_terms)
    if extra:
        bits.extend(extra)
    return unique_join(bits)


def method_terms(entities: Iterable[object]) -> list[str]:
    terms: list[str] = []
    for ent in entities or []:
        typ = getattr(ent, "type", None) or (ent.get("type") if isinstance(ent, dict) else None)
        if typ != "contraceptive_method":
            continue
        value = getattr(ent, "value", None) or (ent.get("value") if isinstance(ent, dict) else "")
        surface = getattr(ent, "surface", None) or (ent.get("surface") if isinstance(ent, dict) else "")
        if surface:
            terms.append(str(surface))
        if value:
            terms.append(str(value))
            terms.append(METHOD_AM.get(str(value), ""))
    return [t for t in terms if t]
