"""Shared constants, language codes, and medical cue lexicons."""

from __future__ import annotations

NLLB_SRC_LANG = "eng_Latn"
NLLB_TGT_LANG = "amh_Ethi"

ETHIOPIC_RANGE = (0x1200, 0x137F)
ETHIOPIC_SUPPLEMENT = (0x1380, 0x139F)
ETHIOPIC_EXTENDED = (0x2D80, 0x2DDF)
ETHIOPIC_EXTENDED_A = (0xAB00, 0xAB2F)

PLACEHOLDER_PREFIX = "WCMT"
PLACEHOLDER_PATTERN = r"WCMT[0-9]{4}"

# Sentence-splitter abbreviations common in MoH / clinical English.
ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr",
    "fig", "figs", "eq", "no", "nos", "vol", "pp", "ed", "eds",
    "inc", "ltd", "st", "ave", "dept", "govt",
    "vs", "etc", "al", "e.g", "i.e", "cf",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "mg", "ml", "mcg", "iu", "im", "iv", "sc", "po", "od", "bd", "tid", "qid",
    "anc", "pnc", "fp", "hiv", "aids", "sti", "rtis", "art", "pmtct", "epi",
    "opd", "cpr", "mmr", "tfr", "iucd", "iud", "dmpa", "lng", "coc", "pop",
    "ec", "bemonc", "cemonc", "larc", "lam", "sam", "hep", "mchn", "moh",
    "rh", "srh", "rmncah", "rmnch", "pph", "via", "htc", "pitc", "cac",
    "ayh", "roc", "hspt", "hspt-ii", "sdg", "mdg", "edhs", "lb",
}

# English cue words used by automated QA (not a clinical ontology).
NEGATION_CUES_EN = {
    "not", "no", "never", "none", "neither", "nor", "without",
    "avoid", "do not", "don't", "must not", "cannot", "can't",
    "contraindicated", "contraindication", "prohibited", "forbid",
    "is not indicated", "are not indicated", "should not",
}

NEGATION_CUES_AM = {
    "አይደለም", "የለም", "አይገባም", "አይቻልም", "ተከለከለ", "የተከለከለ",
    "አይመከርም", "አትጠቀሙ", "አይጠቀሙ", "በጭራሽ", "እንዳይ", "አለመኖር",
}

WARNING_CUES_EN = {
    "warning", "caution", "danger", "risk", "serious", "severe",
    "adverse", "side effect", "life-threatening", "emergency",
    "stop immediately", "seek medical", "do not use",
}

WARNING_CUES_AM = {
    "ማስጠንቀቂያ", "ጥንቃቄ", "አደጋ", "ከባድ", "የጎንዮሽ", "አስቸኳይ",
    "ወዲያውኑ ያቁሙ", "ሕክምና ይፈልጉ",
}

CONTRAINDICATION_CUES_EN = {
    "contraindication", "contraindicated", "do not use if",
    "must not be used", "not recommended in", "absolute contraindication",
    "relative contraindication",
}

CONTRAINDICATION_CUES_AM = {
    "ተቃራኒ ሁኔታ", "የተከለከለ", "መከልከያ", "አይመከርም",
}

PREGNANCY_CUES_EN = {
    "pregnan", "postpartum", "post-partum", "antenatal", "prenatal",
    "postnatal", "trimester", "lactat", "breastfeed", "fetus", "foetus",
    "gestation", "miscarriage", "abortion",
}

AGE_PATTERNS = (
    r"\b(?:under|over|above|below|aged?)\s+\d{1,3}\b",
    r"\b\d{1,3}\s*(?:years?|yrs?|months?|weeks?|days?)\s*(?:old|of age)?\b",
    r"\b(?:adolescent|teenager|neonate|infant|newborn|under-?five|under 5)\b",
)

DURATION_PATTERNS = (
    r"\b\d+(?:\.\d+)?\s*(?:hour|hr|day|week|month|year|minute|min)s?\b",
    r"\b(?:once|twice|three times)\s+(?:daily|a day|weekly|a week)\b",
    r"\b(?:every|for|after|within)\s+\d+(?:\.\d+)?\s*(?:hour|hr|day|week|month|year)s?\b",
)

DOSAGE_UNIT_PATTERN = (
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|kg|ml|mL|L|iu|IU|%|mmHg|mmol|mEq)\b"
)

# Document-derived repeating footer/header fragments (Family Planning guideline).
KNOWN_RUNNING_HEADERS = (
    "MoH, MCHN directorate, FP Service Integration National Implementation Guideline",
    "MoH, MCHN Directorate, FP Service Integration National Implementation Guideline",
)
