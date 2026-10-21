"""Conservative Amharic preprocessing. Negation and clinical wording are kept."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Ethiopic punctuation → ASCII cousins for matching only; originals kept in `original`.
_PUNCT = str.maketrans(
    {
        "።": ".",
        "፣": ",",
        "፤": ";",
        "፥": ":",
        "፦": ":",
        "፧": "?",
        "፨": " ",
        "፡": " ",
        "\u00a0": " ",
        "\u200b": "",
        "\ufeff": "",
    }
)

# Common keyboard/spelling variants that do not flip clinical polarity.
_SPELLING = [
    (re.compile(r"ነዉ"), "ነው"),
    (re.compile(r"ናቸዉ"), "ናቸው"),
    (re.compile(r"አሉዋል"), "አሉ"),
    (re.compile(r"እርግዝናን"), "እርግዝናን"),
    (re.compile(r"መርፌው"), "መርፌ"),
    (re.compile(r"ክኒኑ"), "ክኒን"),
    (re.compile(r"ኮንዶሙ"), "ኮንዶም"),
    (re.compile(r"ኢምፕላኑ"), "ኢምፕላንት"),
    (re.compile(r"ዴፖ\s*ፕሮቬራ"), "መርፌ"),
    (re.compile(r"እንዴት ይሰራል"), "እንዴት ይሠራል"),
    (re.compile(r"የጎንዮሽ\s*ጉዳቶች"), "የጎንዮሽ ጉዳት"),
    (re.compile(r"የጎንዮሽጉዳት"), "የጎንዮሽ ጉዳት"),
    (re.compile(r"አይደለምም"), "አይደለም"),
]

# Ge'ez numerals → Arabic. Applied only to isolated numeral runs.
_GEEZ = {
    "፩": "1",
    "፪": "2",
    "፫": "3",
    "፬": "4",
    "፭": "5",
    "፮": "6",
    "፯": "7",
    "፰": "8",
    "፱": "9",
    "፲": "10",
    "፳": "20",
    "፴": "30",
    "፻": "100",
}

_ABBREV = {
    "ኤችአይቪ": "ኤች.አይ.ቪ",
    "ኤች.አይ.ቪ": "ኤች.አይ.ቪ",
    "HIV": "HIV",
    "AIDS": "AIDS",
    "FP": "FP",
    "IUCD": "IUCD",
    "IUD": "IUD",
    "ANC": "ANC",
    "PNC": "PNC",
    "STI": "STI",
    "mg": "mg",
    "IU": "IU",
}

NEGATION_CUES = (
    "አይደለም",
    "የለም",
    "አይመከርም",
    "የተከለከለ",
    "ተቃራኒ",
    "አይገባም",
    "አትጠቀሙ",
    "አይጠቀሙ",
    "በጭራሽ",
    "እንዳይ",
    "መከልከል",
    "አይቻልም",
    "not",
    "never",
    "contraindicat",
    "do not",
    "must not",
)

_SPACE = re.compile(r"[ \t\r\f\v]+")
_ETH = re.compile(r"[\u1200-\u137F]")
_LAT = re.compile(r"[A-Za-z]")


@dataclass
class PreprocessResult:
    original: str
    normalized: str
    language: str
    has_negation: bool
    code_switched: bool
    numerals_mapped: bool
    notes: list[str] = field(default_factory=list)


def detect_language(text: str) -> str:
    eth = len(_ETH.findall(text))
    lat = len(_LAT.findall(text))
    if eth and lat:
        return "mixed"
    if eth:
        return "am"
    if lat:
        return "en"
    return "und"


def map_geez_numerals(text: str) -> tuple[str, bool]:
    changed = False
    out = []
    for ch in text:
        if ch in _GEEZ:
            out.append(_GEEZ[ch])
            changed = True
        else:
            out.append(ch)
    return "".join(out), changed


def expand_abbrev(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        tok = match.group(0)
        return _ABBREV.get(tok, _ABBREV.get(tok.upper(), tok))

    return re.sub(r"[A-Za-z.]{2,}|ኤች\.?አይ\.?ቪ", repl, text)


def has_negation(text: str) -> bool:
    low = text.lower()
    return any(cue.lower() in low for cue in NEGATION_CUES)


def preprocess(text: str, map_numerals: bool = True) -> PreprocessResult:
    original = text or ""
    notes: list[str] = []
    # NFC only. NFKC can merge characters we want to keep distinct.
    norm = unicodedata.normalize("NFC", original)
    norm = norm.translate(_PUNCT)
    norm = _SPACE.sub(" ", norm).strip()
    for rx, repl in _SPELLING:
        norm = rx.sub(repl, norm)
    mapped = False
    if map_numerals:
        norm, mapped = map_geez_numerals(norm)
        if mapped:
            notes.append("geez_numerals_mapped")
    norm = expand_abbrev(norm)
    lang = detect_language(norm)
    switched = lang == "mixed"
    if switched:
        notes.append("code_switched")
    neg = has_negation(norm)
    if neg:
        notes.append("negation_preserved")
    return PreprocessResult(
        original=original,
        normalized=norm,
        language=lang if lang != "und" else "am",
        has_negation=neg,
        code_switched=switched,
        numerals_mapped=mapped,
        notes=notes,
    )
