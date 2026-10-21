from we_care_nlu.preprocess.normalize import has_negation, preprocess


def test_nfc_and_ethiopic_punctuation():
    r = preprocess("ሰላም።  እርግዝናን   ለመከላከል?")
    assert "  " not in r.normalized
    assert "።" not in r.normalized
    assert r.language in {"am", "mixed"}


def test_negation_not_stripped():
    r = preprocess("ይህ ዘዴ የተከለከለ ነው አይመከርም")
    assert r.has_negation
    assert "የተከለከለ" in r.normalized
    assert has_negation(r.normalized)


def test_code_switch_and_abbreviations():
    r = preprocess("HIV እና IUCD መረጃ እፈልጋለሁ")
    assert r.code_switched
    assert "HIV" in r.normalized
    assert "IUCD" in r.normalized


def test_geez_numerals():
    r = preprocess("፲ ሳምንት")
    assert r.numerals_mapped
    assert "10" in r.normalized


def test_spelling_variant_newe():
    r = preprocess("ይህ መርፌው ነዉ")
    assert "ነው" in r.normalized
    assert "መርፌ" in r.normalized
