from we_care_nlu.translation.compose import polish_amharic, shape_english_answer


def test_shape_english_answer_strips_toc_and_bullets():
    out = shape_english_answer(
        "§ Linkage is an important activity in prevention of STIs, including HIV/AIDS. "
        "Visits at FP clinics offer clients an opportunity for detection …… 19"
    )
    assert not out.startswith("§")
    assert "……" not in out
    assert "Linkage" in out
    assert len(out) < 360


def test_polish_amharic_expands_latin_abbreviations():
    out = polish_amharic("ኮንዶም STIs እና FP ላይ ይረዳል")
    assert "STIs" not in out
    assert "FP" not in out
    assert "የቤተሰብ ምጣኔ" in out
    assert out.endswith("።")
    assert "ኤች.አይ.ቪ" in polish_amharic("HIVን ጨምሮ")
    assert "የ የቤተሰብ" not in polish_amharic("የ የቤተሰብ ምጣኔ")
