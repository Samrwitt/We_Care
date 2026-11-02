from we_care_mt.segmentation.splitter import pack_sentences, split_sentences


def test_does_not_split_on_abbreviations():
    text = "Dr. Meseret Zelalem (MD) noted that Fig. 1 shows CPR at 41%. Next sentence starts here."
    sents = split_sentences(text)
    assert any("Dr. Meseret" in s for s in sents)
    assert any(s.startswith("Next sentence") for s in sents)


def test_does_not_split_decimals_or_ratios():
    text = "The MMR is 401/100,000 live births and CPR is 41.5 percent. Another point follows."
    sents = split_sentences(text)
    assert any("401/100,000" in s for s in sents)
    assert any("Another point" in s for s in sents)


def test_pack_respects_max_chars_without_mid_sentence_when_possible():
    sents = [
        "Family planning improves maternal health.",
        "Condoms prevent STI and HIV infection.",
    ]
    packed = pack_sentences(sents, max_chars=80)
    assert packed[0].endswith("health.") or "Family planning" in packed[0]
    assert all(len(p) <= 80 for p in packed)
