"""Numbered Amharic lists must stay intact and drop empty markers."""

from we_care_nlu.kb.sentences import split_sentences
from we_care_nlu.translation.reply import drop_incomplete_list_items, focus_span_for_query


_HORMONE_LIST = (
    "ሐኪም ዘንድ ሄደው ትክክለኛውን መድሃኒት ከመጀመርዎ ጎን ለጎን የሚከተሉትን ማድረግ ሆርሞንን ለማስተካከል ይረዳል፦ "
    "1. ዝቅተኛ ካርቦሃይድሬት (Low Carb) አመጋገብ፦ እንደ ስኳር፣ ነጭ ዳቦ እና ፓስታ ያሉ ምግቦችን በመቀነስ "
    "በፋይበር እና በፕሮቲን የበለፀጉ ምግቦችን መመገብ የኢንሱሊን መጠንን ያስተካክላል። "
    "2. የአካል ብቃት እንቅስቃሴ በመደበኛነት መስራት፦ በሳምንት ቢያንስ 3-4 ቀን የሰውነት እንቅስቃሴ ማድረግ "
    "ክብደትን ለመቀነስ ብቻ ሳይሆን የሆርሞን መዛባትን ለማስተካከል ይረዳል። "
    "3. እንቅልፍ እና ጭንቀትን መቆጣጠር፦ ጭንቀት የኮርቲሶል ሆርሞንን ስለሚጨምር በቂ እንቅልፍ ማግኘት እና መዝናናት ይመከራል። "
    "4."
)


def test_split_sentences_keeps_numbered_list_markers_with_content():
    sents = split_sentences(_HORMONE_LIST)
    joined = " ".join(sents)
    assert "1. ዝቅተኛ" in joined
    assert "2. የአካል" in joined
    assert "3. እንቅልፍ" in joined
    # Contentful markers must stay attached (empty trailing "4." may be alone).
    assert not any(s.strip() in {"1.", "2.", "3."} for s in sents)


def test_drop_incomplete_list_items_removes_empty_slots():
    assert drop_incomplete_list_items(_HORMONE_LIST).endswith("ይመከራል።")
    assert not drop_incomplete_list_items(_HORMONE_LIST).rstrip().endswith("4.")
    scrambled = (
        "ይረዳል፦ 1. የአካል ብቃት እንቅስቃሴ በመደበኛነት መስራት፦ ይረዳል። "
        "ዝቅተኛ ካርቦሃይድሬት አመጋገብ፦ ያስተካክላል። 2. 3. እንቅልፍ መቆጣጠር፦ ይመከራል። 4."
    )
    cleaned = drop_incomplete_list_items(scrambled)
    assert "2. 3." not in cleaned
    assert not cleaned.rstrip().endswith("4.")
    assert "3. እንቅልፍ" in cleaned


def test_focus_span_does_not_reorder_numbered_lists():
    shaped = focus_span_for_query(_HORMONE_LIST, "ሲያመኝ ምን ላድርግ?")
    assert shaped.index("1. ዝቅተኛ") < shaped.index("2. የአካል") < shaped.index("3. እንቅልፍ")
    assert not shaped.rstrip().endswith("4.")
    assert "2. 3." not in shaped
