"""Regression: diverse asks should land on curated / on-topic answers."""

from pathlib import Path

from we_care_nlu.config import load_config
from we_care_nlu.contracts.schemas import AnalyzeRequest
from we_care_nlu.pipeline import NLUEngine

USER_KB = Path(__file__).resolve().parents[1] / "data" / "kb" / "fp_user_kb.json"


def user_engine():
    eng = NLUEngine(load_config(), model_key="hash")
    eng.ingest_file(str(USER_KB), document_version="demo", default_review_status="unreviewed")
    return eng


CASES = [
    ("የእርግዝና መከላከያ ዘዴዎች ምንድን ናቸው?", "chunk_methods", ("ኮንዶም", "መርፌ")),
    ("ጡት እያጠባሁ መርፌ መጠቀም ይቻላል?", "chunk_bf_allowed", ("መርፌ", "ጡት")),
    ("ኮንዶም ምንድን ነው?", "chunk_condom_def", ("ኮንዶም",)),
    ("ክኒን ከተረሳ ምን ላድርግ?", "chunk_pill_missed", ("ክኒን",)),
    ("ኢምፕላንት ለምን ያህል ጊዜ ይቆያል?", "chunk_implant_def", ("ኢምፕላንት",)),
    ("የአስቸኳይ መከላከያ መቼ ይወሰዳል?", "chunk_ec_when", ("አስቸኳይ",)),
    ("ኮንዶም የት ይገኛል?", "chunk_condom_access", ("ኮንዶም",)),
    ("የመርፌ የጎንዮሽ ጉዳት ምንድን ነው?", "chunk_injectable_se", ("መርፌ", "ጎንዮሽ")),
    ("መርፌ እንዴት ይሠራል?", "chunk_injectable_how", ("መርፌ",)),
    ("ኤች.አይ.ቪ ያለብኝ የእርግዝና መከላከያ መጠቀም እችላለሁ?", "chunk_hiv_fp", ("ኤች.አይ.ቪ", "መከላከያ")),
    ("ቤተሰብ ምጣኔ አገልግሎት የት ይገኛል?", "chunk_where", ("ጤና",)),
    ("የቤተሰብ ምጣኔ አገልግሎት የት ይገኛል?", "chunk_where", ("ጤና",)),
]


def test_diverse_user_kb_replies_are_on_topic():
    eng = user_engine()
    for question, expected_id, must_have in CASES:
        out = eng.reply(AnalyzeRequest(text=question))
        assert out.evidence_id == expected_id, (question, out.evidence_id, (out.amharic_reply or "")[:120])
        body = out.amharic_reply or ""
        for token in must_have:
            assert token in body, (question, token, body)
        assert "ሊሊ" not in body or expected_id == "chunk_lam"
        assert "ታህሳስ" not in (out.document_title or "")


def test_cryptic_pregnancy_not_answered_by_implant_def():
    """Docs-grounded topic must beat curated FP defs that only share እርግዝና."""
    from we_care_nlu.retrieval.relevance import (
        answer_quality,
        distinctive_query_tokens,
        relevant_enough,
        should_force_include,
    )

    q = "ክሪፕቲክ እርግዝና ምንድን ነው?"
    assert distinctive_query_tokens(q) == ["ክሪፕቲክ"]

    implant = "ኢምፕላንት ከላይኛው ክንድ ቆዳ ሥር የሚገባ ትንሽ ዘንግ ነው። እንደ ዓይነቱ ለሦስት እስከ አምስት ዓመት እርግዝናን ይከላከላል።"
    cryptic = "ክሪፕቲክ እርግዝና በሌላ ስሙ ስውር እርግዝና፣ ሚስጥራዊ እርግዝና ወይም ያልታመነ እርግዝና በመባል የሚታወቅ ሲሆን፣ አንዲት ሴት እርግዝናዋን ሳታውቅ ትቆያለች።"

    class _C:
        def __init__(self, eid, text):
            self.evidence_id = eid
            self.amharic_text = text
            self.english_text = ""
            self.document_version = "demo" if eid.startswith("chunk_") else "source"

    assert should_force_include(query=q, chunk=_C("chunk_implant_def", implant), intent="definition", entities=[]) == 0
    assert relevant_enough(implant, query=q, intent="definition", entities=[]) is False
    assert relevant_enough(cryptic, query=q, intent="definition", entities=[]) is True
    aq_i = answer_quality(implant, intent="definition", query=q, entities=[], evidence_id="chunk_implant_def", fused=2.0)
    aq_c = answer_quality(cryptic, intent="definition", query=q, entities=[], evidence_id="ክሪፕቲክ እርግዝና -s0001", fused=1.0)
    assert aq_c > aq_i


def test_hiv_eligibility_does_not_demand_method_clarification():
    from we_care_nlu.contracts.schemas import ContextState

    eng = user_engine()
    # Prior catalogue turn should not force "which method?" on an HIV eligibility ask.
    ctx = ContextState(
        turn_index=1,
        last_intents=["family_planning_methods"],
        last_topic="family_planning_methods",
        last_entities=[],
    )
    out = eng.reply(
        AnalyzeRequest(text="ኤች.አይ.ቪ ያለብኝ መጠቀም እችላለሁ?", context=ctx)
    )
    assert out.requires_clarification is False
    assert out.evidence_id == "chunk_hiv_fp"
    assert "ኤች.አይ.ቪ" in (out.amharic_reply or "")
    assert "methotrexate" not in (out.amharic_reply or "").lower()


def test_hiv_then_injectable_followup_keeps_condition():
    from we_care_nlu.contracts.schemas import ContextState

    eng = user_engine()
    ctx = ContextState(
        turn_index=1,
        last_intents=["eligibility"],
        last_topic="eligibility",
        last_entities=[{"type": "condition", "value": "hiv", "surface": "ኤች.አይ.ቪ"}],
    )
    out = eng.reply(AnalyzeRequest(text="መርፌ", context=ctx))
    assert out.requires_clarification is False
    a = eng.analyze(AnalyzeRequest(text="መርፌ", context=ctx))
    assert any(e.type == "condition" and e.value == "hiv" for e in a.entities)
    assert a.intent == "eligibility"
    assert out.evidence_id == "chunk_hiv_fp"
    assert "ኤች.አይ.ቪ" in (out.amharic_reply or "")
    assert "methotrexate" not in (out.amharic_reply or "").lower()


def test_period_pain_not_answered_by_irregular_cycle_article():
    """Pain/what-to-do must not collapse to irregular-cycle definition via shared ወር አበባ."""
    from we_care_nlu.retrieval.relevance import (
        distinctive_query_tokens,
        relevant_enough,
        soft_topic_hit,
        soft_topic_groups,
    )

    q = "የወር አበባ ሲያመኝ ምን ላድርግ"
    assert distinctive_query_tokens(q) == ["ሲያመኝ", "ላድርግ"]
    irregular = (
        "ጤናማ የወር አበባ ዑደት የሚባለው በአማካኝ የሚከተሉትን መስፈርቶች ሲያሟላ ነው። "
        "ስለዚህ የጤና ባለሙያዎች ከላይ ከተጠቀሰው የጤናማ የወር አበባ ዑደት ውጪ የሚከሠተውን ዑደት "
        "“የወር አበባ ዑደት መዛባት” ይሉታል።"
    )
    pain = (
        "(NSAIDs) ስቴሮይድ ያልሆኑ ፀረ ብግነት መድኃኒቶች በኢንዶሜትሪዮሲስ ምክንያት የሚከሰተውን "
        "የወር አበባ ህመም ለማስታገስ ይረዳሉ።"
    )
    assert relevant_enough(irregular, query=q, intent="treatment", entities=[]) is False
    assert relevant_enough(pain, query=q, intent="treatment", entities=[]) is True
    assert soft_topic_hit(pain, soft_topic_groups(q), query=q) is True

    docs = Path(__file__).resolve().parents[1] / "data" / "kb" / "docs_amharic_kb.json"
    if not docs.exists():
        return
    eng = NLUEngine(load_config(), model_key="hash")
    eng.ingest_file(str(docs), document_version="docs-am", default_review_status="source", rebuild=False)
    eng.ingest_file(str(USER_KB), document_version="demo", default_review_status="unreviewed", rebuild=True)
    out = eng.reply(AnalyzeRequest(text=q))
    body = out.amharic_reply or ""
    assert "መዛባት" not in body or "ህመም" in body or "ማስታገስ" in body
    assert "አብዛኞቹ የእርግዝና መከላከያ" not in body
    assert any(x in body for x in ("ህመም", "ሕመም", "ማስታገስ", "መድኃኒት", "NSAID", "ቁርጠት"))


def test_irregular_menses_when_prefers_definition_not_dangling_intro():
    """'When is a cycle irregular?' must not answer with an empty 'healthy criteria are…' teaser."""
    from we_care_nlu.kb.docx_amharic import docx_to_alignment
    from we_care_nlu.retrieval.relevance import answer_quality, low_quality_penalty
    from we_care_nlu.translation.reply import focus_span_for_query

    dangling = "ጤናማ የወር አበባ ዑደት የሚባለው በአማካኝ የሚከተሉትን መስፈርቶች ሲያሟላ ነው።"
    punchline = (
        "ስለዚህ የጤና ባለሙያዎች ከላይ ከተጠቀሰው የጤናማ የወር አበባ ዑደት ውጪ የሚከሠተውን ዑደት "
        "“የወር አበባ ዑደት መዛባት” ይሉታል።"
    )
    criteria = (
        "ጤናማ የወር አበባ ዑደት የሚባለው በአማካኝ የሚከተሉትን መስፈርቶች ሲያሟላ ነው። "
        "የሚመጣው በ21-35 ቀናት የሚቆየው ከ 2 – 7 ቀናት መጠኑ ከ 80 ሚ.ሊ. ያልበለጠ። "
        "እነዚህ የጤናማ የወር አበባ ዑደት መገለጫዎች ናቸው።"
    )
    q = "የወር አበባ ዑደት ተዛባ የሚባለው መቼ ነው?"
    assert low_quality_penalty(dangling) < low_quality_penalty(criteria)
    aq_d = answer_quality(dangling, intent="definition", query=q, entities=[], evidence_id="a", fused=2.0)
    aq_p = answer_quality(punchline, intent="definition", query=q, entities=[], evidence_id="b", fused=1.5)
    aq_c = answer_quality(criteria, intent="definition", query=q, entities=[], evidence_id="c", fused=1.5)
    assert aq_p > aq_d
    assert aq_c > aq_d

    merged = criteria + " " + punchline
    shaped = focus_span_for_query(merged, q)
    assert shaped.index("መዛባት") < shaped.index("ጤናማ የወር አበባ ዑደት የሚባለው")

    blog = Path(__file__).resolve().parents[2] / "Docs" / "blogs" / "_የወር አበባ ዑደት ተዛባ የሚባለው መቼ ነው_.docx"
    if blog.exists():
        doc = docx_to_alignment(blog, flags="wecare_article,source_amharic", kb_kind="blog_amharic")
        joined = " ".join(s.get("translated_text") or "" for s in doc.get("segments") or [])
        assert "21-35" in joined or "21–35" in joined or "21" in joined
        assert "መዛባት" in joined


def test_hiv_symptoms_then_treatment_not_fp_eligibility():
    """Follow-up 'እንዴት ይታከማል?' after HIV symptoms must not return contraception eligibility."""
    from we_care_nlu.contracts.schemas import ContextState

    docs = Path(__file__).resolve().parents[1] / "data" / "kb" / "docs_amharic_kb.json"
    eng = NLUEngine(load_config(), model_key="hash")
    if docs.exists():
        eng.ingest_file(str(docs), document_version="docs-am", default_review_status="source", rebuild=False)
    eng.ingest_file(str(USER_KB), document_version="demo", default_review_status="unreviewed", rebuild=True)

    first = eng.analyze(AnalyzeRequest(text="ኤች.አይ.ቪ ምልክቶች"))
    ctx = ContextState(
        turn_index=1,
        last_intents=[first.intent],
        last_topic=first.topic or "hiv",
        last_entities=[
            {"type": e.type, "value": e.value, "surface": e.surface} for e in first.entities
        ],
    )
    out = eng.reply(AnalyzeRequest(text="እንዴት ይታከማል?", context=ctx))
    body = (out.amharic_reply or "").lower()
    assert out.evidence_id != "chunk_hiv_fp"
    assert "አብዛኞቹ የእርግዝና መከላከያ" not in (out.amharic_reply or "")
    assert out.requires_clarification is False
    # Prefer ART / HIV medication content when Docs KB is present.
    if docs.exists():
        assert any(
            tok in body
            for tok in ("ኤ.አር.ቲ", "art", "antiretroviral", "ሕክምና", "መድኃኒት", "መድሃኒት", "ቫይረስ")
        )
        assert "ቂጥኝ" not in body or "ኤ.አር.ቲ" in body or "art" in body
