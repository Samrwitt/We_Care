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
