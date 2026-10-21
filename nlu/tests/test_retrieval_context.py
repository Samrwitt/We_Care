from pathlib import Path

from we_care_nlu.config import load_config
from we_care_nlu.contracts.schemas import AnalyzeRequest, ContextState
from we_care_nlu.pipeline import NLUEngine

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_kb.json"
USER_KB = Path(__file__).resolve().parents[1] / "data" / "kb" / "fp_user_kb.json"


def engine():
    eng = NLUEngine(load_config(), model_key="hash")
    eng.ingest_file(str(FIXTURE), default_review_status="unreviewed")
    return eng


def test_paraphrase_retrieval_finds_methods_chunk():
    eng = engine()
    for q in [
        "የእርግዝና መከላከያ ዘዴዎች ምንድን ናቸው?",
        "እርግዝናን ለመከላከል ምን አማራጮች አሉ?",
    ]:
        hits = eng.retriever.search(q, top_k=5)
        ids = [h.evidence_id for h in hits]
        assert "chunk_methods" in ids


def test_breastfeeding_polarity_not_collapsed():
    eng = engine()
    allow = eng.retriever.search("ጡት እያጠባሁ መርፌ መጠቀም ይቻላል?", top_k=5)
    contra = eng.retriever.search("ጡት በማጥባት ጊዜ የትኛው ዘዴ የተከለከለ ነው?", top_k=5)
    assert allow and contra
    # Contraindication query should not rank the "can be used" chunk above the contra chunk.
    contra_ids = [h.evidence_id for h in contra]
    assert contra_ids[0] != "chunk_bf_allowed"
    assert "chunk_bf_contra" in contra_ids


def test_followup_resolves_single_prior_method():
    eng = engine()
    ctx = ContextState(
        turn_index=1,
        last_method="injectable_contraceptive",
        last_entities=[{"type": "contraceptive_method", "value": "injectable_contraceptive", "surface": "መርፌ"}],
    )
    out = eng.analyze(AnalyzeRequest(text="የጎንዮሽ ጉዳቱስ?", context=ctx))
    assert out.context_resolution["resolved_from_previous_turn"] is True
    assert any(e.value == "injectable_contraceptive" for e in out.entities)
    assert out.requires_clarification is False


def test_followup_ambiguous_does_not_guess():
    eng = engine()
    ctx = ContextState(
        last_entities=[
            {"type": "contraceptive_method", "value": "injectable_contraceptive"},
            {"type": "contraceptive_method", "value": "implant"},
        ]
    )
    out = eng.analyze(AnalyzeRequest(text="የጎንዮሽ ጉዳቱስ?", context=ctx))
    assert out.requires_clarification is True
    assert out.context_resolution["assumed_missing_slot"] is False


def test_unreviewed_translation_is_flagged():
    eng = engine()
    out = eng.analyze(AnalyzeRequest(text="የእርግዝና መከላከያ ዘዴዎች ምንድን ናቸው?"))
    notes = " ".join(out.safety.get("review_notes") or [])
    assert "unreviewed" in notes.lower() or "not clinically verified" in notes.lower()
    assert out.generative_answer is None


def test_emergency_rule_overrides():
    eng = engine()
    out = eng.analyze(AnalyzeRequest(text="ከባድ መድማት አለብኝ አስቸኳይ ነው"))
    assert out.intent == "emergency"
    assert out.safety["emergency"]["source"] == "rule_based"


def user_engine():
    eng = NLUEngine(load_config(), model_key="hash")
    eng.ingest_file(str(USER_KB), default_review_status="unreviewed")
    return eng


def test_user_questions_retrieve_matching_method():
    eng = user_engine()
    pairs = [
        ("ኮንዶም ምንድን ነው?", "chunk_condom_def"),
        ("ክኒን ከተረሳ ምን ላድርግ?", "chunk_pill_missed"),
        ("ኢምፕላንት ለምን ያህል ጊዜ ይቆያል?", "chunk_implant_def"),
        ("የአስቸኳይ መከላከያ መቼ ይወሰዳል?", "chunk_ec_when"),
        ("ኮንዶም የት ይገኛል?", "chunk_condom_access"),
    ]
    for question, expected in pairs:
        out = eng.analyze(AnalyzeRequest(text=question))
        ids = [h.evidence_id for h in out.retrieval_results[:4]]
        assert expected in ids, (question, ids)


def test_followup_query_uses_prior_method_for_retrieval():
    eng = user_engine()
    ctx = ContextState(
        turn_index=1,
        last_method="injectable_contraceptive",
        last_entities=[{"type": "contraceptive_method", "value": "injectable_contraceptive", "surface": "መርፌ"}],
    )
    out = eng.analyze(AnalyzeRequest(text="የጎንዮሽ ጉዳቱስ?", context=ctx))
    ids = [h.evidence_id for h in out.retrieval_results[:3]]
    assert "chunk_injectable_se" in ids
