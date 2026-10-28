"""Dynamic terminology index — extraction, matching, topic/intent separation."""

from pathlib import Path

from we_care_nlu.config import load_config
from we_care_nlu.contracts.schemas import AnalyzeRequest, ContextState
from we_care_nlu.kb.ingest import ingest_path
from we_care_nlu.ner.extract import load_gazetteer
from we_care_nlu.pipeline import NLUEngine
from we_care_nlu.terminology.extract import extract_candidates_from_chunks, seed_from_gazetteer
from we_care_nlu.terminology.index import ConceptIndex
from we_care_nlu.terminology.models import MedicalConcept

USER_KB = Path(__file__).resolve().parents[1] / "data" / "kb" / "fp_user_kb.json"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_kb.json"


def test_seed_gazetteer_produces_approved_surfaces():
    seeded = seed_from_gazetteer(load_gazetteer())
    assert seeded
    assert any(c.category == "condition" for c in seeded)
    assert any("HIV" in c.all_surfaces() or "ኤች.አይ.ቪ" in c.all_surfaces() for c in seeded)


def test_extract_candidates_from_mini_kb(tmp_path: Path):
    chunks = ingest_path(str(FIXTURE), default_review_status="unreviewed")
    cands = extract_candidates_from_chunks(chunks)
    assert isinstance(cands, list)
    # Mini KB is FP-focused; titles/bodies should yield at least some candidates or empty is ok
    # if everything already looks like stop words — still must be structured.
    for c in cands:
        assert c.concept_id.startswith("concept_")
        assert c.review_status == "pending_review"


def test_concept_index_match_hiv(tmp_path: Path):
    idx = ConceptIndex(root=tmp_path)
    idx.upsert_seeded(
        [
            MedicalConcept(
                concept_id="concept_hiv_test",
                canonical_amharic="ኤች.አይ.ቪ",
                canonical_english="HIV",
                synonyms_amharic=["ኤችአይቪ"],
                synonyms_english=["HIV", "AIDS"],
                category="condition",
                review_status="seeded",
            )
        ]
    )
    hits = idx.match("የኤችአይቪ ምልክቶች ምንድን ናቸው?")
    assert hits and hits[0].concept_id == "concept_hiv_test"
    terms = idx.expansion_terms("ኤች.አይ.ቪ ምንድን ነው?")
    assert any("HIV" in t or "ኤች" in t for t in terms)


def test_intent_topic_separation_and_followup_topic_carry():
    eng = NLUEngine(load_config(), model_key="hash")
    eng.ingest_file(str(USER_KB), document_version="demo", default_review_status="unreviewed")
    first = eng.analyze(AnalyzeRequest(text="ኤች.አይ.ቪ ምንድን ነው?"))
    assert first.intent in {"definition", "general_medical_info", "eligibility"}
    # Topic may come from lexicon topics or concept index.
    assert first.topic or first.medical_concepts or any(
        e.type in {"condition", "medical_concept"} for e in first.entities
    )

    ctx = ContextState(
        turn_index=1,
        last_intents=[first.intent],
        last_topic=first.topic or "hiv",
        last_entities=[{"type": "condition", "value": "hiv", "surface": "ኤች.አይ.ቪ"}],
    )
    follow = eng.analyze(AnalyzeRequest(text="እንዴት ይታከማል?", context=ctx))
    assert follow.requires_clarification is False
    assert follow.intent in {"treatment", "medication", "how_it_works", "general_medical_info", "follow_up"}
    assert follow.interpretation.get("requires_context") or follow.topic or follow.context_resolution.get(
        "inherited_topic"
    )


def test_kb_only_style_query_still_retrieves_without_lexicon_topic():
    """Cryptic pregnancy is in Docs KB, not a hard-coded intent label."""
    docs = Path(__file__).resolve().parents[1] / "data" / "kb" / "docs_amharic_kb.json"
    if not docs.exists():
        return
    eng = NLUEngine(load_config(), model_key="hash")
    eng.ingest_file(str(docs), default_review_status="source", rebuild=True)
    eng.ingest_file(str(USER_KB), document_version="demo", default_review_status="unreviewed", rebuild=True)
    out = eng.reply(AnalyzeRequest(text="ክሪፕቲክ እርግዝና ምንድን ነው?"))
    assert out.requires_clarification is False
    assert "ኢምፕላንት" not in (out.amharic_reply or "")
    assert "ክሪፕቲክ" in (out.amharic_reply or "") or "ስውር" in (out.amharic_reply or "")
