from we_care_nlu.contracts.schemas import AnalyzeRequest, ChunkRecord
from we_care_nlu.pipeline import NLUEngine
from we_care_nlu.safety.rules import emergency_flags
from we_care_nlu.translation.mt_client import TranslatorNotReady
from we_care_nlu.translation.phrase_map import map_to_english
from we_care_nlu.translation.reply import reply_qa


class LoadingTranslator:
    can_translate_reply = True
    backend = "mt"

    def to_english(self, text: str) -> str:
        raise TranslatorNotReady("loading")

    def to_amharic(self, text: str) -> str:
        raise TranslatorNotReady("loading")


class DroppingTranslator:
    can_translate_reply = True
    backend = "fake"

    def to_english(self, text: str) -> str:
        return "injectable 10 mg must not be used while breastfeeding"

    def to_amharic(self, text: str) -> str:
        return "ጡት በማጥባት ጊዜ መርፌ መጠቀም ይቻላል"


def test_phrase_map_keeps_method_and_polarity():
    english = map_to_english("የእርግዝና መከላከያ ዘዴዎች ምንድን ናቸው?")
    assert "contraceptive" in english
    assert "እርግዝና" not in english
    contra = map_to_english("ጡት በማጥባት ጊዜ መርፌ መጠቀም አይቻልም")
    assert "breastfeeding" in contra
    assert "injectable" in contra
    assert "must not" in contra


def test_emergency_contraception_is_not_an_emergency():
    assert emergency_flags("የአስቸኳይ መከላከያ ምንድን ነው?")["is_emergency"] is False
    assert emergency_flags("what is emergency contraception")["is_emergency"] is False
    assert emergency_flags("ከባድ መድማት አለብኝ")["is_emergency"] is True


def test_semantic_hit_is_kept_when_words_do_not_overlap():
    from we_care_nlu.contracts.schemas import RetrievalHit
    from we_care_nlu.translation.reply import select_span

    hit = RetrievalHit(
        evidence_id="implant-years",
        document_id="mec",
        document_title="MEC",
        english_text="The implant remains effective for three years after insertion.",
        fused_score=0.8,
        dense_score=0.82,
        page=12,
    )
    span, picked = select_span("how long does the implant last", [hit], "duration")
    assert picked is hit
    assert "three years" in span


def test_select_span_prefers_answer_cues_over_weak_keyword_overlap():
    from we_care_nlu.contracts.schemas import RetrievalHit
    from we_care_nlu.translation.reply import select_span

    weak = RetrievalHit(
        evidence_id="implant-mention",
        document_id="mec",
        document_title="MEC",
        english_text="Women who want an implant should talk with a provider about insertion.",
        fused_score=0.7,
        dense_score=0.5,
        page=3,
    )
    strong = RetrievalHit(
        evidence_id="implant-years",
        document_id="mec",
        document_title="MEC",
        english_text="The implant remains effective for three years after insertion.",
        fused_score=0.75,
        dense_score=0.8,
        page=12,
    )
    span, picked = select_span("how long does the implant last", [weak, strong], "duration")
    assert picked is strong
    assert "three years" in span


def test_reply_qa_flags_missing_glossary_term():
    notes = reply_qa(
        "Emergency contraception can prevent pregnancy.",
        "ይህ እርግዝናን መከላከል ይችላል።",
        ["term_missing:Emergency contraception"],
    )
    assert any(note.startswith("term_missing") for note in notes)


def test_missing_guideline_uses_intent_advice_and_referral():
    eng = NLUEngine(model_key="hash")
    out = eng.reply(AnalyzeRequest(text="የብልት ቁስለት ምንድን ነው?"))
    assert out.translation_status == "template"
    assert out.intent == "genital_ulcer"
    assert "9394" in out.amharic_reply
    assert "https://wecare.et/articles/135" in out.amharic_reply
    assert "https://wecare.et/booking" in out.amharic_reply
    english = eng.reply(AnalyzeRequest(text="What is a genital ulcer?"))
    assert english.intent == "genital_ulcer"
    assert "9394" in english.english_reply
    assert "https://wecare.et/articles/135" in english.english_reply
    discharge = eng.reply(AnalyzeRequest(text="I have unusual vaginal discharge"))
    assert discharge.intent == "vaginal_discharge"
    assert "https://wecare.et/articles/420" in discharge.english_reply
    hiv = eng.reply(AnalyzeRequest(text="ስለ ኤች.አይ.ቪ ማወቅ እፈልጋለሁ"))
    assert hiv.intent == "hiv"
    assert "https://wecare.et/articles/410" in hiv.amharic_reply


def test_reply_uses_english_span_without_neural_translation():
    from we_care_nlu.translation.phrase_map import PhraseMapTranslator

    eng = NLUEngine(model_key="hash")
    eng.translator = PhraseMapTranslator()
    eng.ingest_alignment_dict(
        {
            "document_name": "Guideline.pdf",
            "title": "Family Planning",
            "segments": [
                {
                    "segment_id": "en-methods",
                    "kind": "paragraph",
                    "page_number": 4,
                    "section": "Methods",
                    "source_text": "Contraceptive method options include condoms, pills, injectables, and implants.",
                    "translated_text": "",
                    "flags": "source_english",
                }
            ],
        }
    )
    # English ask against English-only guideline text (Amharic asks use the Amharic article KB).
    out = eng.reply(AnalyzeRequest(text="What contraceptive method options are available?"))
    assert "contraceptive" in out.english_query.lower()
    assert out.translation_status == "source_only"
    assert out.evidence_id == "en-methods"
    assert out.page == 4
    assert "አልተተረጎመም" in out.amharic_reply
    assert "Contraceptive method options" in out.english_reply
    assert out.generative_answer is None


def test_reply_prefers_document_amharic_over_live_mt():
    eng = NLUEngine(model_key="hash")

    class Counting:
        can_translate_reply = True
        backend = "mt"

        def to_english(self, text: str) -> str:
            return text

        def to_amharic(self, text: str) -> str:
            raise AssertionError("live En→Am should not run when KB has Amharic")

    eng.translator = Counting()
    eng.ingest_alignment_dict(
        {
            "document_name": "Guideline.pdf",
            "title": "Family Planning",
            "segments": [
                {
                    "segment_id": "bi-methods",
                    "kind": "paragraph",
                    "page_number": 4,
                    "section": "Methods",
                    "source_text": "Contraceptive method options include condoms, pills, injectables, and implants.",
                    "translated_text": "የእርግዝና መከላከያ ዘዴዎች ኮንዶም፣ ክኒን፣ መርፌ እና ኢምፕላንትን ያካትታሉ።",
                }
            ],
        }
    )
    out = eng.reply(AnalyzeRequest(text="የእርግዝና መከላከያ ዘዴዎች ምንድን ናቸው?"))
    assert out.translation_status == "translated"
    assert out.translation_backend == "composed"
    assert "ኮንዶም" in out.amharic_reply
    assert out.evidence_id == "bi-methods"
    assert "composed_reply" in out.qa_notes


def test_amharic_question_returns_document_amharic_without_full_phrase_map():
    """Uncovered Amharic asks should still retrieve bilingual chunks and answer in Amharic."""
    eng = NLUEngine(model_key="hash")
    eng.ingest_alignment_dict(
        {
            "document_name": "Guideline.pdf",
            "title": "Family Planning",
            "segments": [
                {
                    "segment_id": "bi-gov",
                    "kind": "paragraph",
                    "page_number": 4,
                    "section": "Background",
                    "source_text": (
                        "The Government of Ethiopia believes that family planning is one of the "
                        "key strategies to improving maternal health."
                    ),
                    "translated_text": (
                        "የኢትዮጵያ መንግስት የቤተሰብ ምጣኔ የእናቶች ጤና ለማሻሻል አንዱ ቁልፍ ስትራቴጂ ነው ብሎ ያምናል።"
                    ),
                }
            ],
        }
    )
    out = eng.reply(AnalyzeRequest(text="ኢትዮጵያ የቤተሰብ ምጣኔን ለምን ትደግፋለች?"))
    assert out.translation_status == "translated"
    assert out.translation_backend == "composed"
    assert "የቤተሰብ ምጣኔ" in out.amharic_reply
    assert out.evidence_id == "bi-gov"


def test_compose_shapes_short_answer_not_raw_pdf_dump():
    from we_care_nlu.translation.compose import compose_amharic_reply, shape_english_answer

    class FakeMT:
        can_translate_reply = True
        backend = "mt"

        def to_amharic(self, text: str) -> str:
            assert "……" not in text
            assert not text.startswith("§")
            return "ኮንዶም የወሲብ በሽታዎችን ለመከላከል ይረዳል።"

    shaped = shape_english_answer(
        "§ Promote the use of condoms (male and female) for clients who are at risk of "
        "acquiring STIs, even if they are using other methods of FP …… 19"
    )
    assert "……" not in shaped
    assert shaped.startswith("Promote") or shaped.startswith("promote") or "condom" in shaped.lower()
    am, backend, notes = compose_amharic_reply(
        english_span=shaped,
        document_amharic="§ ኮንዶሞች … STIs … FP ……",
        translator=FakeMT(),
    )
    assert backend == "composed"
    assert "mt_rendered" in notes
    assert "STIs" not in am
    assert "ኮንዶም" in am


def test_known_question_skips_the_model_until_the_sentence():
    eng = NLUEngine(model_key="hash")

    class Counting:
        can_translate_reply = True
        backend = "mt"

        def __init__(self):
            self.english_calls = 0
            self.amharic_calls = 0

        def to_english(self, text: str) -> str:
            self.english_calls += 1
            return text

        def to_amharic(self, text: str) -> str:
            self.amharic_calls += 1
            return "ኮንዶም፣ ክኒን፣ መርፌ እና ኢምፕላንት አሉ።"

    counter = Counting()
    eng.translator = counter
    eng.ingest_alignment_dict(
        {
            "document_name": "Guideline.pdf",
            "title": "Family Planning",
            "segments": [
                {
                    "segment_id": "en-methods",
                    "kind": "paragraph",
                    "page_number": 4,
                    "section": "Methods",
                    "source_text": "Contraceptive method options include condoms, pills, injectables, and implants.",
                    "translated_text": "",
                    "flags": "source_english",
                }
            ],
        }
    )
    out = eng.reply(AnalyzeRequest(text="What contraceptive method options are available?"))
    assert counter.english_calls == 0
    assert counter.amharic_calls == 1
    assert out.translation_status == "translated"
    assert "ኮንዶም" in out.amharic_reply


def test_loading_translator_still_returns_the_english_sentence():
    eng = NLUEngine(model_key="hash")
    eng.translator = LoadingTranslator()
    eng.ingest_alignment_dict(
        {
            "document_name": "Guideline.pdf",
            "title": "Family Planning",
            "segments": [
                {
                    "segment_id": "en-methods",
                    "kind": "paragraph",
                    "page_number": 4,
                    "section": "Methods",
                    "source_text": "Contraceptive method options include condoms, pills, injectables, and implants.",
                    "translated_text": "",
                    "flags": "source_english",
                }
            ],
        }
    )
    out = eng.reply(AnalyzeRequest(text="What contraceptive method options are available?"))
    assert out.translation_status == "source_only"
    assert "እየተጫነ" in out.amharic_reply
    assert "Contraceptive method options" in out.english_reply
    assert out.page == 4


def test_bad_reply_translation_is_not_shown():
    eng = NLUEngine(model_key="hash")
    eng.translator = DroppingTranslator()
    eng.ingest_alignment_dict(
        {
            "document_name": "Dose.pdf",
            "title": "Dose",
            "segments": [
                {
                    "segment_id": "dose-1",
                    "kind": "paragraph",
                    "page_number": 2,
                    "source_text": "Injectable 10 mg must not be used while breastfeeding.",
                    "translated_text": "",
                    "flags": "source_english",
                }
            ],
        }
    )
    out = eng.reply(AnalyzeRequest(text="Can injectable be used while breastfeeding?"))
    assert out.translation_status == "qa_failed"
    assert out.qa_ok is False
    assert any(note.startswith("numbers_missing") or note.startswith("polarity") for note in out.qa_notes)
    assert "አልታየም" in out.amharic_reply
    assert "10 mg" in out.english_reply


def test_reply_qa_flags_lost_number():
    notes = reply_qa("Give 10 mg.", "መድኃኒት ይስጡ")
    assert notes and notes[0].startswith("numbers_missing")


def test_chunk_record_marks_source_english():
    record = ChunkRecord(
        evidence_id="x",
        document_id="d",
        document_title="t",
        section="",
        page=1,
        english_text="Condoms are available at health facilities.",
        amharic_text="",
        review_status="source",
        medical_subject="family_planning",
        document_version="source",
        source_kind="english_guideline",
    )
    assert record.review_status == "source"
    assert record.amharic_text == ""


def test_ingest_wecare_article_kb():
    from pathlib import Path

    from we_care_nlu.kb.ingest import ingest_path

    path = Path(__file__).resolve().parents[1] / "data" / "kb" / "wecare_articles_am.json"
    chunks = ingest_path(str(path), document_version="wecare-articles", default_review_status="source")
    assert len(chunks) >= 50
    assert all(c.source_kind == "wecare_article" for c in chunks)
    assert all(c.amharic_text and not c.english_text for c in chunks)
    assert any(c.source_url.startswith("https://wecare.et/") for c in chunks)
    assert any("IUD" in c.amharic_text or "አይ.ዩ.ዲ" in c.amharic_text for c in chunks)


def test_amharic_ask_returns_article_span_without_mt():
    from pathlib import Path

    eng = NLUEngine(model_key="hash")

    class Boom:
        can_translate_reply = True
        backend = "mt"

        def to_english(self, text: str) -> str:
            raise AssertionError("Amharic article replies must not call MT")

        def to_amharic(self, text: str) -> str:
            raise AssertionError("Amharic article replies must not call MT")

    eng.translator = Boom()
    path = Path(__file__).resolve().parents[1] / "data" / "kb" / "wecare_articles_am.json"
    eng.ingest_file(str(path), document_version="wecare-articles", default_review_status="source")
    out = eng.reply(AnalyzeRequest(text="አይ.ዩ.ዲ ምንድን ነው?"))
    assert out.translation_status == "translated"
    assert out.translation_backend == "article"
    assert "አይ.ዩ.ዲ" in out.amharic_reply or "IUD" in out.amharic_reply
    assert "wecare.et" in out.amharic_reply or out.source_url.startswith("https://wecare.et/")
    assert out.english_reply == ""
    assert out.evidence_id


def test_english_ask_translates_article_span():
    from pathlib import Path

    eng = NLUEngine(model_key="hash")

    class FakeMT:
        can_translate_reply = True
        backend = "mt"
        last_qa_notes: list[str] = []

        def to_english(self, text: str) -> str:
            assert any("\u1200" <= ch <= "\u137F" for ch in text)
            return "An IUD is a small T-shaped contraceptive device placed in the uterus."

        def to_amharic(self, text: str) -> str:
            raise AssertionError("English ask should translate Am→En, not En→Am")

    eng.translator = FakeMT()
    path = Path(__file__).resolve().parents[1] / "data" / "kb" / "wecare_articles_am.json"
    eng.ingest_file(str(path), document_version="wecare-articles", default_review_status="source")
    out = eng.reply(AnalyzeRequest(text="What is an IUD?"))
    assert out.translation_status == "translated"
    assert out.translation_backend == "mt"
    assert "IUD" in out.english_reply
    assert "wecare.et" in out.english_reply or out.source_url.startswith("https://wecare.et/")
    assert any("\u1200" <= ch <= "\u137F" for ch in out.amharic_reply)
