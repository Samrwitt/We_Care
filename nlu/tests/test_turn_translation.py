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


def test_reply_uses_english_span_without_neural_translation():
    eng = NLUEngine(model_key="hash")
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
    out = eng.reply(AnalyzeRequest(text="የእርግዝና መከላከያ ዘዴዎች ምንድን ናቸው?"))
    assert "contraceptive" in out.english_query
    assert out.translation_status == "source_only"
    assert out.evidence_id == "en-methods"
    assert out.page == 4
    assert "አልተተረጎመም" in out.amharic_reply
    assert "Contraceptive method options" in out.english_reply
    assert out.generative_answer is None


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
    out = eng.reply(AnalyzeRequest(text="የእርግዝና መከላከያ ዘዴዎች ምንድን ናቸው?"))
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
    out = eng.reply(AnalyzeRequest(text="የእርግዝና መከላከያ ዘዴዎች ምንድን ናቸው?"))
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
    out = eng.reply(AnalyzeRequest(text="መርፌ ጡት ማጥባት"))
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
