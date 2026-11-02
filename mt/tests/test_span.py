import threading

from we_care_mt.glossary.manager import Glossary, GlossaryEntry
from we_care_mt.span import SpanTranslator
from we_care_mt.translation.cache import TranslationCache


class Echo:
    def translate_langs(self, texts, *, src_lang, tgt_lang, **_kwargs):
        return [f"{src_lang}>{tgt_lang}:{texts[0]}"]


def test_english_span_keeps_the_dose():
    out = SpanTranslator(Echo()).translate("Give 10 mg today.", "en-am")
    assert out["direction"] == "en-am"
    assert "10 mg" in out["text"]
    assert "WCMT" not in out["text"]
    assert out["text"].startswith("eng_Latn>amh_Ethi:")


def test_repeat_sentence_is_cached(tmp_path):
    calls = {"n": 0}

    class Once:
        def translate_langs(self, texts, *, src_lang, tgt_lang, **_kwargs):
            calls["n"] += 1
            return [f"{tgt_lang}:{texts[0]}"]

    cache = TranslationCache(tmp_path / "span.sqlite")
    service = SpanTranslator(Once(), cache=cache, model_id="facebook/nllb-200-distilled-600M", num_beams=1)
    first = service.translate("Condoms are available.", "en-am")
    second = service.translate("Condoms are available.", "en-am")
    assert first["text"] == second["text"]
    assert calls["n"] == 1
    assert second["qa_notes"] == ["cache_hit"]
    err = []

    def other_thread():
        try:
            service.translate("Condoms are available.", "en-am")
        except Exception as exc:
            err.append(exc)

    worker = threading.Thread(target=other_thread)
    worker.start()
    worker.join()
    assert err == []


def test_missing_verified_term_is_flagged():
    glossary = Glossary([
        GlossaryEntry("contraceptive", "የእርግዝና መከላከያ", "family_planning", "verified"),
    ])

    class Invented:
        def translate_langs(self, texts, *, src_lang, tgt_lang, **_kwargs):
            return ["የፀረ-ጉሊፈ ዘዴዎች"]

    out = SpanTranslator(Invented(), glossary).translate("contraceptive methods", "en-am")
    assert any(note.startswith("term_missing:contraceptive") for note in out["qa_notes"])


def test_question_goes_to_english_without_glossary_placeholders():
    out = SpanTranslator(Echo()).translate("የእርግዝና መከላከያ", "am-en")
    assert out["text"] == "amh_Ethi>eng_Latn:የእርግዝና መከላከያ"
    assert out["qa_notes"] == []
