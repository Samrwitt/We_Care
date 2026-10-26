from we_care_mt.span import SpanTranslator
from we_care_mt.translation.google import GoogleTranslator


def test_google_backend_maps_nllb_codes(monkeypatch):
    backend = GoogleTranslator(pause_s=0, retries=1)
    monkeypatch.setattr(
        "we_care_mt.translation.google._google_web_translate",
        lambda text, source, target, timeout=30: f"{source}>{target}:{text}",
    )
    out = backend.translate_langs(["Give 10 mg."], src_lang="eng_Latn", tgt_lang="amh_Ethi")
    assert out == ["en>am:Give 10 mg."]


def test_span_with_google_like_backend_keeps_direction():
    class Adapter:
        def translate_langs(self, texts, *, src_lang, tgt_lang, **_kwargs):
            return [f"{tgt_lang}:{texts[0]}"]

    out = SpanTranslator(Adapter(), model_id="external:google-translate").translate(
        "Condoms prevent pregnancy.", "en-am"
    )
    assert out["direction"] == "en-am"
    assert "amh_Ethi:" in out["text"]
