from we_care_mt.translation.engine import TranslationEngine
from we_care_mt.schema import Segment


class FakeGoogle:
    def translate_batch(self, texts, **_kwargs):
        return [f"አማ:{t}" for t in texts]


def test_engine_uses_google_backend(monkeypatch):
    eng = TranslationEngine(
        model_key="google",
        model_entry={"hf_id": "external:google-translate"},
        inference={"cache_translations": False, "batch_size": 1, "checkpoint_every": 100},
        glossary=None,
        backend="google",
    )
    eng._translator = FakeGoogle()
    segs = [
        Segment(
            segment_id="s1",
            document_name="t.pdf",
            page_number=1,
            block_id="b1",
            kind="paragraph",
            source_text="A condom prevents pregnancy.",
            is_translatable=True,
        )
    ]
    out = eng.translate_segments(segs)
    assert out[0].status == "translated"
    assert out[0].translated_text.startswith("አማ:")
    assert out[0].model_id == "external:google-translate"
