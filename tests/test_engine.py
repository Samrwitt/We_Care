from pathlib import Path

from we_care_mt.glossary.manager import Glossary
from we_care_mt.schema import Segment
from we_care_mt.translation.engine import TranslationEngine


def test_mock_engine_preserves_numbers_and_uses_glossary(glossary_csv, tmp_path: Path):
    g = Glossary.load(glossary_csv)
    engine = TranslationEngine(
        model_key="mock",
        model_entry={"hf_id": "mock"},
        inference={"batch_size": 2, "cache_translations": True, "checkpoint_every": 1},
        glossary=g,
        cache_dir=tmp_path / "cache",
        checkpoint_path=tmp_path / "ckpt.jsonl",
        backend="mock",
    )
    segs = [
        Segment(
            segment_id="a",
            document_name="d.pdf",
            page_number=1,
            block_id="b",
            kind="paragraph",
            source_text="Family planning reduces unintended pregnancy. Take 100 mg.",
        )
    ]
    out = engine.translate_segments(segs)
    assert out[0].status == "translated"
    assert "100" in out[0].translated_text
    assert "የቤተሰብ ምጣኔ" in out[0].translated_text
    assert engine.stats["translated"] + engine.stats["cached"] >= 1

    # Second run should hit cache / checkpoint
    segs2 = [
        Segment(
            segment_id="a",
            document_name="d.pdf",
            page_number=1,
            block_id="b",
            kind="paragraph",
            source_text="Family planning reduces unintended pregnancy. Take 100 mg.",
        )
    ]
    engine2 = TranslationEngine(
        model_key="mock",
        model_entry={"hf_id": "mock"},
        inference={"batch_size": 2, "cache_translations": True, "checkpoint_every": 1},
        glossary=g,
        cache_dir=tmp_path / "cache",
        checkpoint_path=tmp_path / "ckpt.jsonl",
        backend="mock",
    )
    engine2.translate_segments(segs2)
    assert segs2[0].status == "translated"
    assert engine2.stats["cached"] >= 1 or segs2[0].translated_text == out[0].translated_text
