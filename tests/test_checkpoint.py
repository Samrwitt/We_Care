from we_care_mt.schema import Segment
from we_care_mt.translation.checkpoint import CheckpointStore


def test_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    store = CheckpointStore(path)
    seg = Segment(
        segment_id="s1",
        document_name="d.pdf",
        page_number=3,
        block_id="b",
        kind="paragraph",
        source_text="hello",
        translated_text="ሰላም",
        model_id="mock",
        status="translated",
    )
    store.save_many([seg])
    store2 = CheckpointStore(path)
    fresh = Segment(
        segment_id="s1",
        document_name="d.pdf",
        page_number=3,
        block_id="b",
        kind="paragraph",
        source_text="hello",
    )
    n = store2.apply([fresh])
    assert n == 1
    assert fresh.translated_text == "ሰላም"
    assert fresh.status == "translated"
