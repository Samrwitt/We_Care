from pathlib import Path

from fastapi.testclient import TestClient

from we_care_nlu.api.app import app
from we_care_nlu.config import load_config
from we_care_nlu.pipeline import NLUEngine

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_kb.json"


def client():
    # Reset singleton toward hash engine with empty KB, then ingest via API.
    import we_care_nlu.api.app as mod

    mod._ENGINE = NLUEngine(load_config(), model_key="hash")
    return TestClient(app)


def test_health_and_models():
    c = client()
    root = c.get("/")
    assert root.status_code == 200
    assert "text/html" in root.headers["content-type"]
    assert "ዌ ኬር" in root.text
    assert "/nlp/analyze" in root.text
    meta = c.get("/meta")
    assert meta.status_code == 200
    assert meta.json()["service"] == "we-care-nlu"
    assert meta.json()["generative_decoding"] is False
    h = c.get("/health")
    assert h.status_code == 200
    assert h.json()["generative_decoding"] is False
    m = c.get("/models")
    assert "labse" in m.json()["embeddings"]
    assert m.json()["embeddings"]["afro_xlmr"]["amharic_sentence_encoder"] is False


def test_ingest_analyze_search():
    c = client()
    ing = c.post(
        "/documents/ingest",
        json={"path": str(FIXTURE), "document_version": "mini-1", "review_status": "unreviewed"},
    )
    assert ing.status_code == 200
    assert ing.json()["ingested_chunks"] >= 5
    an = c.post("/nlp/analyze", json={"text": "የእርግዝና መከላከያ ዘዴዎች ምንድን ናቸው?"})
    assert an.status_code == 200
    body = an.json()
    assert body["generative_answer"] is None
    assert "intent" in body
    assert isinstance(body["retrieval_results"], list)
    se = c.post("/retrieval/search", json={"query": "የመርፌ የጎንዮሽ ጉዳት", "top_k": 3})
    assert se.status_code == 200
    assert "hits" in se.json()
