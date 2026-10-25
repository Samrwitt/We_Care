import json

import httpx

from we_care_nlu.translation.mt_client import MTSpanClient, TranslatorNotReady


def test_client_sends_the_matched_sentence_to_the_pdf_translator():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"text": "የተተረጎመ", "qa_notes": []})

    client = MTSpanClient(
        "http://we-care-mt:8090",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert client.to_amharic("Give 10 mg.") == "የተተረጎመ"
    assert seen["url"] == "http://we-care-mt:8090/translate"
    assert seen["body"] == {"text": "Give 10 mg.", "direction": "en-am"}
    assert client.backend == "mt"
    assert client.can_translate_reply is True


def test_client_reports_when_the_model_is_still_loading():
    client = MTSpanClient(
        "http://we-care-mt:8090",
        client=httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(503, json={"status": "loading"}))),
    )
    try:
        client.to_english("ሰላም")
    except TranslatorNotReady:
        return
    raise AssertionError("expected TranslatorNotReady")
