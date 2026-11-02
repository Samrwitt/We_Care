from pathlib import Path

from we_care_nlu.intent.classifier import IntentClassifier
from we_care_nlu.ner.extract import EntityExtractor

CLF = IntentClassifier()
NER = EntityExtractor()
ROOT = Path(__file__).resolve().parents[1]


def test_paraphrases_share_family_planning_intent():
    qs = [
        "የእርግዝና መከላከያ ዘዴዎች ምንድን ናቸው?",
        "እርግዝናን ለመከላከል ምን አማራጮች አሉ?",
        "ልጅ ላለመውለድ ምን መጠቀም ይቻላል?",
    ]
    intents = [CLF.predict(q)["intent"] for q in qs]
    assert all(i in {"family_planning_methods", "eligibility", "definition"} for i in intents)
    assert intents.count("family_planning_methods") >= 2


def test_side_effects_vs_contraindication():
    se = CLF.predict("የመርፌ የጎንዮሽ ጉዳት ምንድን ነው?")
    contra = CLF.predict("ጡት በማጥባት ጊዜ ይህ ዘዴ የተከለከለ ነው?")
    assert se["intent"] == "side_effects"
    assert contra["intent"] == "contraindications"
    assert se["intent"] != contra["intent"]


def test_injectable_entity():
    ents = NER.extract("እርግዝና መከላከያ መርፌ ምንድን ነው?")
    assert any(e.value == "injectable_contraceptive" for e in ents)


def test_dosage_regex():
    ents = NER.extract("Take doxycycline 100 mg twice daily")
    assert any(e.type == "dosage" and "100" in e.value for e in ents)


def test_does_not_invent_pregnancy():
    ents = NER.extract("ኮንዶም የት ይገኛል?")
    assert not any(e.type == "pregnancy_status" for e in ents)


def test_contraception_phrase_is_not_pregnant_status():
    ents = NER.extract("የእርግዝና መከላከያ ዘዴዎች ምንድን ናቸው?")
    assert not any(e.type == "pregnancy_status" for e in ents)


def test_depo_maps_to_injectable():
    ents = NER.extract("ዴፖ መውሰድ እችላለሁ?")
    assert any(e.value == "injectable_contraceptive" for e in ents)


def test_greeting_intent():
    assert CLF.predict("ሰላም")["intent"] == "greeting"
