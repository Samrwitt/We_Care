from pathlib import Path
import yaml
from we_care_nlu.intent.classifier import IntentClassifier

LEXICON = Path(__file__).resolve().parents[1] / 'data/intents/lexicon.yaml'
clf = IntentClassifier(yaml.safe_load(LEXICON.read_text(encoding='utf-8')))

def test_scenarios():
    checks = [
        ('የኤችአይቪ ምልክቶች ምንድን ናቸው?', 'symptoms', 'hiv'),
        ('የኮንዶም የጎንዮሽ ጉዳት ምንድን ነው?', 'side_effects', 'condom'),
        ('እርጉዝ ሆኜ ኢምፕላንት መጠቀም እችላለሁ?', 'eligibility', 'contraceptive_implant'),
        ('የቤተሰብ ምጣኔ አገልግሎት የት ይገኛል?', 'access', 'family_planning'),
        ('የእርግዝና መከላከያ ዘዴዎች ምንድን ናቸው?', 'family_planning_methods', 'family_planning'),
        ('የወር አበባ ሕመም ምክንያት ምንድን ነው?', 'causes', 'menstruation'),
        ('የጡት ካንሰር ምርመራ መቼ ላድርግ?', 'screening', 'breast_cancer'),
        ('የጎንዮሽ ጉዳቱስ?', 'side_effects', None),
        ('እንዴት ይታከማል?', 'treatment', None),
    ]
    for text, intent, topic in checks:
        result = clf.predict(text)
        assert result['intent'] == intent, (text, result['intent'], intent)
        assert topic in result['topics'] if topic is not None else True, (text, result['topics'])

def test_non_medical_greeting():
    assert clf.predict('hi')['intent'] == 'greeting'
    assert clf.predict('hiv')['intent'] != 'greeting'

def test_safety_cues_not_implied_diagnosis():
    result = clf.predict('መተንፈስ አልችልም')
    assert 'breathing_difficulty' in result['safety_cues']
    assert result['confidence_status'] == 'requires_calibration'
