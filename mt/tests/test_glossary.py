from we_care_mt.glossary.manager import Glossary
from we_care_mt.glossary.protect import apply_protection, restore_protection


def test_longest_match_family_planning(glossary_csv):
    g = Glossary.load(glossary_csv)
    hits = g.find_hits("Emergency contraception is not a regular family planning method.")
    ens = [h.english.lower() for h in hits]
    assert "emergency contraception" in ens
    assert "family planning" in ens


def test_do_not_translate_inn_preserved(glossary_csv):
    g = Glossary.load(glossary_csv)
    text = "Give oxytocin 10 IU intramuscular now."
    protected, mapping = apply_protection(text, g)
    assert "10 IU" in mapping.values() or "10IU" in {v.replace(" ", "") for v in mapping.values()}
    restored = restore_protection(protected, mapping)
    assert "10" in restored


def test_enforce_verified_term(glossary_csv):
    g = Glossary.load(glossary_csv)
    src = "Family planning reduces maternal mortality."
    tgt = "Family planning የእናቶች ሞትን ይቀንሳል።"
    out, notes = g.enforce_target(src, tgt)
    assert "የቤተሰብ ምጣኔ" in out
    assert any(n.startswith("enforced:") for n in notes)


def test_verified_terms_restore_as_amharic_not_google_synonyms(glossary_csv):
    g = Glossary.load(glossary_csv)
    src = "Contraceptive method options include condoms and injectables."
    protected, mapping = apply_protection(src, g)
    assert "Contraceptive" not in protected or "WCMT" in protected
    assert "condoms" not in protected.lower() or "WCMT" in protected
    # Simulate MT leaving placeholders alone, then restoring glossary Amharic.
    restored = restore_protection(protected, mapping)
    assert "የእርግዝና መከላከያ" in restored
    assert "ኮንዶሞች" in restored or "ኮንዶም" in restored
    assert "በመርፌ" in restored


def test_provisional_terms_are_flagged(glossary_csv):
    g = Glossary.load(glossary_csv)
    unverified = g.unverified_in_text("The method mix at the facility is limited.")
    assert any(e.english.lower() == "method mix" for e in unverified)
