from we_care_mt.output.fonts import locate_fonts


def test_ethiopic_roundtrip_utf8(tmp_path):
    text = "የቤተሰብ ምጣኔ፣ ኤች.አይ.ቪ/ኤድስ፣ 100 mg"
    path = tmp_path / "am.txt"
    path.write_text(text, encoding="utf-8")
    assert path.read_text(encoding="utf-8") == text
    assert "\u1200" <= text[0] <= "\u137F"


def test_ethiopic_fonts_present(fonts_dir):
    fonts = locate_fonts(fonts_dir)
    assert fonts.amharic_regular.exists()
    assert fonts.amharic_regular.stat().st_size > 1000
