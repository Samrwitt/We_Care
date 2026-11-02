from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def glossary_csv(project_root: Path) -> Path:
    return project_root / "data" / "glossary" / "medical_glossary.csv"


@pytest.fixture(scope="session")
def fonts_dir(project_root: Path) -> Path:
    return project_root / "data" / "fonts"


@pytest.fixture
def mini_pdf(tmp_path: Path) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    latin = ROOT / "data" / "fonts" / "NotoSans-Regular.ttf"
    if latin.exists():
        pdfmetrics.registerFont(TTFont("MiniLatin", str(latin)))
        font = "MiniLatin"
    else:
        font = "Helvetica"
    path = tmp_path / "mini_guideline.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    c.setFont(font, 16)
    c.drawString(72, h - 72, "FAMILY PLANNING SERVICE")
    c.setFont(font, 11)
    c.drawString(72, h - 110, "Family planning is one of the key strategies to improving maternal health.")
    c.drawString(72, h - 130, "Do not use emergency contraception as a regular method.")
    c.drawString(72, h - 150, "Take doxycycline 100 mg twice daily for 7 days.")
    c.drawString(72, h - 170, "The MMR in Ethiopia still stands at 401/100,000 live birth.")
    c.setFont(font, 8)
    c.drawString(72, 36, "MoH, MCHN directorate, FP Service Integration National Implementation Guideline; June, 2021")
    c.drawRightString(w - 72, 36, "1")
    c.showPage()
    c.setFont(font, 13)
    c.drawString(72, h - 72, "2. RATIONALE")
    c.setFont(font, 11)
    c.drawString(72, h - 110, "1. Policy makers")
    c.drawString(72, h - 130, "2. Health managers")
    c.drawString(72, h - 160, "Contraindicated: Combined oral contraceptives are contraindicated in migraine with aura.")
    c.setFont(font, 8)
    c.drawString(72, 36, "MoH, MCHN directorate, FP Service Integration National Implementation Guideline; June, 2021")
    c.drawRightString(w - 72, 36, "2")
    c.save()
    return path
