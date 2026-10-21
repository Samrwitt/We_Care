# We Care — English→Amharic Medical PDF Translation Pipeline

Production-oriented **English-to-Amharic** pipeline for Ethiopian Ministry of Health reproductive-health guidelines, family-planning materials, clinical guidance, and related PDFs.

The system extracts selectable PDF text, segments it safely, translates with NLLB / AfriScience-MT models (or a mock backend for tests), enforces a bilingual medical glossary, runs automated medical QA, and writes Amharic PDF, bilingual review PDF, DOCX, alignments, and a quality report.

**This is not a clinically validated medical device.** Automated checks that pass do **not** mean a translation is medically safe. Qualified bilingual clinical review is required before any patient-facing or programmatic use.

---

## What was analyzed from the attached guideline

The representative source is *Family Planning Service Integration National Implementation Guideline* (MoH MCHN directorate, August 2021; 56 pages; InDesign 15, untagged PDF).

Observed characteristics the pipeline is built around:

| Feature | Handling |
|---|---|
| Running header/footer `MoH, MCHN directorate, FP Service Integration…` | Detected by repetition; excluded from translation; kept in extraction JSON |
| Numbered sections (`2. RATIONALE`, `11.3 Steps…`) and Annexes | Heading classifier (font size + numbering) |
| Bullets (`§`, `Ø`, numbered lists) | List-item segments, kept atomic when short |
| Complex InDesign tables (Table 1 multi-level integration matrix) | PyMuPDF `find_tables()`; **flagged uncertain** when merged cells / high empty ratio |
| Acronyms (FP, CPR, IUCD, PMTCT, VIA, HEP, WDA, …) | Glossary + freeze patterns; abbreviations not treated as sentence boundaries |
| Numeric public-health stats (`401/100,000`, `41%`, `10 IU`) | Protected placeholders; QA fails the segment if numbers/units change |
| Empty cover reverse (page 2) | Flagged `empty_page`; never silently dropped from the page inventory |

**Layout limitation:** output PDFs are structured reconstructions (headings, paragraphs, tables, source-page labels). They are not pixel-identical InDesign copies. Two-column and overlapping decorative layers are flagged on the source page.

Seventeen source PDFs from the attached RH Guidelines bundle can be processed as a folder once placed in `data/source_pdfs/`.

---

## Models (verified)

| Key | Hugging Face id | License | Domain | VRAM (approx.) | Max input | Clinical? |
|---|---|---|---|---|---|---|
| `dsfsi_600m` | [`dsfsi/nllb_200_distilled_600m-eng-amh`](https://huggingface.co/dsfsi/nllb_200_distilled_600m-eng-amh) | Apache-2.0 | Scientific AfriScience-MT | ~6 GB | 256 tokens | **No** |
| `dsfsi_1_3b` | [`dsfsi/nllb_200_1.3b-eng-amh`](https://huggingface.co/dsfsi/nllb_200_1.3b-eng-amh) | Apache-2.0 | Scientific AfriScience-MT | ~12 GB | 256 tokens | **No** |
| `nllb_baseline_600m` | [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M) | **CC-BY-NC** | General, not medical | ~6 GB | 512 tokens | **No** |
| `mock` | (none) | Apache-2.0 | Tests only | 0 | — | **No** |

Language codes for all NLLB checkpoints: **source `eng_Latn`**, **target `amh_Ethi`**. Tokenizers are `NllbTokenizer` / `NllbTokenizerFast` via Hugging Face Transformers.

Published AfriScience-MT test scores (authors, scientific domain — **not** this medical set):

- 600M Eng→Amh: BLEU 18.57, chrF 44.38
- 1.3B Eng→Amh: BLEU 19.28, chrF 45.51

Meta documents that NLLB-200 is **not intended for medical text**, document translation, or certified translation. DSFSI models are scientific-text fine-tunes, not clinical models. **Do not assume either is better than Google Translate on medical Amharic** until you run `we-care-mt benchmark` with human-reviewed references and, if you wish, a supplied Google baseline file.

CPU fallback is supported for the 600M models (slow). The 1.3B model is not practical on CPU. Generation uses batched beam search with configurable `num_beams` (default 5).

---

## Architecture

```
PDF ─┐
     ├─ extraction (PyMuPDF, reading order, tables, headers)
     ├─ structure analysis (headings, lists, footnotes)
     ├─ sentence-aware segmentation (≤ ~200 tokens)
     ├─ glossary protection (INN names, units, verified terms)
     ├─ batch NLLB inference + SQLite cache + JSONL checkpoints
     ├─ glossary enforcement + automated medical QA
     └─ outputs: Amharic PDF, bilingual PDF, DOCX, JSON/CSV, QA report
```

Package: `src/we_care_mt/` with modules `extraction`, `segmentation`, `translation`, `glossary`, `qa`, `evaluation`, `output`. Settings live in `configs/default.yaml` and `configs/models.yaml` — not in code.

---

## Installation

Python 3.10+ (3.11/3.12 recommended). GPU optional (CUDA PyTorch).

```bash
cd We_Care
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
# GPU: install the CUDA build of torch for your driver, then reinstall transformers if needed.
```

Ethiopic fonts ship in `data/fonts/` (`NotoSansEthiopic-*.ttf`, `NotoSans-*.ttf`). System copies under `/usr/share/fonts/truetype/noto` are also detected.

### Download models

```bash
# Default / recommended starting point (Apache-2.0, ~2.5 GB)
we-care-mt download-models --model dsfsi_600m

# Larger model (~5 GB weights, more VRAM)
we-care-mt download-models --model dsfsi_1_3b

# Baseline (CC-BY-NC — non-commercial / research only)
we-care-mt download-models --model nllb_baseline_600m

# Inspect licenses, language codes, VRAM notes
we-care-mt verify-models
```

Weights go to `.hf_cache/` (configurable). First load will also work via `transformers` if the cache is empty and the network is available.

---

## Usage

```bash
# Smoke / CI: no neural download
we-care-mt translate data/samples --model mock --backend mock -o output

# One document
we-care-mt translate \
  "data/source_pdfs/FAMILY PLANNING SERVICE INTEGRATION NATIONAL IMPLEMENTATION GUIDELINE.pdf" \
  --model dsfsi_600m --device auto --batch-size 4 --beams 5 -o output

# Whole folder (10–50 PDFs)
we-care-mt translate data/source_pdfs --model dsfsi_600m -o output

# Extract only
we-care-mt extract path/to/file.pdf -o output/file.structure.json

# Re-run QA on an alignment JSON
we-care-mt qa output/.../file.alignment.json

# Benchmark the three neural models on the sample eval set
we-care-mt benchmark --models dsfsi_600m,dsfsi_1_3b,nllb_baseline_600m \
  --eval-set data/evaluation/sample_eval.jsonl -o output/benchmark/benchmark_report.json
```

Interrupted jobs resume from `checkpoints/<stem>.<model>.jsonl`. Identical segments are reused from `cache/translations.sqlite`.

### Google Translate comparison

This pipeline **does not call Google**. Put externally obtained translations in `data/evaluation/google_translate_baseline.jsonl` (`id`, `source`, `google_translate`). The benchmark includes that row when every eval item is filled.

---

## Outputs (per PDF)

1. `*.am.pdf` — Amharic reconstruction with source-page labels  
2. `*.bilingual.pdf` — English | Amharic review table  
3. `*.docx` — editable bilingual DOCX (Noto Sans Ethiopic)  
4. `*.alignment.json` + `*.alignment.csv` — segment id, document, page, source, target, model  
5. `*.qa.md` / `*.qa.json` / `*.qa.pdf` — findings with page, severity, recommended review action  
6. `*.structure.json` — full extraction audit (warnings, complex pages, tables)  
7. `run_summary.json`

Every translated segment stores **document name, page number, source text, translated text, and model identifier**.

---

## Medical terminology and QA

Glossary: `data/glossary/medical_glossary.csv`  
Freeze regexes: `data/glossary/freeze_patterns.json`  
Unresolved terms: `data/glossary/uncertain_terms.txt`

Confidence labels:

- `verified` / `standard` — enforced if the English term leaks into Amharic output  
- `provisional` — **flagged, never invented as if official**  
- `do_not_translate` / `preserve_source=true` — INN drug names, device brands, many abbreviations  

Automated QA (heuristic, not clinical sign-off):

- missing / empty / failed translation  
- number and dosage/unit mismatches; hallucinated numbers  
- English negation / warning / contraindication cues lost in Amharic  
- pregnancy, age, and duration references  
- leftover English; missing Ethiopic script  
- glossary inconsistency  
- extraction/table uncertainty  
- structural page omissions  

A report that contains zero findings still prints: *Automated checks do not establish medical safety.*

---

## Evaluation metrics

On identical test segments and inference settings the benchmark reports:

| Metric | Implementation |
|---|---|
| BLEU | SacreBLEU `corpus_bleu` |
| chrF++ | SacreBLEU `corpus_chrf(word_order=2)` |
| Adequacy | heuristic proxy (numbers + glossary + length) — **not** human adequacy |
| Amharic fluency | Ethiopic-character ratio proxy — **not** human fluency |
| Medical terminology accuracy | verified/standard glossary hit rate |
| Numerical / dosage accuracy | exact number-span match |
| Critical mistranslation rate | number or negation-loss rate |
| Speed | sentences / second |
| GPU memory | `torch.cuda.max_memory_allocated` |

**Ranking uses critical-error rate first, then terminology accuracy, then chrF++. BLEU is reported but is not the selection rule.**

`data/evaluation/sample_eval.jsonl` contains 25 items drawn from the Family Planning guideline plus dosage/warning sentences. References are **`developer_draft`**, not gold clinical translations. Replace them with blinded bilingual-reviewer references before trusting any ranking.

If a model is not downloaded, the benchmark records `status: error` and **does not invent a score**.

---

## Tests

```bash
pytest
```

Coverage includes UTF-8 Ethiopic round-trip, Amharic PDF font embedding, table/header extraction, glossary longest-match, number/negation QA, checkpoint resume, cache hits, failed/empty translations, and an **end-to-end smoke test that uses `--backend mock` and does not download NLLB**.

Mark `@pytest.mark.slow` / `@pytest.mark.gpu` if you add live-model tests locally.

---

## Docker

```bash
docker compose build
docker compose run --rm we-care-mt translate /app/data/source_pdfs --model mock --backend mock -o /app/output
```

The default image is CPU. For GPU, start `FROM nvidia/cuda` and install a CUDA PyTorch wheel.

---

## Configuration

Edit `configs/default.yaml` for paths, batch size, beams, QA thresholds, and output flags. Edit `configs/models.yaml` for Hugging Face ids and licenses.

---

## Limitations (explicit)

1. **Not clinically validated.** No model here is approved for unsupervised medical use.  
2. **AfriScience-MT is scientific, not clinical.** Domain shift on guidelines, dosages, and counselling language is expected.  
3. **facebook/nllb-200-distilled-600M is CC-BY-NC** and Meta advises against medical/legal use.  
4. **NLLB 256-token cap** on DSFSI checkpoints: long sentences are packed/split; a hard wrap is a last resort and is flagged.  
5. **PDF layout:** InDesign tables, text-on-shape, and scans without a text layer cannot be perfectly reconstructed. Uncertain tables and empty/image pages are flagged, never omitted quietly.  
6. **Mock backend** is for tests. It applies the glossary and leaves leftover English on purpose so QA can fire.  
7. **Google Translate** is compared only when you supply outputs.  
8. **Developer-draft Amharic references** in the eval set are not a substitute for MoH / clinician review.  
9. **Amharic morphology:** chrF++ is more informative than BLEU; both can look “good” while a negation or dose is wrong.  
10. Output PDFs require an Ethiopic-capable font (bundled). Word processors must also have Noto Sans Ethiopic installed to display DOCX correctly.

---

## Project layout

```
configs/                 default.yaml, models.yaml
data/glossary/           medical_glossary.csv, freeze_patterns.json
data/evaluation/         sample_eval.jsonl, google_translate_baseline.jsonl
data/fonts/              Noto Sans Ethiopic + Latin
data/source_pdfs/        place 10–50 input PDFs here
src/we_care_mt/          library + CLI (we-care-mt)
scripts/download_models.py
tests/                   unit, integration, smoke
examples/output/         expected artifact layout
```

---

## Citation / provenance

- NLLB: NLLB Team et al., *No Language Left Behind*, 2022  
- AfriScience-MT: Abdulmumin et al., arXiv:2605.29741  
- Source guideline: Ethiopia Ministry of Health, MCHN directorate, *Family Planning Service Integration National Implementation Guideline*, 2021
