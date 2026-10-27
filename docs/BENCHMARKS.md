# Benchmark evidence

Reference record of the measurements behind the defaults of the service (OCR, NER, fuzzy matching, embeddings, chunking, refusal threshold) and of the capacity estimate. All figures are copied from the result files of the original benchmark session; nothing is re-estimated here except where a derivation is stated explicitly. Small scripts that produced the numbers are in [`../benchmarks/`](../benchmarks/README.md); the capacity estimate is recomputed by [`capacity_estimate.py`](capacity_estimate.py).

Conventions: percentages unless stated; `[a-b]` or `(a-b)` is the 95 % percentile-bootstrap confidence interval reported by the source (resampling documents for the 600-document corpus, 200 documents for the name lists, sentences for the public NER sets, queries for the retrieval sets; 1,000 resamples, seed 0, 500 for the refusal tables). A cell without an interval means the source reports none. "Source" names the result file in the original session folder (`bench/...`); those folders were not committed (only scripts are, in `benchmarks/`).

Environment recorded in the sources (details per section): Windows 11 (10.0.22631), Python 3.12.10, NVIDIA GeForce RTX 4070 Ti, "16 cores" (ner_corpus `reproduce.sh`). The result files do not record the CPU model; the machine on which this document was assembled reports an AMD Ryzen 7 7700X (8 cores, 16 logical processors), which matches the "16 cores" note but is not recorded in any result file. OCR ran inside a Linux container (Docker Desktop on that host, Docker version not recorded). Accuracy runs of NER and embeddings used the GPU; every CPU latency figure used 4 threads.

## Contents

1. Synthetic corpus and its limitations
2. OCR, DPI, preprocessing and language handling
3. NER on public data
4. NER end to end on OCR text (span rules, post-processing)
5. Fuzzy threshold and scorer comparison
6. Embedders
7. Chunking
8. Refusal threshold
9. Capacity estimate
10. Data-leakage and provenance caveats
11. Defaults chosen
12. Not adopted / future work

---

## 1. Synthetic corpus and its limitations

**What it is.** 600 synthetic, image-only (no text layer) business PDFs, 120 per language (en, es, pt, fr, de), 1,045 pages (1-3 pages per document: 190 / 375 / 35 documents), seven document types (memo 90, minutes 85, contract 85, kyc 85, report 85, invoice 85, letter 85). Noise level per language exactly 40 % clean / 40 % medium / 20 % hard (240 / 240 / 120 documents; 417 / 420 / 208 pages). Ground truth: 7,489 person mentions (3,373 per-document persons, 3,100 distinct first+last names), 2,425 non-person "trap" spans (org, person-like org, street, e-mail, product, place), 2,871 answerable questions (4.79 per document; 600 also have a cross-lingual paraphrase), 300 unanswerable questions (60 per language: 20 absent-entity, 20 cross-document, 20 off-topic), 200 name lists (40 per language) with 1,687 queries (decoy 493, exact 432, near-miss decoy 400, case 181, accent 107, OCR-typo 74). Mention types: full 2,726, title+full 1,570, ALL-CAPS 1,090, "Last, First" 1,019, title+last 655, initial+last 296, last only 133. Source: `bench/data/README.md` (copy: `benchmarks/corpus/DATA_README.md`).

**How it is generated.** `benchmarks/corpus/make_corpus.py`, deterministic, `BASE_SEED = 20260601`; document *i* of language index *l* uses `seed = BASE_SEED + l*10000 + i*7 + 1` (layout `random.Random(seed)`, page count `Random(seed ^ 0x5bd1e995)`, degradation `Random(seed*3+11)` / `Random(seed*5+3)` / `numpy.default_rng(seed*101+page)`, QA and name lists `Random(seed ^ 0xA5A5)`, unanswerable set `Random(BASE_SEED+99)`). Names come from curated pools mixed with Faker locales (about 12 % of persons carry a name from another language's pool). Layout is drawn with PyMuPDF using eight Windows TrueType fonts (Arial, Times New Roman, Calibri, Georgia, Verdana, Tahoma, Trebuchet, Courier New), 8-12 pt, single/two-column, tables, letterheads, running header/footer, signature scribbles. Exact word boxes are read from the PDF before rasterisation and matched one-to-one to the drawn token stream. Pages are rasterised at 150 DPI (a 300-DPI variant exists for 100 documents), then degraded:

| level | rotation | shear | blur sigma @150 dpi | noise sd | salt & pepper | JPEG q | contrast loss | dark edges / shadow |
|---|---|---|---|---|---|---|---|---|
| clean | 0-0.3 deg | ~0 | 0.3-0.5 | 1-3 | 0 | 80-90 | 0-6 | none |
| medium | 0.5-1.0 deg | 0.05-0.25 deg | 0.55-0.85 | 4-8 | 0.04 % | 55-72 | 10-25 | 20 % / 30 % |
| hard | 1.0-2.0 deg | 0.2-0.6 deg | 0.85-1.15 | 9-15 | 0.25 % | 35-50 | 25-45 | 55 % / 60 % |

Also: 30 % colour scans, low-frequency uneven illumination, JPEG compression; boxes are mapped through the same affine transform. Python libraries installed in the benchmark virtual environment (the generator's own environment is not recorded explicitly): PyMuPDF 2.0, Pillow 13.0.0, numpy 2.5.4, Faker 40.43.0. The source README states that byte-for-byte reproduction requires the same library versions and fonts.

**Verification.** `validate_corpus.py` (counts, no text layer, one image per page, boxes inside the page, QA answers on the gold page, decoys never equal a real person) and `check_alignment.py` (word boxes cover ink): clean 3120/3132 = 99.6 %, medium 3086/3111 = 99.2 %, hard 2424/2556 = 94.8 %, overall 8630/8799 = 98.1 %; control with all boxes shifted 7 pt down: 253/8799 = 2.9 % (source `bench/align_summary.txt`). The hard-level failures are almost all faint 8-pt grey footer/letterhead text.

**Limitations (carried over from the source README and reports).**

- Text is template-generated: about 150 body templates per language (plus about 120 labels and 40 question templates), approximate grammar. Retrieval is easier than on real documents; questions are fact-lookup only (no multi-hop, summarisation or numeric reasoning). Documents are near-duplicates in structure, so lexical overlap between documents is high.
- Clean vector text in eight fonts, rasterised and degraded parametrically: no handwriting, stamps, photographs, merged-cell tables, landscape pages, fax-style 1-bit artefacts, moire or page curvature. The hard level is "realistic but moderate"; real archives can be far worse.
- Names are Latin-script (Western European / Latin American with some foreign names); no CJK, Arabic or Cyrillic. Only Western/Latin names were tested, so letters that NFKD does not decompose (o-slash, l-stroke, d-stroke, ae) are not covered.
- Boxes are font-metric word boxes (ascender to descender), taller than the ink. The OCR evaluation calibrates the ink/truth height ratio at 0.732 (centre offset +0.011 x truth height) for the chosen configuration, i.e. truth boxes are about 1.37 times taller than the ink, so a perfectly placed ink box reaches an IoU of about 0.73. IoU >= 0.5 is therefore the criterion used everywhere; IoU >= 0.7 values are reported but are not meaningful as an accuracy measure. The "adjusted" columns (`adjIoU`, `adjF1`) shrink the truth box to ink height.
- `ocr_typo` and `accent` name-list variants are rule-based, not real OCR errors. Unanswerable questions are template-based and easier to separate than real ones. Letters and contracts contain few person mentions (about 5 each); minutes about 30.
- The three sample PDFs of the product are plain black Helvetica-like text on white 150-DPI pages; the corpus resembles them mostly at the `clean` level.

**Reproduce** (from the repository root; fonts are read from `C:/Windows/Fonts/`, so the generator runs on Windows):

```
python -I benchmarks/corpus/make_corpus.py --out benchmarks/data
python -I benchmarks/corpus/validate_corpus.py benchmarks/data
python -I benchmarks/corpus/check_alignment.py benchmarks/data --n 60
python -I benchmarks/corpus/make_readme.py benchmarks/data
```

---

## 2. OCR, DPI, preprocessing and language handling

**What was measured.** Tesseract on the 1,045 pages (all 600 documents), against ground truth: character / word error rate (CER, WER), accent error rate (AccErr: share of truth non-ASCII letters not reproduced exactly), whether each of the 7,489 person mentions is read exactly (NameExact), after NFKD accent strip + casefold (NameNorm), or with `token_sort_ratio >= 90` (NameFuzzy), and the IoU of the union box of the matching OCR words against the truth box (IoU >= .5 / >= .7, mean IoU, `adj` = ink-calibrated truth box). Source: `bench/ocr/results/final_*.md`, `stageA_grid_summary.md`, `stageC_300dpi_vs_150.md`, `timing_summary.md`, `bench/ocr/logs/stageA2_by_noise.md`.

**Setup (as recorded in `ocr/run_all.sh`, `Dockerfile`, `logs/pip_freeze.txt`).** Container `python:3.12-slim` (Debian 13), apt `tesseract-ocr` 5.5.0 with `tessdata_fast` (LSTM-only integer models) for eng, spa, por, fra, deu; PyMuPDF 2.0, RapidFuzz 3.14.6, numpy 2.5.4, opencv-python-headless 5.0.0.93. Each page is rendered to greyscale by PyMuPDF at the chosen DPI, optionally preprocessed, and passed to one `tesseract` process: `--oem 1 --psm <n> --dpi <render dpi> -l <lang(s)> tsv`, `OMP_THREAD_LIMIT=1`, one page per process. Every embedded image in the corpus is exactly 150 DPI (300 DPI for the 176 pages of the 300-DPI subset; `logs/native_res.txt`). Stage A used a seed-0 stratified sample of 120 documents (8 per language x noise level, 208 pages) with 14 parallel workers (the s/page columns of stages A and B are contended times, not latencies); Stage B ran all 600 documents.

**Preprocessing "flat"** (reference implementation `preprocess()` in `benchmarks/ocr/scripts/ocr_run.py`): divide the grey image by a Gaussian-blurred background (sigma 25 x dpi/150) to remove uneven illumination, 3x3 median filter, then stretch levels from the 1st percentile to white. No binarisation.

### 2.1 Headline (all 600 documents, 1,045 pages, 7,489 mentions)

| configuration (dpi / pre / psm / language argument) | CER | WER | AccErr | NameExact | NameNorm | NameFuzzy | IoU>=.5 | IoU>=.7 | mean IoU | adjIoU>=.5 | s/page (14 workers) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 150 / none / 3 / single | 15.6 [13.5-17.7] | 20.5 [18.2-22.8] | 16.2 [13.8-19.0] | 81.5 [78.7-84.3] | 85.6 [82.7-88.3] | 87.2 [84.5-89.9] | 84.8 [81.8-87.6] | 49.8 [47.7-52.0] | 0.622 [0.600-0.642] | 83.8 [80.8-86.6] | 0.86 |
| 200 / none / 3 / single | 16.0 [14.0-18.1] | 22.9 [20.7-25.3] | 16.2 [13.7-18.9] | 82.1 [79.5-84.6] | 86.4 [83.7-88.8] | 88.3 [85.8-90.5] | 85.0 [82.0-87.7] | 52.4 [50.0-54.6] | 0.630 [0.607-0.650] | 82.0 [78.7-85.1] | 1.04 |
| **200 / flat / 3 / single (default)** | **5.4 [4.8-6.1]** | **11.0 [10.2-12.0]** | **6.4 [5.6-7.3]** | **90.5 [89.3-91.7]** | **95.7 [94.8-96.5]** | **97.6 [96.8-98.3]** | **96.2 [95.1-97.0]** | 51.2 [49.4-52.8] | 0.694 [0.687-0.701] | **94.0 [92.8-95.0]** | 1.31 |
| 200 / flat / 3 / all five languages | 5.5 [4.9-6.2] | 12.3 [11.4-13.2] | 19.5 [18.3-20.8] | 86.1 [85.0-87.2] | 96.6 [95.8-97.3] | 97.8 [97.1-98.5] | 96.5 [95.5-97.3] | 51.2 [49.5-52.9] | 0.697 [0.690-0.703] | 94.1 [93.0-95.2] | 2.35 |

Source: `final_overall.md`. "single" = Tesseract told the document's true language; "all" = `eng+spa+por+fra+deu`. The mean IoU of about 0.70 is close to the 0.73 ceiling given by the box-height mismatch (section 1).

### 2.2 By noise level (same four configurations)

| configuration | noise | CER | WER | NameFuzzy | IoU>=.5 | adjIoU>=.5 |
|---|---|---|---|---|---|---|
| 150 / none / single | clean | 2.5 [2.1-3.0] | 5.4 [4.9-5.9] | 99.5 [99.2-99.8] | 99.4 [99.0-99.7] | 99.4 [99.1-99.7] |
| | medium | 5.0 [4.1-6.1] | 9.6 [8.6-10.7] | 97.4 [96.3-98.2] | 95.9 [94.8-96.9] | 95.0 [93.6-96.1] |
| | hard | 64.5 [60.4-68.7] | 74.1 [70.5-77.4] | 36.0 [29.6-42.3] | 26.4 [21.1-31.6] | 22.8 [18.1-27.7] |
| 200 / none / single | clean | 2.2 [1.9-2.6] | 5.0 [4.6-5.4] | 99.7 [99.4-99.9] | 99.6 [99.2-99.8] | 99.6 [99.2-99.8] |
| | medium | 7.3 [6.4-8.3] | 15.4 [14.4-16.4] | 96.9 [95.8-97.8] | 94.5 [93.3-95.6] | 91.8 [90.4-93.2] |
| | hard | 62.4 [58.8-65.9] | 75.8 [73.1-78.5] | 42.7 [36.3-48.8] | 29.8 [24.9-34.4] | 19.2 [15.3-23.1] |
| **200 / flat / single** | clean | 2.7 [2.3-3.1] | 5.9 [5.4-6.3] | 99.4 [99.1-99.7] | 99.2 [98.8-99.6] | 98.7 [98.2-99.1] |
| | medium | 3.5 [3.0-3.9] | 8.0 [7.3-8.7] | 99.0 [98.6-99.4] | 98.4 [97.8-98.9] | 96.0 [95.1-96.8] |
| | hard | 14.9 [12.6-17.4] | 27.9 [25.0-31.0] | 90.3 [85.9-93.7] | 84.1 [79.7-88.3] | 78.5 [74.0-83.1] |
| 200 / flat / all | clean | 2.9 [2.5-3.3] | 7.4 [6.9-8.0] | 99.4 [99.0-99.7] | 99.2 [98.8-99.6] | 98.8 [98.3-99.2] |
| | medium | 3.6 [3.2-4.1] | 9.3 [8.7-10.1] | 99.2 [98.8-99.6] | 98.5 [98.0-99.0] | 96.0 [95.1-96.9] |
| | hard | 14.7 [12.5-17.3] | 28.4 [25.5-31.6] | 91.2 [87.3-94.5] | 85.5 [81.4-89.5] | 79.2 [74.6-83.8] |

Source: `final_by_noise.md`. Flattening is what rescues the hard level (illumination gradients, shadow bands and dark edges defeat Tesseract's own binarisation): CER 62.4 -> 14.9, NameFuzzy 42.7 -> 90.3.

### 2.3 By language and by language x noise (default configuration 200 / flat / psm 3 / single)

| lang | CER | WER | AccErr | NameExact | NameNorm | NameFuzzy | IoU>=.5 | s/page (14 workers) |
|---|---|---|---|---|---|---|---|---|
| en | 4.5 [3.6-5.5] | 9.6 [8.0-11.3] | 77.2 [61.7-89.5] (see note) | 90.7 [88.7-92.5] | 97.1 [95.8-98.2] | 98.8 [97.7-99.6] | 98.1 [96.9-99.1] | 1.25 |
| es | 5.0 [3.8-6.2] | 10.2 [8.4-12.2] | 8.1 [6.6-9.7] | 88.5 [85.5-91.3] | 94.5 [92.3-96.4] | 96.3 [94.5-97.9] | 95.1 [92.8-97.1] | 1.35 |
| pt | 5.9 [4.7-7.2] | 12.3 [10.3-14.6] | 6.0 [4.8-7.4] | 91.9 [89.5-93.9] | 95.6 [93.5-97.2] | 98.3 [96.9-99.3] | 96.6 [94.6-98.0] | 1.30 |
| fr | 6.1 [4.5-8.0] | 11.6 [9.3-14.1] | 4.5 [2.9-6.3] | 91.2 [88.2-93.9] | 95.8 [93.2-97.8] | 97.2 [95.1-98.8] | 95.2 [92.1-97.6] | 1.27 |
| de | 5.2 [3.8-7.0] | 11.3 [9.4-13.5] | 6.9 [4.9-9.4] | 90.4 [86.9-93.2] | 95.5 [92.6-97.7] | 97.5 [94.8-99.4] | 95.8 [92.7-98.2] | 1.39 |

Note: the English AccErr (and the English values of the "all" row, 49.4 [35.8-63.1]) have very wide intervals and are not interpretable; the source gives no explanation. Source: `final_by_lang.md`.

| lang | noise | CER | WER | NameExact | NameNorm | NameFuzzy | IoU>=.5 |
|---|---|---|---|---|---|---|---|
| en | clean | 3.0 [2.1-4.1] | 5.8 [4.9-6.9] | 92.6 [90.0-95.0] | 99.4 [98.5-100.0] | 100.0 [100.0-100.0] | 99.9 [99.5-100.0] |
| en | medium | 3.0 [2.1-3.9] | 7.1 [5.8-8.7] | 91.3 [88.4-93.9] | 97.7 [96.2-99.1] | 99.3 [98.5-100.0] | 98.6 [97.7-99.6] |
| en | hard | 11.8 [8.5-15.3] | 24.2 [18.4-30.6] | 82.9 [77.3-88.5] | 88.3 [84.0-92.7] | 93.7 [89.1-97.5] | 91.4 [86.9-96.1] |
| es | clean | 2.5 [1.7-3.5] | 5.2 [4.4-6.3] | 92.3 [88.7-96.0] | 97.5 [95.8-98.9] | 98.3 [96.8-99.5] | 98.0 [96.5-99.3] |
| es | medium | 3.3 [2.5-4.2] | 7.9 [6.5-9.5] | 91.0 [87.6-93.9] | 96.3 [94.6-97.8] | 97.9 [96.7-99.0] | 97.6 [96.3-98.8] |
| es | hard | 13.9 [9.8-18.0] | 25.6 [20.0-31.5] | 72.2 [60.6-81.0] | 82.0 [72.1-89.3] | 87.3 [77.8-94.3] | 81.2 [71.4-88.8] |
| pt | clean | 2.8 [2.0-3.9] | 6.2 [5.3-7.4] | 93.3 [90.5-95.7] | 97.8 [96.5-98.9] | 99.3 [98.5-99.8] | 99.1 [98.2-99.8] |
| pt | medium | 4.6 [3.4-5.7] | 9.4 [7.9-11.0] | 94.4 [92.0-96.6] | 97.5 [95.8-98.6] | 99.4 [98.6-99.9] | 98.9 [98.0-99.6] |
| pt | hard | 14.4 [10.2-19.4] | 29.6 [22.9-37.5] | 84.6 [75.5-91.4] | 88.3 [80.1-94.2] | 94.8 [87.5-98.5] | 88.0 [80.3-92.8] |
| fr | clean | 2.6 [1.8-3.3] | 5.6 [4.7-6.5] | 94.1 [90.0-97.8] | 99.8 [99.5-100.0] | 99.8 [99.5-100.0] | 99.5 [98.9-100.0] |
| fr | medium | 3.1 [2.4-3.9] | 7.2 [6.1-8.6] | 94.5 [91.9-96.7] | 98.6 [97.5-99.4] | 99.3 [98.4-100.0] | 98.3 [97.2-99.2] |
| fr | hard | 18.5 [12.4-25.9] | 31.6 [24.6-39.2] | 74.7 [65.8-84.7] | 78.0 [68.3-88.6] | 84.6 [76.3-93.1] | 75.9 [64.5-87.2] |
| de | clean | 2.5 [1.7-3.4] | 6.6 [5.6-7.8] | 94.3 [91.4-96.9] | 98.8 [97.8-99.6] | 99.7 [99.2-100.0] | 99.7 [99.2-100.0] |
| de | medium | 3.4 [2.2-5.1] | 8.5 [6.7-10.8] | 93.6 [90.1-96.4] | 98.3 [96.8-99.6] | 99.2 [98.3-100.0] | 98.5 [96.3-100.0] |
| de | hard | 15.0 [9.6-22.5] | 27.2 [21.1-34.9] | 77.0 [63.5-86.9] | 84.0 [71.6-93.7] | 90.1 [76.2-98.6] | 83.4 [71.8-93.0] |

Source: `final_by_lang_noise.md` (the same file holds the other three configurations and the `AccErr`, `IoU>=.7`, `meanIoU`, `adjIoU` columns). By document type (same configuration, `final_by_doctype.md`): NameFuzzy 100.0 (letter), 99.1 (memo), 98.4 (minutes), 98.1 (report), 97.4 (contract), 93.6 (kyc), 93.2 (invoice); CER 1.8 (letter) to 12.5 (kyc), 7.9 (invoice). By mention type (`final_by_mentiontype.md`): NameFuzzy 94.9 [92.0-97.3] for `initial_last` to 100.0 for `last_only`.

### 2.4 Preprocessing variants (120-document sample, 208 pages, 200 DPI, psm 3, single language, no intervals)

| preprocessing | CER clean | CER medium | CER hard | NameFuzzy clean | NameFuzzy medium | NameFuzzy hard | IoU>=.5 hard |
|---|---|---|---|---|---|---|---|
| none (grey) | 2.5 | 6.4 | 63.3 | 98.7 | 97.8 | 32.1 | not tabulated per level |
| median 3x3 | 3.0 | 6.1 | 55.3 | 98.5 | 97.8 | 48.7 | 33.0 |
| Otsu (global) | 2.7 | 6.6 | 62.7 | 98.7 | 97.5 | 33.5 | 21.7 |
| adaptive threshold | 2.5 | 6.9 | 47.1 | 97.9 | 98.0 | 61.2 | 42.7 |
| flat + Otsu | 4.0 | 4.6 | 19.8 | 97.2 | 98.4 | 78.8 | 70.9 |
| **flat (default)** | 3.1 | 3.3 | 16.2 | 98.5 | 99.8 | 88.5 | 81.3 |

Source: `stageA_grid_summary.md` (row "none") and `logs/stageA2_by_noise.md` (other rows; 40 documents per level). Stage A3 tried further flattening variants (`flat2`, `flatg12`, `flatg50`, unsharp mask) on the 40 hard documents; its per-page results are in `ocr/results/stageA3_*.jsonl` but no summary table was kept, so no figure is quoted here.

### 2.5 DPI and page-segmentation mode (120-document sample, single language, flat preprocessing; no intervals)

| dpi / psm | CER all | WER all | NameFuzzy all | IoU>=.5 all | s/page (14 workers) |
|---|---|---|---|---|---|
| 150 / 3 | 9.4 | 18.5 | 94.1 | 90.7 | 1.00 |
| **200 / 3** | **7.6** | **14.8** | **95.9** | **93.3** | 1.25 |
| 300 / 3 | 9.4 | 19.5 | 94.2 | 90.6 | 1.71 |
| 150 / 4 | 15.1 | 25.2 | 93.8 | 90.4 | 0.97 |
| 200 / 4 | 12.4 | 20.7 | 95.5 | 93.3 | 1.19 |
| 300 / 4 | 14.1 | 25.4 | 94.3 | 90.4 | 1.62 |
| 150 / 6 | 12.9 | 26.1 | 95.3 | 92.8 | 1.04 |
| 200 / 6 | 19.6 | 35.6 | 95.0 | 90.1 | 1.28 |
| 300 / 6 | 35.9 | 67.7 | 93.3 | 86.3 | 1.98 |

Source: `stageA_grid_summary.md`, section stageAflat. Without flattening the same grid gives, for psm 3 / single: CER 24.4 (150), 24.4 (200), 24.6 (300) and NameFuzzy 77.2, 78.0, 77.7 (driven by the hard level). With flattening psm 3 has the lowest CER at every DPI; psm 6 at 150 DPI reads names slightly better (NameFuzzy 95.3 vs 94.1) at 3.5 CER points worse and 1.04 vs 1.00 s/page, and at 200 DPI psm 3 is better on every column.

**Native 300-DPI PDFs vs 150-DPI PDFs** (100 documents, 176 pages, single language, psm 3; no intervals; `stageC_300dpi_vs_150.md`). The file has no header row; the nine values of each row are CER, WER, AccErr, NameNorm, NameFuzzy, IoU>=.5, IoU>=.7, adjIoU>=.5 and s/page (14 workers). This order is **inferred** by comparing the 150 / 200 / none and 150 / 200 / flat rows with `final_overall.md` (values agree within the subset effect); it is not stated in the file.

| PDF dpi / render dpi / preprocessing | CER | WER | NameFuzzy | IoU>=.5 | adjIoU>=.5 | s/page |
|---|---|---|---|---|---|---|
| 300 / 300 / flat | 4.0 | 7.8 | 99.4 | 98.5 | 95.2 | 2.29 |
| 300 / 200 / flat | 4.3 | 9.0 | 98.8 | 97.6 | 91.0 | 1.43 |
| 300 / 150 / flat | 5.7 | 13.0 | 98.5 | 96.7 | 85.3 | 1.20 |
| 150 / 300 / flat (upsampled) | 6.3 | 13.7 | 98.1 | 94.6 | 86.1 | 2.41 |
| **150 / 200 / flat (corpus default)** | 5.3 | 10.9 | 98.6 | 97.5 | 89.7 | 1.25 |
| 300 / 200 / none | 11.9 | 15.7 | 93.2 | 89.0 | 84.7 | 1.36 |
| 150 / 200 / none | 16.5 | 23.0 | 85.8 | 81.2 | 70.7 | 1.02 |
| 150 / 150 / none | 15.8 | 20.7 | 85.8 | 82.3 | 78.0 | 0.85 |

What the source supports: rendering a 150-DPI scan at 300 DPI gives no gain (150 / 300 / flat is worse than 150 / 200 / flat on CER and adjusted IoU and costs 1.9 times the time); a genuine 300-DPI scan does improve the result (CER 4.0 at 300 DPI vs 5.3 for 150-DPI input at 200 DPI) at 1.8 times the time. The 300-DPI subset is synthetic and holds only 100 documents.

### 2.6 Threads and timing (sequential, 30 first pages = 2 documents per language x noise level; `timing_summary.md`)

| dpi / pre / psm / language | OMP_THREAD_LIMIT | t_ocr mean / median / p95 (s) | render + prep mean (s) | tesseract peak RSS MB (mean / max) |
|---|---|---|---|---|
| 150 / flat / 3 / single | 1 | 0.75 / 0.74 / 0.94 | 0.08 | 44 / 49 |
| **200 / flat / 3 / single** | 1 | **0.87 / 0.86 / 1.08** | **0.15** | 51 / 55 |
| 300 / flat / 3 / single | 1 | 1.14 / 1.12 / 1.38 | 0.35 | 67 / 71 |
| 150 / flat / 3 / all | 1 | 1.57 / 1.53 / 2.13 | 0.08 | 108 / 123 |
| 200 / flat / 3 / all | 1 | 1.65 / 1.59 / 2.25 | 0.15 | 115 / 126 |
| 300 / flat / 3 / all | 1 | 2.21 / 1.91 / 3.65 | 0.35 | 132 / 145 |
| 200 / none / 3 / single | 1 | 0.75 / 0.77 / 1.07 | 0.07 | 52 / 56 |
| 200 / none / 6 / all | 1 | 3.07 / 2.59 / 5.81 | 0.07 | 119 / 140 |

Effect of `OMP_THREAD_LIMIT` on one sequential stream (none / 3 / single, 30 pages, mean t_ocr): at 150 DPI unset (all 16 logical processors) 0.55 s, 1 -> 0.64 s, 4 -> 0.56 s; at 300 DPI 0.97 s, 1.06 s, 0.96 s. Throughput on 54 pages (200 / flat / 3 / single): 16 processes x `OMP_THREAD_LIMIT=1`: 7.9 s wall (mean per-page t_ocr 1.46 s); 4 processes x 4 threads: 15.2 s (0.88 s); 1 process, variable unset: 43.7 s (0.66 s); 16 processes with the variable unset (oversubscription): 756.7 s (mean per-page t_ocr 220.99 s). Hence one thread per Tesseract process and parallelism across pages. Under 16 busy workers on this 8-core / 16-thread host the per-page OCR time rose from 0.87 s (sequential) to 1.46 s (different page sets; indicative only).

### 2.7 Language handling

- Telling Tesseract the true language ("single") versus all five: AccErr 6.4 [5.6-7.3] vs 19.5 [18.3-20.8], NameExact 90.5 vs 86.1, WER 11.0 vs 12.3, OCR time 0.87 s vs 1.65 s per page sequential (section 2.1, 2.6). NameFuzzy and IoU are essentially equal (97.6 vs 97.8, 96.2 vs 96.5): the combined model keeps names readable but loses accents.
- Language detection by a small stop-word vote on the output of the combined pass identified the document language of 600/600 documents (100.0 % in every language and noise level; `logs/lang_detect.txt`); the detected-language map therefore differs from the truth for no document (`lang_langmap.json` is empty), so the "two-pass" strategy (pass 1 combined model to detect, pass 2 single-language model) reproduces the "single" figures at the cost of both passes. The end-to-end cost of two passes was not measured as a unit.
- The NER, embedding and fuzzy benchmarks (sections 4-8) all used the single-correct-language OCR output.

**Reproduce** (Git Bash on Windows + Docker Desktop; run from `benchmarks/ocr/`, corpus generated into `benchmarks/data` first; set `BENCH_DATA` to the generated corpus folder before running `dk.sh`, see `benchmarks/README.md`):

```
bash run_all.sh                                   # whole OCR benchmark (builds image ocrbench:1; timing step needs an idle machine, ~30 min)
./dk.sh python -I scripts/report.py --tag stageB --by lang,noise --metrics "CER,WER,AccErr,NameExact,NameNorm,NameFuzzy,IoU>=.5,IoU>=.7,meanIoU,adjIoU>=.5,adjIoU>=.7"
bash scripts/timing.sh && ./dk.sh python -I scripts/timing_summary.py
```

---

## 3. NER on public data

**What was measured.** Exact-span PERSON F1 of 13 off-the-shelf NER models on public test sets in the five languages. Leading honorifics (Mr, Dr, Sr, Sra, Herr, Frau, Mme, M., ...) are stripped from both gold and predicted spans; all other entity types are mapped to O. Test splits capped at 5,000 sentences (seed 0): WikiANN (5 languages), MultiNERD (5), WikiNEuRal (5), CoNLL-2003 (en), CoNLL-2002 (es), Few-NERD coarse person (en), GermEval-2014 outer layer (de). Source: `bench/ner_public/results_tables.md`, `scripts/{prepare_data,models,common,evaluate,aggregate,cpu_bench}.py`.

**Versions.** Python 3.12.10 environment `ner_public/venv`: torch 2.14.1+cu126, transformers 5.16.1, tokenizers 0.23.3, safetensors 0.8.0, spaCy 3.8.18 (`xx_ent_wiki_sm` 3.8.0, `en_core_web_sm` 3.8.0, `en/es/pt/fr/de` `_core_*_md` 3.8.0), gliner 0.2.29, gliner2 2.1.0, datasets 5.1.0, numpy 2.5.4, pandas 3.1.0. Hugging Face snapshots: `Davlan/xlm-roberta-base-ner-hrl` 253f557bd8249b8515114cfd7f71974fe5fa4d2f, `Davlan/bert-base-multilingual-cased-ner-hrl` e756de7f7b8f64fea0c3d7c3872f1322fab747b1, `Davlan/distilbert-base-multilingual-cased-ner-hrl` d421f57d5b1d36b375408588669e9340f9b11a89, `Babelscape/wikineural-multilingual-ner` bed6ee7a45d2827b6c90a4fd7983f0241ae0a5c1. Transformers used through `pipeline("token-classification", aggregation_strategy="first")`; GLiNER with label "person", threshold 0.5. GPU for accuracy, CPU (4 threads) for latency.

**Pooled micro exact F1, "common_clean" pool** (WikiANN + Few-NERD + GermEval-2014; the pool without known training overlap for the models below; en = WikiANN + Few-NERD, es/pt/fr = WikiANN, de = WikiANN + GermEval):

| model | en | es | pt | fr | de |
|---|---|---|---|---|---|
| **davlan_xlmr_base_hrl** | 77.0 [75.8-78.2] | 73.4 [71.6-75.2] | 66.2 [64.2-68.3] | 74.1 [72.0-76.0] | 82.1 [80.8-83.4] |
| davlan_distilbert_hrl | 77.4 [76.2-78.5] | 76.8 [75.0-78.6] | 71.0 [69.2-72.8] | 77.8 [76.0-79.4] | 84.1 [82.9-85.3] |
| davlan_mbert_hrl | 78.7 [77.6-79.9] | 77.0 [75.2-78.7] | 73.3 [71.5-75.2] | 79.9 [78.2-81.6] | 86.0 [84.9-87.1] |
| wikineural_mbert | 81.3 [80.2-82.3] | 86.0 [84.5-87.3] | 84.5 [83.0-86.0] | 88.7 [87.3-89.8] | 86.7 [85.6-87.7] |
| xlmr_large_conll03_en | 79.7 [78.6-80.9] | - | - | - | - |
| gliner_x_large | 72.8 [71.6-74.0] | 74.5 [72.8-76.0] | 71.5 [69.7-73.3] | 77.0 [75.5-78.5] | 77.9 [76.7-79.0] |
| gliner_multi_v2.1 | 68.5 [67.4-69.7] | 81.0 [79.5-82.5] | 74.2 [72.4-75.9] | 79.3 [77.7-80.7] | 69.0 [67.7-70.3] |
| spacy_xx_ent_wiki_sm | 70.5 [69.3-71.7] | 74.4 [72.4-76.2] | 69.0 [67.1-71.0] | 75.6 [73.8-77.3] | 74.9 [73.6-76.3] |
| spacy_router_md | 61.1 [59.6-62.6] | 67.2 [65.1-69.1] | 68.8 [66.7-70.8] | 71.3 [69.4-73.1] | 77.0 [75.7-78.4] |
| spacy_en_core_web_sm | 56.1 [54.7-57.6] | 36.4 [34.3-38.6] | 34.5 [32.4-36.8] | 41.2 [39.1-43.2] | 18.5 [17.5-19.4] |

Remaining models in the source: gliner_x_small (63.6 / 60.2 / 53.3 / 60.9 / 62.1), gliner_x_base (66.2 / 62.5 / 57.4 / 64.5 / 67.8), gliner2.5_multi (64.4 / 58.4 / 53.9 / 60.9 / 59.5). Lenient (overlap) F1 of the default model on this pool: 86.0 [84.9-86.9] en, 86.1 [84.8-87.3] es, 80.4 [78.9-82.1] pt, 82.8 [81.1-84.5] fr, 91.9 [91.2-92.6] de, i.e. most errors are boundary errors. P/R of davlan_xlmr_base_hrl: en 70.9/84.3, es 68.1/79.6, pt 58.6/76.2, fr 67.3/82.5, de 80.4/83.9.

**News sets (CoNLL-2003 en, CoNLL-2002 es, GermEval-2014 de; "L" = the Davlan and xlm-roberta-large-conll03 models were trained on CoNLL):** davlan_xlmr_base_hrl 96.0 [95.1-96.9] / 97.3 [96.4-98.2] / 91.4 [90.1-92.6]; davlan_mbert_hrl 94.1 / 97.0 / 92.3; davlan_distilbert_hrl 95.0 / 96.3 / 90.4; xlmr_large_conll03_en 97.6 [96.9-98.2] (en only); wikineural_mbert 91.9 / 93.6 / 85.6; spacy_xx_ent_wiki_sm 76.0 / 83.0 / 72.3 (en / es / de). The en and es columns are inflated by training overlap (section 10); the benchmark does not flag GermEval-2014 as an overlap for these models.

**Silver Wikipedia sets (MultiNERD + WikiNEuRal, five languages; wikineural_mbert is trained on these, spaCy `xx_ent_wiki_sm` and the router models on WikiNER silver data, so those rows are optimistic):** davlan_xlmr_base_hrl 93.5 / 96.6 / 94.5 / 94.5 / 95.3 (en es pt fr de); davlan_mbert_hrl 95.1 / 98.0 / 95.5 / 96.0 / 96.7; davlan_distilbert_hrl 93.9 / 97.2 / 94.2 / 94.2 / 95.8; wikineural_mbert 97.4 / 99.0 / 97.2 / 97.5 / 99.2 (trained on it); spacy_xx_ent_wiki_sm 84.0 / 92.7 / 90.0 / 92.3 / 89.8 (trained on related data). Intervals: see source.

**CPU (4 threads, no GPU; 500 sentences, batch and single-sentence latency):**

| model | load s | RSS after load MB | peak RSS MB | batched sentences/s | latency p50 / p95 ms | disk MB | licence |
|---|---|---|---|---|---|---|---|
| davlan_distilbert_hrl | 6.7 | 768 | 1054 | 51.4 | 17.2 / 27.9 | 515 | AFL-3.0 |
| davlan_mbert_hrl | 6.7 | 769 | 1222 | 27.0 | 31.9 / 50.2 | 677 | AFL-3.0 |
| **davlan_xlmr_base_hrl** | 7.6 | 1045 | 1517 | 24.2 | 33.6 / 53.8 | 1063 | AFL-3.0 |
| wikineural_mbert | 6.9 | 789 | 1234 | 27.1 | 31.1 / 48.9 | 679 | CC-BY-NC-SA-4.0 |
| xlmr_large_conll03_en | 8.3 | 1072 | 2379 | 7.5 | 108.0 / 170.6 | 2149 | not stated on card |
| spacy_xx_ent_wiki_sm | 1.9 | 850 | 954 | 784.4 | 2.1 / 3.5 | 39 | MIT (data CC-BY-4.0) |
| spacy_router_md | 6.5 | 1907 | 1940 | 362.5 | 4.6 / 7.4 | 279 | MIT en / GPL-3.0 es / CC-BY-SA-4.0 pt / LGPL-LR fr / MIT de |
| gliner_x_large | 19.6 | 4265 | 7563 | 5.0 | 146.8 / 254.8 | 9520 | Apache-2.0 |
| gliner_multi_v2.1 | 11.4 | 2797 | 3876 | 14.6 | 95.1 / 134.1 | 2205 | Apache-2.0 |

(`spacy_en_core_web_sm`, `gliner_x_small`, `gliner_x_base`, `gliner2.5_multi`: see source.)

**Reading.** On sentence-level public data the ranking is wikineural_mbert > Davlan mBERT >= distilBERT >= XLM-R-base on the clean pool, with the GLiNER and spaCy models behind. This ranking did **not** carry over to scanned-page text (section 4), where Davlan XLM-R-base beat mBERT by +2.5 F1 points (paired CI [+1.9, +3.1]). Public-data scores say little about page text with ALL-CAPS names, "Last, First" lists, tables and OCR noise; the corpus benchmark decides.

**Caveats.** Honorific stripping and the PERSON-versus-everything-else mapping are the benchmark's own choices. Few-NERD merges adjacent persons (no B/I distinction in the coarse mapping). Scores on WikiANN are markedly lower than on the other sets for every model (for example davlan_xlmr_base_hrl 66.2-75.9). Results are sentence-level; the end-to-end run uses whole pages.

**Reproduce:** `benchmarks/ner_public/scripts/prepare_data.py <data dir>`, `evaluate.py <model>` for each model (`run_all.sh` for the GLiNER set), `aggregate.py` for the tables, `cpu_bench.py <model>` per model (`run_cpu.sh`). Run with `python -I -X utf8`.

---

## 4. NER end to end on OCR text

**What was measured.** Tesseract words (200 DPI + flattening, psm 3, single correct language: `ocr/results/stageB_flat200.jsonl`) -> text units -> NER -> post-processing rules -> union box of the OCR words under each predicted span (pixels to points with the real raster size, section 1) -> match to the 7,489 truth mentions of the same page at IoU >= 0.5 (greedy one-to-one). Precision counts every unmatched prediction as a false positive; the 2,425 traps are never positives ("trapFP" = share of traps hit by at least one predicted span). Source: `bench/ner_corpus/results/e1..e6`, `README.md`, `scripts/`.

**Inputs ("variants").** `line` = each Tesseract line separately; `par` = each Tesseract paragraph; `page` = whole page text with line breaks; `pagesp` = whole page with line breaks replaced by spaces; suffix `_tc` = ALL-CAPS words title-cased before NER (offsets preserved); `_dn` = dashes and quotes folded. Transformer models: Hugging Face `aggregation_strategy="first"`, `stride=64`, batch 32, fp32, GPU for accuracy. spaCy `xx_ent_wiki_sm` on the same units.

### 4.1 Models x inputs (recommended post-processing "REC", 600 documents)

| model | input | precision | recall | F1 | adjF1 | trapFP |
|---|---|---|---|---|---|---|
| **xlmr (Davlan XLM-R-base-hrl)** | **pagesp_tc** | 94.2 [93.3-95.0] | 89.9 [88.7-90.9] | **92.0 [91.2-92.7]** | 90.6 [89.7-91.5] | 7.1 [5.7-8.5] |
| xlmr | pagesp_tc_dn | 94.1 [93.3-95.0] | 89.9 [88.8-91.0] | 92.0 [91.2-92.7] | 90.7 [89.7-91.5] | 7.2 [5.8-8.6] |
| xlmr | par_tc | 93.3 [92.2-94.3] | 89.1 [87.9-90.2] | 91.1 [90.2-91.9] | 89.9 [88.8-90.9] | 7.3 [5.9-8.7] |
| xlmr | line_tc | 92.6 [91.5-93.6] | 88.0 [86.9-89.1] | 90.2 [89.4-91.0] | 89.0 [88.0-90.0] | 8.4 [7.1-9.8] |
| xlmr | pagesp (no title-casing) | 94.6 [93.7-95.4] | 86.6 [85.4-87.8] | 90.4 [89.6-91.2] | 89.2 [88.2-90.2] | 6.6 [5.3-8.0] |
| xlmr | page_tc (newlines kept) | 91.1 [89.9-92.2] | 73.4 [71.9-74.9] | 81.3 [80.2-82.4] | 81.8 [80.7-83.0] | 6.5 [5.3-7.9] |
| mbert (Davlan mBERT-hrl) | par_tc | 91.9 [90.8-93.0] | 87.2 [86.1-88.3] | 89.5 [88.6-90.3] | 88.2 [87.1-89.1] | 9.4 [8.0-11.0] |
| mbert | par (no title-casing) | 91.6 [90.4-92.7] | 78.2 [76.8-79.6] | 84.4 [83.4-85.4] | 83.3 [82.1-84.3] | 9.1 [7.6-10.6] |
| distil (Davlan DistilBERT-hrl) | par_tc | 88.9 [87.6-90.2] | 84.0 [82.7-85.2] | 86.4 [85.4-87.4] | 85.2 [84.1-86.3] | 12.6 [11.0-14.3] |
| distil | line_tc | 89.2 [87.9-90.3] | 84.1 [82.9-85.4] | 86.6 [85.6-87.6] | 85.4 [84.3-86.5] | 11.7 [10.1-13.3] |
| spacy (xx_ent_wiki_sm) | line_tc | 63.6 [61.9-65.2] | 68.3 [66.8-69.7] | 65.9 [64.5-67.1] | 65.0 [63.5-66.3] | 30.8 [28.5-33.2] |

Source: `e1_models_variants.md` (all 4 models x 8+ variants, plus multi-view unions: xlmr `par_tc -> line_tc -> page_tc` 91.6 [90.8-92.5], i.e. no gain over `pagesp_tc`). Title-casing ALL-CAPS words is worth +1.5 [+1.2, +1.9] F1 for XLM-R (paired, `pagesp` -> `pagesp_tc`) and +5.1 F1 points for mBERT (`par` 84.4 -> `par_tc` 89.5, difference of point estimates). The whole page with spaces (`pagesp_tc`) beats paragraph-wise input by +0.9 [+0.4, +1.3] and line-wise by +1.7 [+1.2, +2.2]; keeping raw newlines on a whole page (`page`) collapses recall (73.4 vs 89.9). Folding dashes/quotes (`_dn`) changes nothing (+0.0 [-0.1, +0.1]). Paired differences (`e1b_paired.md`): xlmr `pagesp_tc` - mbert `par_tc` = +2.5 [+1.9, +3.1] overall; per language en +1.5 [+0.6, +2.4], es +3.6 [+1.9, +5.2], pt +4.3 [+2.8, +5.7], fr +1.9 [+0.8, +3.1], de +1.0 [-0.1, +2.1] (not significant); xlmr - spaCy +26.1 [+24.9, +27.3].

### 4.2 Default configuration by language and noise level (xlmr / pagesp_tc / REC; `e3_default_xlmr_pagesp_tc.md`)

| group | docs | mentions | precision | recall | F1 | adjF1 | trapFP |
|---|---|---|---|---|---|---|---|
| **all** | 600 | 7489 | 94.2 [93.3-95.0] | 89.9 [88.7-90.9] | 92.0 [91.2-92.7] | 90.6 [89.7-91.5] | 7.1 [5.7-8.5] |
| en | 120 | 1478 | 98.3 [97.5-99.0] | 93.0 [91.2-94.6] | 95.5 [94.5-96.5] | 93.9 [92.4-95.3] | 0.2 [0.0-0.6] |
| es | 120 | 1505 | 87.7 [84.2-91.0] | 88.4 [85.8-90.6] | 88.1 [85.6-90.3] | 87.0 [84.3-89.3] | 16.2 [11.6-21.1] |
| pt | 120 | 1493 | 94.7 [93.2-96.0] | 91.0 [88.8-92.8] | 92.8 [91.2-94.1] | 90.9 [88.9-92.8] | 3.4 [1.9-5.1] |
| fr | 120 | 1527 | 96.4 [95.3-97.4] | 88.0 [85.1-90.7] | 92.0 [90.2-93.7] | 90.6 [88.3-92.6] | 3.6 [2.0-5.2] |
| de | 120 | 1486 | 94.2 [92.9-95.4] | 89.1 [86.1-91.6] | 91.6 [89.9-93.1] | 91.0 [89.2-92.6] | 12.2 [9.1-15.2] |
| clean | 240 | 3079 | 95.5 [94.3-96.6] | 92.8 [91.7-93.9] | 94.2 [93.2-95.0] | 94.3 [93.3-95.1] | 5.9 [4.1-8.2] |
| medium | 240 | 3064 | 94.3 [92.9-95.5] | 91.3 [90.0-92.6] | 92.8 [91.7-93.7] | 91.2 [90.0-92.2] | 7.4 [5.3-9.8] |
| hard | 120 | 1346 | 90.5 [87.3-93.1] | 79.9 [75.9-83.4] | 84.8 [81.7-87.4] | 80.7 [77.3-84.0] | 8.5 [5.7-11.9] |
| en / clean | 48 | 680 | 98.2 [97.0-99.2] | 94.9 [92.9-96.7] | 96.5 [95.4-97.6] | 96.5 [95.4-97.6] | 0.0 |
| en / medium | 48 | 576 | 98.3 [97.1-99.3] | 91.8 [88.9-94.5] | 95.0 [93.2-96.6] | 93.5 [91.4-95.5] | 0.5 [0.0-1.6] |
| en / hard | 24 | 222 | 98.5 [96.6-100.0] | 90.1 [85.3-94.9] | 94.1 [91.5-96.7] | 87.1 [82.7-91.6] | 0.0 |
| es / clean | 48 | 637 | 91.1 [85.9-95.0] | 91.8 [88.9-94.8] | 91.5 [88.2-94.1] | 91.6 [88.6-93.9] | 13.0 [6.8-21.1] |
| es / medium | 48 | 623 | 88.3 [83.4-92.5] | 89.6 [86.9-92.2] | 88.9 [85.6-91.7] | 87.5 [84.3-90.1] | 16.4 [9.0-24.2] |
| es / hard | 24 | 245 | 77.4 [66.5-87.1] | 76.7 [68.2-84.8] | 77.0 [68.6-84.6] | 73.4 [64.9-81.7] | 22.0 [12.9-31.3] |
| pt / clean | 48 | 539 | 95.5 [93.0-97.5] | 94.1 [92.0-96.1] | 94.8 [93.0-96.3] | 94.8 [93.0-96.4] | 3.1 [0.5-6.3] |
| pt / medium | 48 | 629 | 94.6 [92.4-96.4] | 91.6 [88.2-94.3] | 93.1 [90.7-95.0] | 90.8 [88.0-93.1] | 2.1 [0.5-4.2] |
| pt / hard | 24 | 325 | 93.5 [89.4-96.3] | 84.6 [77.4-89.3] | 88.9 [83.4-92.1] | 84.7 [77.0-90.1] | 6.7 [2.9-11.1] |
| fr / clean | 48 | 578 | 98.0 [96.7-99.0] | 91.5 [88.7-94.3] | 94.6 [92.9-96.2] | 95.0 [93.2-96.7] | 1.7 [0.0-3.6] |
| fr / medium | 48 | 708 | 96.4 [94.9-97.7] | 90.8 [87.8-93.6] | 93.5 [91.7-95.1] | 92.2 [90.5-93.9] | 4.7 [2.1-7.9] |
| fr / hard | 24 | 241 | 92.0 [88.0-95.8] | 71.4 [63.0-81.1] | 80.4 [74.0-86.8] | 73.8 [67.1-82.1] | 4.8 [0.0-10.7] |
| de / clean | 48 | 645 | 95.2 [93.6-96.6] | 91.8 [89.3-93.9] | 93.4 [91.8-95.0] | 93.4 [91.6-95.0] | 11.8 [7.5-16.8] |
| de / medium | 48 | 528 | 94.3 [92.4-95.9] | 93.2 [90.5-95.6] | 93.7 [92.0-95.4] | 92.2 [89.8-94.4] | 14.4 [9.1-20.6] |
| de / hard | 24 | 313 | 92.0 [87.6-95.4] | 76.7 [65.1-85.3] | 83.6 [75.8-89.0] | 83.3 [75.9-88.7] | 8.8 [2.9-16.3] |

More breakdowns in the same file: by document type (F1 85.3 contract and 86.0 kyc are the lowest, 94.2 memo and 94.1 minutes the highest), by number of pages (94.2 / 91.6 / 91.8 for 1 / 2 / 3 pages). Recall by mention type (all languages, IoU >= 0.5): full 90.8 [89.3-92.4], title+full 97.8 [97.0-98.5], title+last 96.0 [94.1-97.9], "Last, First" 79.9 [76.4-82.9], ALL-CAPS 90.6 [88.3-92.8], initial+last 88.5 [84.7-92.2], **last only 20.3 [13.3-28.3]** (n = 133; a bare surname is not recovered as a person box in most cases), mentions in the page footer 80.3 [74.6-86.3]. Trap false-positive rate by kind: organisation 11.4 [8.5-14.5], person-like organisation 10.1 [7.0-13.5], street 4.2 [2.5-6.3], product / place / e-mail 0.0; concentrated by language (Spanish person-like organisations 37.2 [26.6-47.5], German organisations 44.4 [34.8-53.5]). Of the 417 false positives, 185 are on traps, 93 are boundary errors on real mentions, 139 are spurious. Distinct-person recall (a person found at least once, 3,373 persons): 93.7 [92.6-94.7] by box match; 88.4 [86.9-89.8] when a matched mention reads as the full name (`token_sort >= 90`). NER recall as a function of OCR quality of the mention: OCR exact (6,780 mentions) 92.3 %, normalised-equal (386) 90.7 %, fuzzy-only (145) 66.2 %, not found by OCR (178) 14.0 %.

### 4.3 Span post-processing ablation (xlmr / pagesp_tc; `e2_ablation_xlmr_pagesp_tc.md`)

Rules: T title/honorific stripping; M1 hyphen/apostrophe merge; M1p particle bridge (de, van, von, ...); M2 "Last, First" merge (strict: 1-token surname; relaxed: up to 4-token surname); M3 "F. Last" initial extension; Dsingle drop single-token spans unless titled; Dlower drop all-lowercase spans; Ddigit drop spans with digits or "@"; Dshort drop spans with fewer than 3 letters; Xsplit split spans at line breaks; Xmerge merge single-token spans on consecutive lines; Btrim proportional box trim; Dtrap drop spans next to legal-form/street words (generic lexicon, exploratory); Rlf regex rescue of missed "Last, First" pairs (exploratory); minscore model-score floor. BASE = raw spans snapped to OCR words. FULL = T, M1, M1p, M2 relaxed, M3, Dsingle (titled), Dlower, Ddigit, Dshort, Btrim. **REC (recommended) = FULL - M1p - Btrim + Xsplit**, i.e. T, M1, M2 (relaxed), M3, Dsingle (titled), Dlower, Ddigit, Dshort, Xsplit.

Add one rule to BASE (F1 %, change vs BASE with paired CI, `*` = CI excludes 0):

| rule | precision | recall | F1 | dF1 | trapFP |
|---|---|---|---|---|---|
| BASE | 70.2 [68.6-71.7] | 81.4 [80.3-82.6] | 75.4 [74.1-76.7] | 0 | 13.5 [11.7-15.5] |
| M2 relaxed "Last, First" | 85.3 [83.8-86.7] | 90.2 [89.0-91.2] | 87.7 [86.5-88.7] | +12.3 [+11.3, +13.3] * | 13.5 |
| M2 strict | 83.3 [81.7-84.8] | 89.5 [88.3-90.6] | 86.3 [85.1-87.4] | +10.9 [+9.9, +11.9] * | 13.5 |
| Dsingle (keep if titled) | 91.0 [89.9-92.1] | 79.7 [78.6-80.9] | 85.0 [84.1-85.8] | +9.6 [+8.8, +10.3] * | 7.3 [5.9-8.7] |
| Dsingle (drop all) | 90.6 [89.4-91.7] | 72.6 [71.4-73.9] | 80.6 [79.7-81.5] | +5.2 [+4.3, +6.1] * | 6.7 [5.3-8.2] |
| Xsplit | 70.1 [68.5-71.6] | 82.3 [81.2-83.5] | 75.7 [74.4-77.0] | +0.3 [+0.2, +0.5] * | 13.6 |
| T | 70.4 | 81.5 | 75.5 [74.3-76.8] | +0.1 [-0.0, +0.2] | 13.5 |
| M1, M3, Xmerge | | | 75.4 | +0.0 | 13.5 |
| Dlower / Ddigit / Dshort | | | 75.6 / 75.5 / 75.5 | +0.2 / +0.1 / +0.1 * | 12.4 / 12.6 / 13.5 |
| M1p particles | 70.0 | 81.1 | 75.2 [73.9-76.4] | -0.2 [-0.4, -0.1] * | 13.5 |
| Btrim (box trim) | 69.7 | 80.8 | 74.8 [73.6-76.1] | -0.6 [-0.7, -0.4] * | 13.5 |
| Dtrap (exploratory) | 71.7 | 81.1 | 76.1 [74.9-77.3] | +0.7 [+0.5, +0.9] * | 10.8 [9.3-12.5] |
| Rlf (exploratory) | 68.8 | 81.6 | 74.7 [73.4-75.9] | -0.8 [-0.9, -0.6] * | 13.7 |
| minscore >= 0.5 / 0.8 / 0.95 | | | 75.6 / 76.6 / 77.1 | +0.2 / +1.2 / +1.7 * | 13.4 / 10.6 / 7.4 |
| FULL | 92.5 [91.6-93.5] | 88.3 [87.2-89.4] | 90.4 [89.6-91.2] | +15.0 [+13.9, +16.1] * | 7.4 [6.1-8.9] |

Leave one rule out of REC, and add rejected ones back (F1 and change vs REC):

| variant | precision | recall | F1 | dF1 vs REC | trapFP |
|---|---|---|---|---|---|
| REC | 94.2 [93.3-95.0] | 89.9 [88.7-90.9] | 92.0 [91.2-92.7] | 0 | 7.1 [5.7-8.5] |
| REC - M2 | 92.2 [91.1-93.2] | 80.5 [79.4-81.7] | 85.9 [85.1-86.8] | -6.0 [-6.6, -5.5] * | 7.1 |
| REC - Dsingle | 86.2 [84.7-87.5] | 91.3 [90.2-92.3] | 88.7 [87.6-89.7] | -3.3 [-3.9, -2.8] * | 12.5 [10.7-14.2] |
| REC - Xsplit | 93.2 [92.3-94.1] | 88.8 [87.6-89.8] | 90.9 [90.1-91.7] | -1.0 [-1.3, -0.8] * | 7.0 |
| REC - T / M1 / M3 / Dlower / Ddigit / Dshort | | | 91.9 / 92.0 / 92.0 / 92.0 / 92.0 / 92.0 | -0.1 [-0.2, +0.1] / 0.0 / -0.0 / -0.0 / +0.0 / +0.0 (none significant) | 7.1 |
| REC + M1p (particles) | 93.5 | 89.5 | 91.5 [90.7-92.2] | -0.5 [-0.7, -0.3] * | 7.5 |
| REC + Btrim | 94.1 | 89.8 | 91.9 [91.1-92.6] | -0.1 [-0.2, +0.0] | 7.1 |
| REC + Dtrap | 95.2 [94.4-95.8] | 89.5 [88.4-90.5] | 92.3 [91.5-92.9] | +0.3 [+0.1, +0.5] * | 5.2 [4.1-6.4] |
| REC + Rlf | 91.8 | 89.8 | 90.8 [89.9-91.6] | -1.2 [-1.5, -0.9] * | 7.3 |
| REC + Xmerge | 94.1 | 89.8 | 91.9 [91.1-92.6] | -0.1 [-0.1, -0.0] * | 7.1 |
| REC with M2 strict | 93.1 | 89.1 | 91.1 [90.3-91.9] | -0.9 [-1.2, -0.6] * | 7.1 |
| REC with Dsingle = drop all | 94.1 [93.1-95.0] | 82.7 [81.5-83.9] | 88.0 [87.2-88.8] | -3.9 [-4.4, -3.6] * | 6.5 |
| REC + minscore 0.5 / 0.8 | | | 91.9 / 91.0 | -0.1 / -1.0 * | 7.1 / 6.2 |

False-positive anatomy (600 documents): BASE 8,683 predicted spans, 6,097 true positives, 411 false positives on traps, 1,785 boundary errors, 390 spurious; FULL 7,149 spans, 6,614 TP, 196 / 146 / 193; **REC 7,148 spans, 6,731 TP, 185 / 93 / 139**. The rules that matter are M2 ("Last, First", +12.3 F1 alone), Dsingle (single-token spans are mostly false positives, +9.6 alone) and Xsplit; the rest are neutral. Box trimming did not help (design D4 keeps the whole word box). Dtrap (+0.3 F1, trapFP 5.2) is labelled generic-lexicon and exploratory in the source and is not part of REC.

### 4.4 Controls and latency

**NER on clean ground-truth text (no OCR), REC rules; text-level matching (word-set Jaccard >= 0.5), comparable in trend, not identical in definition** (`e4_control_truth_text.md`): xlmr `truthpagesp_tc` P 96.8 [95.9-97.5] R 95.4 [94.8-96.0] F1 96.1 [95.6-96.6]; xlmr `truthline_tc` 95.6 [95.1-96.0]; mbert `truthline_tc` 94.8 [94.2-95.4]; spaCy `truthline_tc` 81.9 [80.8-82.9]. Without title-casing, mbert `truthline` is 89.5 [88.7-90.2] (recall 84.8 vs 94.6 with title-casing). Read with the definitional caveat above, OCR and box construction cost the default model roughly 4 F1 points (92.0 end to end vs 96.1 on truth text). Per language on truth text (xlmr `truthline_tc`): en 97.7 [97.1-98.3], es 92.0 [90.0-93.6], pt 97.0 [96.1-97.7], fr 95.4 [94.5-96.3], de 96.0 [95.3-96.8]; the Spanish deficit comes from precision (88.9) and is present without OCR.

**CPU latency, per page of real OCR text** (4 threads, fresh process per model, fp32, after 2 warm-up pages, 8 pages per length bucket; `e6_cpu_speed_*.json`; median / mean ms; page = whole page in one call; OCR page length median 1,765 chars, mean 1,763):

| model | load s | RSS after load MB | peak RSS MB | 300-800 chars (mean 534) | 800-1500 (1207) | 1500-2200 (1881) | 2200-3000 (2420) |
|---|---|---|---|---|---|---|---|
| **xlmr, page** | 8.44 | 1006 | 1652 | 113.1 / 117.2 | 237.7 / 248.1 | **373.8 / 416.7** | 821.8 / 823.1 |
| xlmr, par | | | | 191.5 / 265.0 | 723.0 / 935.1 | 723.6 / 748.1 | 1008.8 / 1032.5 |
| xlmr, line | | | | 224.5 / 218.7 | 416.4 / 415.6 | 724.3 / 698.1 | 815.1 / 806.2 |
| mbert, page | 7.62 | 731 | 1369 | 106.0 / 112.1 | 225.3 / 225.6 | 377.8 / 528.3 | 820.0 / 810.7 |
| distil, page | 7.54 | 728 | 1232 | 55.7 / 54.6 | 117.0 / 115.8 | 159.7 / 224.7 | 371.0 / 364.0 |
| spacy, page | 3.36 | 840 | 953 | 7.2 / 7.1 | 13.7 / 13.5 | 21.0 / 20.7 | 26.9 / 25.9 |

For XLM-R the recommended whole-page input is the fastest or equal-fastest of the three input modes at every length bucket (paragraph-wise median latency is 1.2 to 3.0 times the whole-page one in the four buckets; line-wise is equal at 2200-3000 characters, 815.1 vs 821.8 ms). Latency grows faster than linearly with page length (the 2200-3000 bucket is 2.2 times the 1500-2200 bucket for 1.3 times the characters).

**Reproduce** (from `benchmarks/ner_corpus/`, using the `ner_public` environment; rapidfuzz 3.14.6 in `pylib/`): `bash scripts/reproduce.sh` (raw predictions `scripts/run_ner.py MODEL VARIANT` ~20 min GPU, evaluation ~5 min, CPU timing ~10 min on RTX 4070 Ti + 16 cores); evaluation only: `bash scripts/run_eval_all.sh`; CPU timing only: `bash scripts/run_speed.sh`. Always `python -I -X utf8`.

---

## 5. Fuzzy threshold and scorer comparison

**What was measured.** For each of the 200 name lists (1,687 queries: 794 positives, 893 negatives) the query string `"first last"` is compared with every extracted name of the document, both normalised with NFKD accent strip + casefold (apostrophes removed, other punctuation to space); the score is `rapidfuzz` scorer / 100, best over candidates; match if score >= threshold. Positives are exact 432, case 181, accent 107, OCR-typo 74; negatives are decoy 493 (name absent, surname also absent) and near-miss decoy 400 (same surname as a person in the document, different first name). 42 positives have no full-name mention in the document (only surname or title + surname) and cannot match `"first last"` by design; recall is reported with and without them. Two settings: ORACLE (a perfect extractor: every printed mention) and CHAIN (default NER on OCR text, section 4). RapidFuzz 3.14.6. Source: `ner_corpus/results/e5_fuzzy.md`, `scripts/{fuzzy,e5_fuzzy}.py`.

### 5.1 Threshold sweep, `token_sort_ratio`

| thr | ORACLE precision | ORACLE recall (all positives) | ORACLE recall (with full-name mention) | ORACLE F1 | ORACLE near-miss FP | CHAIN precision | CHAIN recall (all) | CHAIN recall (with full name) | CHAIN F1 | CHAIN near-miss FP |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.80 | 94.4 | 95.8 | 99.7 | 95.1 | 45 / 400 | 94.5 | 90.3 | 94.0 | 92.3 | 42 / 400 |
| 0.85 | 99.0 | 95.1 | 99.7 | 97.0 | 8 / 400 | 99.0 | 89.2 | 93.8 | 93.8 | 7 / 400 |
| 0.88 | 100.0 | 94.6 | 99.5 | 97.2 | 0 / 400 | 100.0 | 88.2 | 92.8 | 93.7 | 0 / 400 |
| **0.90** | **100.0** | **94.0** | 98.9 | 96.9 | **0 / 400** | **100.0** | **87.5** | 92.2 | 93.4 | **0 / 400** |
| 0.92 | 100.0 | 92.3 | 97.3 | 96.0 | 0 / 400 | 100.0 | 85.8 | 90.4 | 92.3 | 0 / 400 |
| 0.95 | 100.0 | 87.9 | 92.8 | 93.6 | 0 / 400 | 100.0 | 81.2 | 85.8 | 89.6 | 0 / 400 |

Decoy false positives are 0 / 493 at every threshold from 0.80 to 0.98. Bootstrap 95 % CI over the 200 documents, CHAIN, `token_sort_ratio`: at 0.85 precision 99.0 [98.3-99.6], recall 89.2 [86.5-91.6], F1 93.8 [92.4-95.2], false-positive rate on negatives 0.8 [0.3-1.4]; at **0.90** precision 100.0 [100.0-100.0], recall **87.5 [84.8-90.2]**, F1 93.4 [91.8-94.8], FP rate 0.0 [0.0-0.0]; at 0.92 recall 85.8 [83.1-88.5]; at 0.95 recall 81.2 [78.2-84.1]. ORACLE at 0.90: recall 94.0 [92.0-95.6], F1 96.9 [95.8-97.8]. The best F1 threshold is 0.88 for ORACLE (97.2) and 0.84 for CHAIN (93.9 vs 93.4 at 0.90, i.e. 0.5 F1 points); the chosen 0.90 lies above the highest near-miss decoy score (0.878; 99th percentile 0.863 ORACLE / 0.857 CHAIN; 95th 0.824 / 0.821; median 0.667 / 0.659). **Caveat (design risk note):** the generated near-miss decoys are separated from positives by a wider margin than real-world look-alikes (for example "Jon Smith" vs "John Smith" scores about 0.95 by the design's estimate; not measured on the corpus), so real precision may be lower than the measured 100 %. Positives with a full-name mention score above 0.933 at the 5th percentile and 0.895 at the 1st (ORACLE); in CHAIN the 5th percentile is 0.633 because extraction errors are included.

### 5.2 Scorer comparison (rapidfuzz, threshold 0.90 and best threshold in 0.80-0.98)

| scorer | ORACLE P / R / F1 at 0.90 | ORACLE FP at 0.90 | ORACLE best thr (F1) | CHAIN P / R / F1 at 0.90 | CHAIN FP at 0.90 | CHAIN best thr (F1) |
|---|---|---|---|---|---|---|
| **token_sort_ratio** | **100.0 / 94.0 / 96.9** | **0** | 0.88 (97.2) | **100.0 / 87.5 / 93.4** | **0** | 0.84 (93.9) |
| ratio | 100.0 / 84.6 / 91.7 | 0 | 0.88 (92.0) | 100.0 / 80.6 / 89.3 | 0 | 0.86 (89.6) |
| WRatio | 91.1 / 96.2 / 93.6 | 75 | 0.91 (95.8) | 91.6 / 90.2 / 90.9 | 66 | 0.91 (92.6) |
| token_set_ratio | 90.7 / 97.4 / 93.9 | 79 | 0.90 (93.9) | 91.0 / 90.9 / 91.0 | 71 | 0.85 (91.0) |
| partial_ratio | 86.3 / 91.1 / 88.6 | 115 | 0.92 (88.7) | 86.8 / 87.2 / 87.0 | 105 | 0.90 (87.0) |

`WRatio`, `token_set_ratio` and `partial_ratio` match a requested name as a substring or token subset of a longer name (e.g. "Heinz Mueller" inside "Karl-Heinz Mueller"), producing 75-115 false positives at 0.90 in the oracle setting (66-105 in the chain); plain `ratio` is order-sensitive and loses "Last, First" (recall 84.6 vs 94.0 oracle). The default NER model is not critical to the fuzzy step: with the mBERT chain the same threshold gives P 100.0, R 87.9, F1 93.6; with the spaCy chain R 75.6, F1 86.1.

### 5.3 CHAIN at 0.90, by query type, language and noise

| query type | correct % (positives: matched and located on the right person; negatives: rejected) |
|---|---|
| exact (432) | 89.1 |
| case (181) | 90.6 |
| accent (107) | 84.1 |
| ocr_typo (74) | 68.9 |
| decoy (493) | 100.0 |
| near_miss_decoy (400) | 100.0 |

| lang | precision | recall | positives matched + located | negatives rejected | overall correct |
|---|---|---|---|---|---|
| en | 100.0 | 90.0 | 90.0 | 100.0 | 95.2 |
| es | 100.0 | 84.5 | 83.9 | 100.0 | 92.2 |
| pt | 100.0 | 88.4 | 88.4 | 100.0 | 94.7 |
| fr | 100.0 | 88.9 | 86.9 | 100.0 | 94.0 |
| de | 100.0 | 86.1 | 85.5 | 100.0 | 93.1 |

By noise at 0.90 (CHAIN recall; precision 100 everywhere): clean 89.0, medium 89.6, hard 76.4 (ORACLE: 92.0 / 95.1 / 96.4; with a perfect extractor the hard level is not lower, so the CHAIN drop comes from extraction on degraded pages, not from the matching). Overall at 0.90 the positive is matched and located on the correct mention in 86.9 % of cases (91.5 % among positives with a full-name mention) and the overall correct decision + location rate is 93.8 %. Recall by query type at 0.90 (ORACLE / CHAIN): exact 96.1 / 89.6, case 96.1 / 90.6, accent 91.6 / 86.0, OCR-typo 79.7 / 70.3. The OCR-typo variants are rule-based approximations of Tesseract errors.

**Reproduce:** `python -I -X utf8 scripts/e5_fuzzy.py default_xlmr_pagesp_tc mbert_par_tc spacy_line_tc` in `benchmarks/ner_corpus/` (part of `run_eval_chain.sh`).

---

## 6. Embedders

Two independent measurements. (A) public multilingual retrieval sets (model selection, CPU cost); (B) the synthetic corpus on real OCR text (the decision that matters).

### 6.1 Public data

**Data.** MMARCO dev (Google-translated MS MARCO; 1,600 queries per language of which 400 are held out with their gold passages removed, 1,200 answerable), WebFAQ retrieval test (2,000 per language: 1,500 answerable + 500 held out), MIRACL hard-negative dev (en, es, fr, de only; 799 / 648 / 343 / 305 queries); the candidate index holds 40,000 passages per language and dataset (all gold kept, the rest sampled, seed 13). Metrics nDCG@10 and recall@k with bootstrap over queries. Source: `C:/eb/out/tables/retrieval_tables.md`, `results_long.csv`, `calibration.md`, `cpu_table.md`; scripts `emb_public/src/`.

**Models and settings.** Sentence-Transformers; fp16 on GPU for retrieval (harrier fp32), fp32 on CPU for latency; `query: `/`passage: ` prefixes for e5, `query: ` only for arctic, none for bge-m3, MiniLM, gte; max sequence 512. Two environments were used (versions read from the installed packages, not from a result file): `C:/eb/.venv` (Python 3.12.10, torch 2.14.1+cu126, transformers 5.21.0, sentence-transformers 6.2.0, numpy 2.5.4, pandas 3.1.0) for most models and `C:/eb/.venv2` (transformers 4.57.6, sentence-transformers 5.7.0, torch 2.14.1+cu126) for the two remote-code models `gte-multilingual-base` and `snowflake-arctic-embed-m-v2.0`. Model revisions of the public run are not recorded (those of the three corpus models are, section 6.2).

**Macro nDCG@10 over MMARCO + WebFAQ + MIRACL (per language, 95 % CI over queries):**

| model | en | es | pt | fr | de | mean of 5 | mean R@10 |
|---|---|---|---|---|---|---|---|
| arctic-m-v2 | 86.0 [85.2-86.8] | 82.3 [81.4-83.2] | 84.5 [83.4-85.5] | 79.3 [77.9-80.5] | 79.1 [77.7-80.4] | 82.2 | 91.4 |
| bge-m3 | 84.3 [83.5-85.2] | 82.0 [81.0-82.8] | 83.6 [82.4-84.7] | 79.2 [77.8-80.3] | 77.6 [76.2-78.9] | 81.3 | 91.2 |
| **e5-base** | 83.0 [82.2-83.9] | 80.4 [79.5-81.4] | 82.0 [80.9-83.2] | 75.4 [73.9-76.7] | 75.4 [73.8-76.7] | **79.3** | 89.4 |
| granite-311m-r2 | 80.9 [80.0-81.8] | 78.0 [77.0-79.0] | 81.4 [80.1-82.6] | 74.1 [72.7-75.5] | 72.9 [71.4-74.3] | 77.5 | 87.8 |
| gte-mlg-base | 83.6 [82.8-84.5] | 78.5 [77.5-79.5] | 78.6 [77.4-79.9] | 75.2 [73.8-76.6] | 70.9 [69.4-72.4] | 77.4 | 88.1 |
| harrier-270m | 83.6 [82.7-84.4] | 78.8 [77.9-79.7] | 79.5 [78.3-80.7] | 73.0 [71.6-74.2] | 70.4 [68.9-71.8] | 77.0 | 88.3 |
| e5-small | 81.4 [80.5-82.4] | 79.0 [78.1-80.0] | 77.8 [76.6-79.0] | 73.1 [71.8-74.4] | 73.3 [71.8-74.7] | 76.9 | 87.6 |
| granite-97m-r2 | 79.1 [78.0-80.1] | 75.1 [74.1-76.1] | 77.5 [76.2-78.7] | 71.9 [70.5-73.2] | 69.0 [67.4-70.5] | 74.5 | 85.7 |
| minilm-l12 | 71.5 [70.4-72.6] | 56.2 [54.9-57.5] | 58.9 [57.4-60.4] | 48.9 [47.4-50.4] | 47.8 [46.2-49.3] | 56.6 | 69.5 |

Paired macro nDCG@10 difference vs e5-small (points): e5-base +1.6 [+1.0, +2.2] en, +1.4 [+0.7, +2.0] es, +4.3 [+3.4, +5.1] pt, +2.3 [+1.4, +3.4] fr, +2.1 [+1.2, +3.0] de; arctic-m-v2 +4.5 / +3.2 / +6.7 / +6.2 / +5.8 (all significant); bge-m3 +2.9 / +3.0 / +5.8 / +6.1 / +4.3. The dataset-level tables (mmarco, webfaq, miracl; nDCG@10, R@1, R@10) are in the source.

**CPU cost (4 threads, fp32, batch 16, median of 5 timed runs; peak RSS includes about 750 MB of torch baseline; `cpu_table.md`, `cpu_bench.jsonl`):**

| model | params M | disk MB | load s | peak RSS MB | ms/passage @~120 tok | ms/passage @~400 tok | licence |
|---|---|---|---|---|---|---|---|
| e5-small | 118 | 471 | 2.41 | 1843 | 42.2 | 126.1 | MIT |
| **e5-base** | 278 | 1112 | 2.71 | 2575 | **106.2** (measured 122 tokens) | **356.2** (measured 402 tokens) | MIT |
| arctic-m-v2 | 305 | 1221 | 0.5 | 2059 | 94.4 | 329.1 | Apache-2.0 (remote code) |
| gte-mlg-base | 305 | 611 | 1.27 | 2815 | 93.8 | 330.2 | Apache-2.0 (remote code) |
| bge-m3 | 568 | 2271 | 2.09 | 2796 | 246.5 | 834.6 | MIT |
| granite-97m-r2 | 97 | 195 | 1.08 | 1565 | 51.2 | 171.8 | Apache-2.0 |
| granite-311m-r2 | 312 | 623 | 2.01 | 2436 | 164.7 | 516.2 | Apache-2.0 |
| harrier-270m | 268 | 536 | 2.38 | 2462 | 183.7 | 557.3 | MIT |
| minilm-l12 | 118 | 471 | 1.31 | 1615 | 41.9 | 42.0 | Apache-2.0 |

(A first CPU attempt is kept as `cpu_bench_firstattempt_discarded.jsonl` and was discarded; its values differ slightly, for example e5-base 110.1 ms at ~120 tokens. The table of the kept run is quoted.)

**Calibration on public data (cosine of the top-1 hit; A = answerable with gold in top-10, H = held-out queries whose gold passage was removed, O = off-topic queries of another dataset).** For e5-base the global (same for all five languages) balanced-accuracy-optimal threshold is **0.86** (balanced accuracy 79.0 %, keeping 86.7 % of A, refusing 54.0 % of H and 88.7 % of O); the per-language optima are 0.860 / 0.862 / 0.866 / 0.868 / 0.862 (en es pt fr de), and a single global threshold loses at most 1.5 points (pt) against per-language ones. The e5-base cosine scores are compressed (answerable median 0.887-0.891, off-topic median 0.832-0.842). For arctic-m-v2 the global optimum is 0.52 and for bge-m3 0.63, for e5-small 0.88: score scales are model specific. This public threshold (0.86) is **not** the corpus threshold (0.808, section 8): thresholds are data-specific.

### 6.2 Synthetic corpus on real OCR text

**Setup.** `intfloat/multilingual-e5-small` (revision 614241f622f53c4eeff9890bdc4f31cfecc418b3, weights sha256 1a55775f...c98477, 384-d), `intfloat/multilingual-e5-base` (revision d128750597153bb5987e10b1c3493a34e5a4502a, sha256 a18a44fa...de09a7, 768-d), `Snowflake/snowflake-arctic-embed-m-v2.0` (revision 95c2741480856aa9666782eb4afe11959938017f, sha256 3d80d472...c26604, 768-d, CLS pooling, `trust_remote_code` with the local code audited: no urllib/requests/socket/subprocess/exec/eval/pickle). Python 3.12.10, torch 2.14.1+cu126, transformers 4.57.6, sentence-transformers 5.7.0, numpy 2.5.4, scipy 1.18.1, pandas 3.1.0, RapidFuzz 3.14.6, CUDA 12.6, RTX 4070 Ti (`emb_corpus/results/ENV.json`, copy in `benchmarks/emb_corpus/ENV.json`; note transformers 4.57.6 here vs 5.16.1 in the NER environment). e5: `query: ` / `passage: ` prefixes, mean pooling; arctic: `query: ` for queries only, CLS pooling; max_seq 512 (arctic also at 1024 for the whole-page baseline); fp16 on GPU, no NaNs; fp32 check on `s1200_o12`: top-1 agreement 98.99 % (e5-base), answer R@5 73.46 vs 73.35, mean |top-1 cosine difference| 0.00025, max 0.00110.

**Protocol.** All 600 documents (1,045 pages, 13 of them empty after OCR, 1,032 non-empty) in one filter-free collection; cosine on L2-normalised vectors; top-10 chunks per query. Queries: 2,871 same-language answerable questions, 600 cross-lingual paraphrases (question language differs from document language), 300 unanswerable. Metrics: doc hit@k (a top-k chunk is from the gold document), page hit@k, **answer hit@k** (a top-k chunk of the gold document contains the answer string; matching is normalised substring, or space-less substring of at least 5 chars, or `partial_ratio_alignment >= 85` whose window has exactly the same digits, so letter-level OCR errors are tolerated but a wrong digit never is; this definition is the benchmark's own). `ceiling` = share of questions whose answer string exists in some chunk of the gold document after OCR: 95.8 % overall, clean 98.2, medium 98.0, hard 86.5 (86.0 in the chunk-sweep table). Source: `emb_corpus/results/REPORT_tables.md` (generated), `README.md`.

**Headline: `s800_o12` (800 characters, 12 % overlap), same-language questions, answer R@5 and doc R@1 by language and noise level**

| group | n | e5-small ans R@5 | e5-base ans R@5 | arctic ans R@5 | e5-small doc R@1 | e5-base doc R@1 | arctic doc R@1 | ceiling |
|---|---|---|---|---|---|---|---|---|
| ALL | 2871 | 59.4 (57.7-61.1) | **64.4 (62.7-66.2)** | 60.5 (58.7-62.4) | 65.7 (64.0-67.6) | **68.4 (66.6-70.2)** | 65.0 (63.3-66.8) | 95.8 |
| en | 578 | 65.4 (61.8-69.0) | 68.3 (64.5-72.0) | 67.1 (63.5-70.8) | 73.0 (69.2-76.6) | 72.3 (68.9-76.1) | 68.9 (64.9-73.0) | 97.4 |
| es | 570 | 57.0 (53.0-61.2) | 62.3 (58.2-66.3) | 56.5 (52.5-60.9) | 65.8 (61.8-69.3) | 68.8 (65.1-72.5) | 62.8 (58.9-66.7) | 95.8 |
| pt | 583 | 59.5 (55.4-63.5) | 65.2 (61.2-69.1) | 62.1 (58.0-65.9) | 62.1 (58.3-65.7) | 68.3 (64.2-71.9) | 65.4 (61.2-69.1) | 95.7 |
| fr | 579 | 56.1 (52.0-60.1) | 62.0 (57.9-65.8) | 54.9 (50.8-58.9) | 62.9 (58.9-66.8) | 65.8 (62.2-69.8) | 63.0 (59.1-66.8) | 94.3 |
| de | 561 | 58.8 (54.5-62.6) | 64.3 (60.2-68.3) | 61.9 (57.8-66.0) | 64.7 (60.6-68.8) | 66.8 (62.9-70.9) | 64.9 (61.0-68.8) | 95.5 |
| clean | 1140 | 62.5 (59.6-65.1) | 67.5 (64.9-70.1) | 64.3 (61.5-66.9) | 64.3 (61.3-66.9) | 65.5 (63.0-68.2) | 61.7 (58.9-64.4) | 98.2 |
| medium | 1146 | 62.2 (59.4-65.1) | 67.0 (64.4-69.8) | 62.3 (59.2-65.2) | 66.4 (63.6-69.1) | 70.1 (67.3-72.7) | 66.8 (64.0-69.4) | 98.0 |
| hard | 585 | 47.9 (43.6-52.0) | 53.3 (49.4-57.4) | 49.6 (45.5-53.7) | 67.0 (63.1-70.9) | 70.8 (67.0-74.5) | 68.0 (64.1-71.8) | 86.5 |

More metrics for e5-base in the same file (ALL): ans R@1 38.4 (36.6-40.2), R@3 58.2 (56.4-60.0), R@10 70.7 (69.0-72.3), MRR@10 49.5 (47.9-51.1); doc R@5 92.0 (91.0-93.1), R@10 96.4 (95.7-97.1); page R@1 60.1 (58.3-62.0), R@5 89.0 (87.8-90.2). Answer R@1 is low compared with R@10; the source attributes part of the retrieval error to corpus ambiguity (template documents share project names and topics, `bm25_diag.py` docstring). Paired differences, same-language questions, `s800_o12`: e5-base minus e5-small ans R@5 +5.1 [+3.9, +6.2], ans R@1 +8.2 [+6.7, +9.7] (all five languages significant on R@5: en +2.9, es +5.3, pt +5.7, fr +5.9, de +5.5); arctic minus e5-base ans R@5 -3.9 [-5.1, -2.8], ans R@1 -1.1 [-2.5, +0.3] (not significant), doc R@1 -3.4 [-5.0, -1.8].

**Cross-lingual questions (600; `s800_o12`)**

| model | ans R@5 | ans R@1 | doc R@1 | doc R@5 |
|---|---|---|---|---|
| e5-small | 39.2 (35.5-43.3) | 19.3 (16.2-22.3) | 40.0 (36.3-43.7) | 60.3 (56.3-64.2) |
| e5-base | 50.0 (45.8-54.0) | 29.0 (25.5-32.8) | 50.2 (46.2-54.0) | 71.5 (67.5-74.8) |
| **arctic-m-v2** | **54.7 (50.7-58.5)** | **34.3 (30.7-38.0)** | **62.0 (58.0-65.8)** | **81.8 (78.7-84.8)** |

On the same 600 questions the same-language minus cross-lingual gap in ans R@5 is +18.8 pp (e5-small), +12.8 [+10.2, +15.8] (e5-base) and +3.8 [+1.8, +5.8] (arctic-m-v2); for English documents the e5-base gap is +22.5 [+14.2, +30.0]. Break-even mix: arctic is better than e5-base on ans R@1 when more than 17 % of queries are cross-lingual and on ans R@5 when more than 46 % are. One global index versus one index per language with **oracle routing** (the query is sent to the index of the gold document's language, which a real query does not reveal): same-language ans R@5 arctic 60.5 -> 64.6, +4.1 [+3.4, +4.8]; e5-base 64.4 -> 64.6, +0.2 [+0.0, +0.3]; cross-lingual ans R@5 e5-base 50.0 -> 60.0, +10.0 [+7.8, +12.3], arctic 54.7 -> 62.2, +7.5 [+5.5, +9.7]. The global index is therefore close to optimal for e5-base on same-language questions, and per-language gains require knowing the document language.

**Effect of OCR on retrieval** (same chunker on ground-truth text vs OCR text; ans R@5): `s800_o12` e5-base truth 66.1, OCR-flat-structure 63.0, OCR with Tesseract paragraphs 64.4; OCR (one paragraph per page) minus truth -3.1 pp (CI -4.3 to -2.0) overall and -10.3 pp on the hard level (-14.0 to -6.7), -0.3 pp on clean (-1.6 to +1.1). Using Tesseract paragraph structure instead of one paragraph per page is worth +1.4 [+0.4, +2.5] for e5-base.

**Reproduce:** `bash benchmarks/emb_corpus/scripts/run_all.sh` (about 15 minutes on an RTX 4070 Ti; needs the OCR result file and the corpus; the scripts hard-code `C:/eb/...` and `.venv2` paths, see `benchmarks/README.md`). Public data: `emb_public/src/prep_mmarco.py`, `prep_webfaq_miracl.py webfaq|miracl`, `dl_models.py`, `run_all.sh` (+ `run_2.sh`, `run_rest.sh`) for retrieval, `run_cpu.sh` for latency, `analyze.py`, `calib.py` for the tables.

---

## 7. Chunking

**What was measured.** Chunk size (300 / 500 / 800 characters, with 0 or 12 % overlap, plus whole page, 1,200 and 1,600 outside the requested grid) with the three embedders on OCR text; answer-level recall@k as in section 6.2. Chunker (`benchmarks/emb_corpus/scripts/chunker.py`): page -> Tesseract paragraph -> sentence (abbreviation / ordinal / initial guarded) -> word; greedy packing to `size` characters; never splits a word, never crosses a page; overlap = the last whole words of the previous chunk (12 % of size = 36 / 60 / 96 / 144 characters for 300 / 500 / 800 / 1,200); lines of a paragraph joined with a space; de-hyphenation of `letter-` + lowercase line starts (fired 0 times in the corpus, the generator does not hyphenate); `_w800` = pages of at most 800 characters kept whole (no measurable effect). Source: `REPORT_tables.md` section 1, `logs/chunk_ocr.log`.

**Answer-level recall, same-language questions, global index (CI in parentheses; ctx = mean characters passed to the LLM for the e5-base ranking):**

| config | chunks | chunks/page | mean chars | ceiling % | e5-small R@5 | e5-base R@3 | e5-base R@5 | arctic R@5 | ctx k=3 chars | ctx k=5 chars |
|---|---|---|---|---|---|---|---|---|---|---|
| s300_o0 | 7999 | 7.65 | 225 | 95.6 | 40.6 (38.9-42.4) | 35.1 (33.4-36.9) | 41.3 (39.6-43.1) | 37.9 (36.2-39.8) | 725 | 1192 |
| s300_o12 | 8804 | 8.42 | 233 | 95.7 | 41.2 (39.4-43.2) | 36.6 (34.9-38.5) | 42.3 (40.5-44.1) | 40.7 (38.9-42.6) | 741 | 1214 |
| s500_o0 | 4639 | 4.44 | 388 | 95.5 | 47.5 (45.7-49.4) | 43.6 (41.8-45.4) | 49.9 (48.0-51.7) | 46.0 (44.0-47.9) | 1188 | 1977 |
| s500_o12 | 5054 | 4.84 | 402 | 95.6 | 52.1 (50.2-54.0) | 48.3 (46.5-50.2) | 54.2 (52.3-56.1) | 51.1 (49.2-53.0) | 1216 | 2020 |
| s800_o0 | 3005 | 2.88 | 600 | 95.7 | 56.6 (54.8-58.4) | 57.4 (55.5-59.1) | 63.7 (62.0-65.4) | 58.3 (56.6-60.3) | 1833 | 3075 |
| **s800_o12** | 3196 | 3.06 | 626 | 95.8 | 59.4 (57.7-61.1) | 58.2 (56.4-60.0) | 64.4 (62.7-66.2) | 60.5 (58.7-62.4) | 1870 | 3128 |
| s1200_o12 | 2202 | 2.11 | 894 | 95.7 | 67.2 (65.5-68.9) | 66.6 (64.9-68.5) | 73.5 (71.7-75.1) | 70.0 (68.4-71.8) | 2705 | 4533 |
| s1600_o12 | 1735 | 1.66 | 1117 | 95.8 | 68.7 (67.0-70.4) | 68.1 (66.2-69.8) | 76.3 (74.7-77.9) | 74.6 (73.0-76.2) | 3520 | 5906 |
| page | 1032 | 0.99 | 1749 | 95.7 | 73.7 (72.3-75.4) | 72.2 (70.5-73.9) | 79.7 (78.2-81.2) | 78.9 (77.4-80.4) | 5211 | 8747 |

`chunks/page` here is chunks divided by all 1,045 pages (`chunk_ocr.log`); `REPORT_tables.md` divides by the 1,032 non-empty pages and prints 3.10 for `s800_o12`, 0.99 -> 1.00 for the whole page. Mean tokens per chunk: `s800_o12` 158 (arctic-m-v2 on OCR text; e5-base on the closely related `ocrflat` chunking 159, truth text 157), `s1200_o12` 227, `s1600_o12` 283 (e5-base); share over 512 tokens: 0 % up to `s1200`, 0.7 % at `s1600`, 33.4 % for whole pages (so whole-page embeddings are truncated).

Paired differences in answer R@5 (e5-base): overlap 12 % vs 0 % at 800 characters +0.7 [-0.3, +1.8] (not significant; significant for e5-small +2.8 and arctic +2.2), at 500 characters +4.2 [+3.1, +5.6]; 500 vs 300 +11.9 [+10.3, +13.6]; 800 vs 500 +10.2 [+8.6, +11.9]; 1,200 vs 800 +9.0 [+7.5, +10.6]; whole page vs 800 +15.3 [+13.4, +17.1].

**What-if: retrieve small chunks, hand the parent page to the LLM** (hit = the answer string is on one of the distinct pages of the top-k chunks; context = summed OCR length of those distinct pages; `REPORT_tables.md` section 1b, `logs/page_expansion.log`):

| config | k chunks | e5-small | e5-base | arctic-m-v2 | ctx chars (e5-base) |
|---|---|---|---|---|---|
| s300_o12 | 3 / 5 | 75.1 / 82.2 | 76.2 / 83.8 | 67.5 / 74.6 | 4707 / 7452 |
| s500_o12 | 3 / 5 | 77.3 / 83.4 | 79.3 / 85.8 | 73.4 / 79.1 | 4734 / 7530 |
| **s800_o12** | 1 | 52.6 (50.9-54.5) | 59.8 (58.0-61.8) | 55.5 (53.6-57.3) | 1951 |
| **s800_o12** | **3** | 76.0 (74.5-77.6) | **81.4 (79.9-82.9)** | 74.5 (72.9-76.1) | **4798** |
| **s800_o12** | **5** | 82.9 (81.6-84.3) | **87.0 (85.7-88.3)** | 80.6 (79.0-82.2) | **7669** |
| s1200_o12 | 3 / 5 | 72.8 / 79.9 | 77.6 / 84.4 | 76.0 / 81.7 | 4970 / 7955 |
| page as chunk (reference, from the sweep table above) | 3 / 5 | 65.7 / 73.7 | 72.2 / 79.7 | 72.8 / 78.9 | 5211 / 8747 |

At about the same context size (k = 3: 4,798 vs 5,211 characters) small chunks plus page expansion reach 81.4 against 72.2 for whole-page embeddings, because a whole-page vector is diluted (and truncated beyond 512 tokens for a third of the pages). By noise level at s800_o12 + expansion, k = 3, e5-base: clean 81.7 (79.2-83.9), medium 85.2 (83.0-87.2), hard 73.7 (69.9-77.1).

**Lexical diagnostic (BM25 over the same chunks, not an embedder; no intervals; `bm25_diag.py`):** whole page ans R@5 91.4, s800_o12 R@1 / R@3 / R@5 41.5 / 62.6 / 67.6 (e5-base 38.4 / 58.2 / 64.4), s1200_o12 R@5 80.7. BM25 is above all three embedders at every size, which indicates that exact-term matching is easy on this corpus (documents quote exact names, project terms and values); it is a diagnostic, not a candidate (section 12).

**Caveats.** The 800-character result is the best inside the tested grid, tuned on template text with pages of about 1,750 characters; real pages may prefer other sizes. The ranking of chunk sizes by raw chunk recall (larger is better, up to the page) is not the ranking by context budget; the parent-page strategy is what makes 800 the chosen size.

**Reproduce:** `scripts/chunker.py --src ocr|ocrflat|truth`, `embed.py`, `analyze_sweep.py`, `chunk_pairs.py`, `page_expansion.py`, `bm25_diag.py` (all in `run_all.sh`).

---

## 8. Refusal threshold

**What was measured.** The assistant refuses a question when the cosine of the top-1 retrieved chunk is below *t*. 2,871 answerable questions against 300 unanswerable (100 absent-entity, 100 cross-document, 100 off-topic), e5-base, `s800_o12`, OCR text, filter-free index. Scenarios: `global` (all 300), `global_no_xdoc` (without the 100 cross-document items; 79 of them are verbatim questions of another document in the corpus and therefore answerable in a filter-free index, and only 53-62 % retrieve that source document at top-1), `doc_scoped` (search restricted to the gold/paired document; the 50 corpus-level off-topic items have no paired document and are dropped, 250 negatives). Bootstrap B = 500 over questions. Source: `REPORT_tables.md` section 4.

| model | scenario | n pos / neg | AUROC | median top-1 ans / unans | Youden t (CI) | kept / refused at Youden | t at 95 % kept (CI) | kept / refused at that t | t at 90 % refused | kept at 90 % refused |
|---|---|---|---|---|---|---|---|---|---|---|
| **e5-base** | **global** | 2871 / 300 | 0.816 (0.787-0.846) | 0.863 / 0.809 | 0.821 (0.820-0.838) | 87.0 / 63.3 | **0.808 (0.806-0.809)** | **95.0 / 48.3** | 0.875 | 36.5 / 90.3 |
| e5-base | global_no_xdoc | 2871 / 200 | 0.944 (0.931-0.955) | 0.863 / 0.795 | 0.830 (0.821-0.838) | 79.8 / 93.0 | 0.808 (0.806-0.809) | 95.0 / 66.0 | 0.828 | 81.6 / 90.5 |
| e5-base | doc_scoped | 2871 / 250 | 0.976 (0.968-0.982) | 0.857 / 0.770 | 0.816 (0.804-0.818) | 88.0 / 94.4 | 0.804 (0.801-0.806) | 95.0 / 86.0 | 0.812 | 90.4 / 90.4 |
| e5-small | global | 2871 / 300 | 0.821 (0.791-0.851) | 0.877 / 0.824 | 0.838 (0.834-0.851) | 86.6 / 66.3 | 0.825 (0.823-0.826) | 95.0 / 50.3 | 0.886 | 35.1 / 90.3 |
| arctic-m-v2 | global | 2871 / 300 | 0.831 (0.800-0.859) | 0.559 / 0.369 | 0.417 (0.415-0.457) | 88.3 / 67.3 | 0.370 (0.365-0.378) | 95.0 / 50.7 | 0.603 | 37.5 / 90.3 |

**The 0.808 default is the e5-base `global` threshold that keeps 95.0 % of the answerable questions**, and at that value it refuses 48.3 % of all unanswerable questions (66.0 % without the cross-document items). A stricter threshold (Youden 0.821) refuses 63.3 % but keeps only 87.0 %. Further e5-base results at 0.808 (`global_no_xdoc`, per category): off-topic questions are refused 94 %, absent-entity questions only 38 % (at the Youden threshold 100 % and 86 %); an absent-entity question names a person, organisation or project that is verified absent from the paired document, and its median top-1 score is 0.814 against 0.776 for off-topic questions. Answerable questions kept at 0.808 by noise level: clean 96.1 %, medium 95.0 %, hard 93.0 %; cross-lingual answerable questions kept: 84.7 % (at the Youden threshold 75.0 %). Top-1 chunk quality among the kept answerable questions at 0.808: 68.4 % from the gold document, only 39.1 % containing the answer. Per-language Youden thresholds for e5-base (`global_no_xdoc`, only 40 negatives per language, noisy): en 0.817 (0.801-0.817), es 0.828 (0.815-0.828), pt 0.842 (0.834-0.845), fr 0.830 (0.812-0.843), de 0.836 (0.819-0.836) against the pooled 0.830; using the pooled value costs Youden J of 0.119 (en), 0.012, 0.126 (pt), 0.002, 0.014 (so a single threshold is a compromise, worst for English and Portuguese).

**Caveats.** The unanswerable questions are template-generated and easier to separate than real ones; thresholds are valid only for this model, chunk configuration and OCR pipeline (the public-data calibration of the same model gives 0.86, section 6.1); cosine was computed in fp16 on GPU, and fp32 differs by at most 0.0011 per top-1 score (mean 0.00025) on the checked configuration (section 6.2), comparable to the width of the threshold's own interval (0.806-0.809). The cross-document scenario is ambiguous by construction. The threshold should be re-calibrated on a sample of real documents and questions before relying on it.

**Reproduce:** `python -I -X utf8 scripts/refusal.py --src ocr --cfg s800_o12 ...` then `make_report.py s800_o12` (`run_all.sh`).

---

## 9. Capacity estimate

**Purpose.** An order-of-magnitude estimate of CPU cost per document and documents per hour per core, derived from measured quantities only, for the two heavy endpoints. It is recomputed by `python docs/capacity_estimate.py` (stdlib only). **It MUST be re-measured on the target hardware**: the inputs below come from one desktop machine (AMD Ryzen 7 7700X, 8 cores / 16 logical processors, RTX 4070 Ti not used for CPU timings; the CPU model is not recorded in the result files, see the preamble), a Linux container for OCR and native Windows for NER and embeddings, synthetic documents, and no concurrent load except where stated. Real documents and pages differ (page count, text density, scan quality, language mix).

**Formula** (per page *p*; *P* = pages per document):

```
core_s_per_page_extract = t_render_prep + t_ocr + 4 * t_ner(page)                         # /api/extract = OCR + NER
core_s_per_page_ingest  = t_render_prep + t_ocr + 4 * c * t_embed(chunk)                   # /api/ingest  = OCR + chunk + embed
core_s_per_doc          = P * core_s_per_page
documents_per_hour_per_core = 3600 / core_s_per_doc
cores_for_1000_docs_per_hour = 1000 * core_s_per_doc / 3600
```

with P = 1,045 / 600 = 1.7417 pages per document (`data/README.md`); `c` = chunks per page; the factor 4 converts a wall-clock time measured with 4 threads into core-seconds (an upper bound: it assumes the four threads were fully busy, and 1-thread time was not measured); OCR is single-threaded per process (`OMP_THREAD_LIMIT=1`).

**Measured inputs**

| quantity | value | source and conditions |
|---|---|---|
| render + preprocess per page | 0.15 s (mean) | `timing_summary.md`: 200 DPI, flat; 30 pages, sequential, idle machine |
| Tesseract per page, language known | 0.87 s mean (median 0.86, p95 1.08) | same row, psm 3, oem 1, `OMP_THREAD_LIMIT=1`, tesseract 5.5.0 `tessdata_fast`, `single` |
| Tesseract per page, five languages | 1.65 s mean (median 1.59, p95 2.25) | same, `all` |
| 16 workers on 16 logical cores | 54 pages in 7.9 s wall (render + OCR) | `thr_16x1.jsonl`; 16 x 7.9 / 54 = 2.34 core-s per page |
| NER per page (XLM-R-base-hrl, whole page) | **bucket 1500-2200 chars: mean 416.7 ms, median 373.8 ms** at 4 threads | `e6_cpu_speed_xlmr.json`; the bucket (8 pages, mean 1,881 chars) contains the median OCR page length 1,765 chars; fp32 CPU, 4 threads, after warm-up |
| embedding per passage (e5-base) | 106.2 ms at 122 tokens, 356.2 ms at 402 tokens | `cpu_table.md` / `cpu_bench.jsonl`; fp32 CPU, 4 threads, batch 16, median of 5 runs |
| tokens per 800-char chunk | 159 (e5-base, `ocrflat` chunks; arctic on OCR chunks 158) | `emb_corpus/logs/embed_e5-base_ocrflat.log`, `embed_arctic-m-v2_ocr.log` |
| chunks per page, `s800_o12` | 3,196 / 1,045 = 3.06 (3.10 if divided by the 1,032 non-empty pages) | `logs/chunk_ocr.log`; `REPORT_tables.md` |

Derived (marked in the script): embedding time at 159 tokens by linear interpolation between the two measured points = 139.2 ms per chunk (proportional scaling of the 122-token figure gives 140.7 ms, the report's own estimate 142.7 ms); core-seconds = 4 x wall for NER and embedding. The tokens in the CPU benchmark exclude the `passage: ` prefix (about 2 tokens).

**Result** (`python docs/capacity_estimate.py`, pasted verbatim):

```
Capacity estimate (core-seconds; 100 % core utilisation, no headroom)
pages per document = 1045 / 600 = 1.7417
embedding ms/chunk at 159 tokens (interpolated between 122 and 402 tokens) = 139.2
NER ms/page, bucket 1500-2200 chars: mean 416.7, median 373.8 (primary = mean)

Scenarios
  A  language known, idle machine (timing_grid_flat, single)
    per page : render+prep+OCR 1.020 s | NER 1.667 core-s | embedding 1.703 core-s (3.058 chunks x 139.2 ms x 4)
    /api/extract (OCR+NER)          2.687 core-s/page   4.680 core-s/doc    769.3 docs/h/core   1.30 cores for 1000 docs/h
    /api/ingest (OCR+chunk+embed)   2.723 core-s/page   4.743 core-s/doc    759.0 docs/h/core   1.32 cores for 1000 docs/h
  B  all five languages in one pass (timing_grid_flat, all)
    per page : render+prep+OCR 1.800 s | NER 1.667 core-s | embedding 1.703 core-s (3.058 chunks x 139.2 ms x 4)
    /api/extract (OCR+NER)          3.467 core-s/page   6.038 core-s/doc    596.2 docs/h/core   1.68 cores for 1000 docs/h
    /api/ingest (OCR+chunk+embed)   3.503 core-s/page   6.102 core-s/doc    590.0 docs/h/core   1.69 cores for 1000 docs/h
  C  as A, but render+prep+OCR taken from the 16-worker run (16 x 7.9 s / 54 pages)
    per page : render+prep+OCR 2.341 s | NER 1.667 core-s | embedding 1.703 core-s (3.058 chunks x 139.2 ms x 4)
    /api/extract (OCR+NER)          4.008 core-s/page   6.980 core-s/doc    515.8 docs/h/core   1.94 cores for 1000 docs/h
    /api/ingest (OCR+chunk+embed)   4.044 core-s/page   7.043 core-s/doc    511.1 docs/h/core   1.96 cores for 1000 docs/h
  D  two-pass language handling, derived: render once + OCR(all languages) + OCR(single), every page
    per page : render+prep+OCR 2.670 s | NER 1.667 core-s | embedding 1.703 core-s (3.058 chunks x 139.2 ms x 4)
    /api/extract (OCR+NER)          4.337 core-s/page   7.553 core-s/doc    476.6 docs/h/core   2.10 cores for 1000 docs/h
    /api/ingest (OCR+chunk+embed)   4.373 core-s/page   7.617 core-s/doc    472.6 docs/h/core   2.12 cores for 1000 docs/h

Sensitivity (scenario A)
  A with NER median instead of mean
    per page : render+prep+OCR 1.020 s | NER 1.495 core-s | embedding 1.703 core-s (3.058 chunks x 139.2 ms x 4)
    /api/extract (OCR+NER)          2.515 core-s/page   4.381 core-s/doc    821.8 docs/h/core   1.22 cores for 1000 docs/h
    /api/ingest (OCR+chunk+embed)   2.723 core-s/page   4.743 core-s/doc    759.0 docs/h/core   1.32 cores for 1000 docs/h
  A with 3.10 chunks/page (3196 / 1032 non-empty pages)
    per page : render+prep+OCR 1.020 s | NER 1.667 core-s | embedding 1.725 core-s (3.097 chunks x 139.2 ms x 4)
    /api/extract (OCR+NER)          2.687 core-s/page   4.680 core-s/doc    769.3 docs/h/core   1.30 cores for 1000 docs/h
    /api/ingest (OCR+chunk+embed)   2.745 core-s/page   4.781 core-s/doc    753.1 docs/h/core   1.33 cores for 1000 docs/h
```

**Reading.** On the measured hardware and corpus, with the language known and an idle machine (scenario A), `/api/extract` costs 4.68 core-seconds per document (about 770 documents per hour per core) and `/api/ingest` 4.74 core-seconds (about 760 per hour per core); 1,000 documents per hour needs about 1.3 cores at 100 % utilisation. Under a saturated machine, with five-language OCR, or with a two-pass language strategy the figure falls to about 480-600 documents per hour per core (about 1.7 to 2.1 cores). A production sizing needs headroom on top (utilisation well below 100 %), model loading and memory (NER process peak 1.65 GB; embedding peak about 2.6 GB including the torch baseline; Tesseract about 50 MB per process), and the measurements below.

**What is not in the estimate.** The LLM call of `/api/ask` (latency and throughput depend on the provider and model, not measured here); PDF parsing overhead, chunking (pure Python), fuzzy matching, geometry, vector-store upsert/search, HTTP and JSON serialisation, queueing, cold start; NER and embedding run with 4 threads, so the 1-thread cost and the parallel efficiency are unknown (the x4 conversion is an upper bound for 4-core use). Not measured: GPU inference cost for NER/embedding on the target hardware (GPU was used only for accuracy runs). The NER cost uses the median-length page, whereas cost grows faster than linearly with page length (section 4.4), so a skewed page-length distribution costs more than this figure suggests. Synthetic pages are single-column, single-font, 1,763 characters on average; dense or multi-column real pages may be slower for OCR.

---

## 10. Data-leakage and provenance caveats

- **Public NER sets.** The Davlan `*-ner-hrl` models and `xlm-roberta-large-finetuned-conll03-english` were trained on CoNLL (CoNLL-2003 / 2002 columns marked "L" in the source); their news-set scores are inflated. `wikineural_mbert` is trained on WikiNEuRal and its sister MultiNERD sets, and spaCy `xx_ent_wiki_sm` and the router models on WikiNER silver data ("W"); their silver-set scores are inflated. The leak flags are the benchmark's own (`leak()` in `ner_public/scripts/models.py`); the "clean" pool (WikiANN, Few-NERD, GermEval-2014) is clean in the sense that no overlap was known, not verified against the model cards.
- **Public retrieval sets.** The statements made during the benchmark about the training data of `multilingual-e5` and `bge-m3` (relevant to possible overlap between their training data and the public evaluation sets) were written from memory and **not verified** against the model cards or papers. Treat the public rankings of those two models with that uncertainty; the corpus benchmark (section 6.2) is not affected, because the documents are newly generated.
- **Tuned on the reporting data.** The post-processing rule set (REC), the fuzzy threshold, the chunk size and the refusal threshold were chosen on the same 600-document corpus on which they are reported; the sources describe no separate held-out split. Intervals reflect sampling over documents/questions, not selection effects.
- **Synthetic data.** All limitations of section 1 apply; in particular template text makes retrieval easier, near-miss decoys are easier than real look-alikes, and the unanswerable questions are easier to separate than real ones.
- **Environment mismatch.** OCR ran in a Linux container, NER and embeddings on native Windows; NER used transformers 5.16.1 and the embedding runs 4.57.6; accuracy runs on GPU (fp16 for embeddings, fp32 for NER) while the service targets CPU (fp32): the fp32 check (section 6.2) found top-1 chunk agreement of 98.5-99.9 % between fp16 and fp32, for the three embedders only (not checked for NER or for the CPU run itself).
- **Time and machine.** Stage A/B OCR times are contended times with 14 workers; sequential times (section 2.6) are the ones used for capacity. The CPU model of the benchmark machine is not stored in any result file.

---

## 11. Defaults chosen

| default | value | evidence (section, table) | headline figure |
|---|---|---|---|
| OCR rendering | 200 DPI, grey, illumination flattening ("flat") | 2.1 and 2.2 (headline, by noise), 2.4 (preprocessing), 2.5 (DPI grid) | CER 5.4 [4.8-6.1], NameFuzzy 97.6 [96.8-98.3], IoU>=.5 96.2 [95.1-97.0]; hard level NameFuzzy 90.3 vs 42.7 without flattening |
| Tesseract options | `--oem 1 --psm 3 --dpi <render dpi>`, `OMP_THREAD_LIMIT=1`, parallelism across pages | 2.5 (psm), 2.6 (timing, threads) | psm 3 best CER at every DPI; 16 x OMP 1: 7.9 s vs 756.7 s for unset variable on 54 pages |
| Pixel to point conversion | real raster size over `page.rect`, never an assumed DPI | 1 (boxes in points), 4 (box construction) | box coordinates in PDF points, IoU criterion >= 0.5 |
| NER model and input | `Davlan/xlm-roberta-base-ner-hrl`, whole page with line breaks as spaces, ALL-CAPS words title-cased, REC span rules | 3 (public ranking), 4.1 (models x inputs), 4.2 (by language and noise), 4.3 (rules) | P 94.2 [93.3-95.0], R 89.9 [88.7-90.9], F1 92.0 [91.2-92.7]; beats mBERT by +2.5 [+1.9, +3.1] |
| Span post-processing | REC = title strip, hyphen/apostrophe merge, "Last, First" merge (up to 4-token surname), initial extension, drop single-token spans unless titled, drop lowercase / digit / <3-letter spans, split at line breaks; no particle bridge, no box trim | 4.3 (add-one-in, leave-one-out, FP anatomy) | BASE F1 75.4 -> REC 92.0 |
| Fuzzy matching | `rapidfuzz.fuzz.token_sort_ratio >= 0.90` on NFKD accent-stripped, casefolded `"first last"` | 5.1 (sweep), 5.2 (scorers), 5.3 | precision 100.0, recall 87.5 [84.8-90.2] end to end (94.0 oracle), 0 / 893 false positives; WRatio / token_set / partial_ratio give 66-115 false positives |
| Embedder | `intfloat/multilingual-e5-base` (`query: ` / `passage: ` prefixes, mean pooling, fp32 on CPU) | 6.1 (public nDCG@10 mean 79.3, CPU 106.2 / 356.2 ms), 6.2 (corpus) | ans R@5 64.4 (62.7-66.2) at `s800_o12`; +5.1 [+3.9, +6.2] over e5-small; arctic-m-v2 is -3.9 [-5.1, -2.8] below it on same-language questions and +4.7 [+1.8, +7.5] above it on cross-lingual ones; e5-base needs no remote code |
| Chunking | 800 characters, 12 % overlap (96 characters), word-safe, never crossing a page; retrieve chunks, pass the parent page to the LLM | 7 (sweep, paired, parent-page table) | k = 3 chunks + parent pages: 81.4 (79.9-82.9) at 4,798 characters vs 72.2 for whole-page chunks at 5,211 |
| Refusal threshold | top-1 cosine < 0.808 -> refuse | 8 (e5-base `global`, `s800_o12`) | keeps 95.0 % of answerable, refuses 48.3 % of unanswerable (66.0 % without cross-document items) |

---

## 12. Not adopted / future work

- **Hybrid BM25 + dense retrieval.** In the diagnostic BM25 over the same chunks beat all three embedders (whole page ans R@5 91.4 vs 79.7 for e5-base; `s800_o12` 67.6 vs 64.4), but the corpus is template-generated, near-duplicate, and its questions quote exact names and project terms, which favours lexical matching; the result is not evidence for real documents. Not adopted; worth re-testing on real questions (for example with rank fusion of BM25 and e5; a suggestion, not tested).
- **`Snowflake/snowflake-arctic-embed-m-v2.0` for cross-lingual traffic.** It is far better when questions are in a different language from the document (ans R@5 54.7 vs 50.0, doc R@1 62.0 vs 50.2, gap to same-language only +3.8 pp) and wins overall on public data (mean nDCG@10 82.2 vs 79.3); it loses 3.9 pp ans R@5 on same-language questions, needs `trust_remote_code`, and its same-language result improves by +4.1 pp ans R@5 only with a per-language index (oracle routing, section 6.2). Candidate if cross-lingual questions exceed about 17-46 % of traffic (break-even on ans R@1 / R@5, section 6.2) or if a per-language index is acceptable.
- **Model-complexity router** (a cheap model for easy pages, a larger one for hard pages). Not designed or measured; only the per-model latencies of section 4.4 and the public speed-quality table (section 3) exist.
- Other items measured and rejected: whole-page or 1,200/1,600-character embedding (higher raw recall but worse recall per context character once the parent page is passed, section 7); `wikineural_mbert` (best public NER score but CC-BY-NC-SA-4.0 licence and not evaluated on OCR text); spaCy, GLiNER and larger transformer NER models (section 3-4); particle-bridge, box-trim and regex-rescue span rules (section 4.3); `WRatio`, `token_set_ratio`, `partial_ratio`, plain `ratio` (section 5.2); Otsu / adaptive binarisation, 300-DPI rendering of 150-DPI scans, psm 4 and 6 (sections 2.4, 2.5); per-language refusal thresholds (section 8, noisy); the generic legal-form lexicon rule `Dtrap` (+0.3 F1, section 4.3).
- **Open measurements:** single-thread NER and embedding cost; GPU service cost; end-to-end cost of the two-pass language strategy; behaviour on real scans, handwriting, non-Latin scripts, letters that NFKD does not decompose (o-slash, l-stroke, d-stroke, ae); re-calibration of the refusal and fuzzy thresholds on real documents; the verification of the e5 and bge-m3 training-data statements (section 10).
