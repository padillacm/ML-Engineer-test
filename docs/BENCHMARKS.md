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
13. HTTP edge (`add-http-api`): parity, overhead, concurrency, resources, start-up
14. NER pipeline pool (task 10.1)
15. Concurrency with the NER pool (tasks 10.1 and 10.7): job limit, pool size, 50-page documents, resources

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

### 2.8 Adapter re-measurement (final adapter: pypdfium2 rendering + exact OpenCV-equivalent flattening)

**Sample and method** (task 7.2 of `add-ocr-engine`). `TesseractOcrEngine` with the default `TesseractOcrConfig` (`workers=4`, `detect_max_pages=3`) run inside the `docintel-test` image (Tesseract 5.5.0, pypdfium2 5.14.0 / PDFium 156.0.8076.0, numpy flattening). Two sets: (a) the seed-0 stratified sample of 60 documents, 12 per language (5 clean / 4 medium / 3 hard), 102 pages, 844 mentions, run with the true language as hint and without hint; (b) 50 hard documents, 89 pages, 544 mentions (the 15 hard documents of the sample plus stage A3's 40), with hint. Words were scored with the benchmark's own `evaluate.py` (lines = Tesseract block/par/line keys of the kept words, as in the benchmark) and compared, **paired by document**, with the stage B default rows (200 / flat / 3 / single) for the same documents; 95 % bootstrap CIs over documents (1,000 reps, seed 0), as in `report.py`.

| set | metric | benchmark, 600 docs | benchmark, same docs | adapter (hint) | paired diff, adapter - benchmark |
|---|---|---|---|---|---|
| 60 docs, all | NameExact | 90.5 [89.4-91.7] | 92.3 [89.4-94.9] | 91.8 [88.5-94.5] | -0.5 [-1.6, +0.2] |
| 60 docs, all | NameNorm | 95.7 [94.7-96.5] | 97.3 [95.2-99.0] | 96.7 [94.1-98.6] | -0.6 [-1.8, +0.1] |
| 60 docs, all | NameFuzzy | 97.6 [96.9-98.4] | 98.2 [96.4-99.6] | 98.1 [96.3-99.6] | -0.1 [-1.1, +0.7] |
| 60 docs, all | IoU>=.5 | 96.2 [95.2-97.1] | 97.2 [94.8-98.8] | 96.8 [94.4-98.6] | -0.4 [-1.4, +0.5] |
| 60 docs, all | CER | 5.4 [4.7-5.9] | 5.6 [3.9-7.7] | 5.5 [3.9-7.3] | -0.2 [-0.8, +0.5] |
| clean (25 docs) | all name metrics / IoU>=.5 | 93.3-99.4 | 93.9-100.0 | identical | 0.0 [0.0, 0.0] |
| medium (20 docs) | NameExact / NameFuzzy / IoU>=.5 | 93.0 / 99.0 / 98.4 | 95.3 / 99.7 / 99.7 | 95.3 / 99.7 / 99.1 | 0.0 / 0.0 / -0.6 [-1.7, 0.0] |
| hard (15 docs) | NameExact | 78.5 [74.0-82.7] | 82.6 [73.9-90.9] | 80.1 [71.7-88.4] | -2.5 [-8.2, +1.5] |
| hard (15 docs) | NameFuzzy | 90.3 [86.4-93.7] | 91.3 [85.1-97.4] | 90.7 [85.6-97.0] | -0.6 [-5.7, +4.0] |
| hard (15 docs) | IoU>=.5 | 84.1 [79.9-87.9] | 86.3 [80.7-92.6] | 85.7 [82.1-90.8] | -0.6 [-6.2, +4.0] |
| hard (15 docs) | CER | 14.9 [12.5-17.2] | 15.0 [9.8-19.6] | 13.5 [9.0-17.5] | -1.4 [-3.7, +0.1] |
| **50 hard docs** | NameExact | 78.5 [74.0-82.7] | 75.0 [67.8-81.3] | 73.9 [66.5-80.8] | -1.1 [-2.9, +0.4] |
| **50 hard docs** | NameNorm | 84.3 [80.1-88.2] | 82.5 [75.5-88.4] | 81.1 [73.8-87.1] | -1.5 [-3.3, 0.0] |
| **50 hard docs** | NameFuzzy | 90.3 [86.4-93.7] | 88.6 [81.5-94.0] | 87.7 [80.8-93.3] | -0.9 [-2.7, +0.6] |
| **50 hard docs** | IoU>=.5 | 84.1 [79.9-87.9] | 81.6 [74.6-87.8] | 80.9 [74.2-86.9] | -0.7 [-2.8, +1.1] |
| **50 hard docs** | CER | 14.9 [12.5-17.2] | 16.3 [13.0-20.1] | 15.9 [13.0-19.5] | -0.4 [-1.6, +0.7] |

(For the 50 hard documents the "600 docs" column is the benchmark's hard stratum.) Per language on the 60-document sample every paired NameFuzzy / IoU>=.5 CI includes 0 (en -0.6 / 0.0, es 0.0 / 0.0, pt -1.8 [-6.2, 0.0] / -2.4 [-7.2, 0.0], fr +1.3 / +1.3, de +0.7 / -0.7); on the 50 hard documents likewise (largest: pt NameFuzzy -3.7 [-11.7, 0.0], IoU>=.5 -4.9 [-12.7, +2.4]; fr +2.5 / +3.7).

**Without a hint**, detection chose the true language for 60/60 documents (no fallback to the combined model) and the output equalled the hinted run word for word on 102/102 pages, so the no-hint table is identical to the one above.

**Timing** (same container, 16-thread host, sequential documents, `workers=4`; render + flatten + OCR): with a hint 0.71 s/page (102 pages in 72.7 s; median one-page document 1.01 s; per-document s/page p95 1.24), 0.78 s/page on the 89 hard pages; without a hint 1.64 s/page (median one-page document 2.40 s), because of the extra combined-model pass.

**Renderer differences** (89 hard pages; `render_cmp.py`). Every corpus page is one 150-DPI RGB (ICCBased) image upscaled to 200 DPI, with no text or annotations. PDFium and PyMuPDF produce the same raster size on 89/89 pages; 27.6 % of pixels differ, mean |diff| 0.34 grey levels, 0.42 % by more than 2 and none by more than 16, PDFium slightly darker (mean -0.18): image interpolation and RGB-to-grey conversion, not geometry. Rendering in RGB and converting with Rec.601 weights narrows it a little (23.9 %, mean |diff| 0.29). `no_smoothtext`, `draw_annots=False` and `optimize_mode="print"` change nothing (no text or annotations on these pages); `no_smoothimage` (nearest-neighbour upscaling) is much further away (87.8 % differ, 6 % by more than 16) and was not pursued. The residual paired differences above (point estimates -0.1 to -1.5 points, all CIs covering 0) are what this sub-grey-level interpolation difference leaves; closing it entirely would need MuPDF's image resampler, i.e. PyMuPDF, which the licence rules out.

**Verdict.** Accepted. The final adapter matches the benchmark's default configuration: every paired NameFuzzy and IoU>=.5 CI includes 0, overall, per noise level, per language and on the 50 hard documents; clean pages are identical, and CER is equal or lower. Language detection was correct on 60/60 documents.

**History.** The first adapter (PyMuPDF rendering, Pillow `GaussianBlur` flattening) lost 1-4 points of name recall on hard pages (paired NameFuzzy -4.3 [-8.6, -1.3] on the 15 hard sample documents), entirely because Pillow's blur is a box-blur approximation (13.4 % of flattened pixels differed from OpenCV). `preprocess.py` now reproduces OpenCV 5.0's 8-bit `GaussianBlur` bit-exactly (kernel `round(6 sigma + 1) | 1` with taps quantised to 1/256, integer fixed-point row / column passes with `BORDER_REFLECT_101`, done as exact float64 FFT convolutions), so flattened pixels equal OpenCV's on 89/89 hard pages and 20 synthetic cases (flatten 0.19 s per A4 page at 200 DPI, +141 MB on a 40 M pixel page); the renderer then moved from PyMuPDF to pypdfium2 for licensing, which is what is measured above.

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

### 4.5 Implemented chain re-measurement (task 7.2 of `add-name-extraction`)

**Method.** `ExtractNames` with the real adapters at their default configurations (`TesseractOcrEngine`, `TransformersNameRecognizer` on the baked model, `threads=2`) inside the `docintel-test` image, on the seed-0 sample of section 2.8 (60 documents, 102 pages, 844 mentions), language hint = document language. Every `NameMention` was scored with the benchmark's own `score_doc` (IoU >= 0.5, greedy one-to-one, trap rule) and compared **paired by document** with the default's per-document rows (`rows_default_xlmr_pagesp_tc.jsonl`); bootstrap over documents, 1,000 reps, seed 0. A control fed the benchmark's own OCR words (`stageB_flat200`) to the same use case, which isolates the NER runtime and span rules from OCR.

| run (60 docs) | precision | recall | F1 | trapFP | persons (box) | paired dF1 vs default |
|---|---|---|---|---|---|---|
| benchmark default, same docs | 95.7 | 91.6 | 93.6 [91.9-95.1] | 6.6 | 93.5 | - |
| **final chain, hint** | 95.7 [93.4-97.5] | 91.2 [88.6-93.7] | **93.4 [91.7-95.0]** | 5.8 [2.3-10.3] | 93.5 [90.1-96.6] | **-0.2 [-0.8, +0.3]** |
| final chain, no hint | identical (same mentions, names and boxes on 60/60 documents) | | | | | -0.2 [-0.8, +0.3] |
| control: final rules on benchmark OCR words (first run + "pr") | 95.7 | 91.6 | 93.6 | 6.6 | 93.5 | 0.0 [0.0, 0.0] (808/808 spans identical) |

Paired, final chain (hint and no hint): precision -0.0 [-0.6, +0.5], recall -0.4 [-1.3, +0.5]; trapFP -0.8 [-2.2, 0.0]; distinct persons (box) +0.0 [-1.4, +1.4] (full-name reading 91.4 vs 91.2). dF1 by language: en +0.0 [0.0, 0.0], es +0.2 [-0.8, +1.0], pt -0.3 [-2.5, +1.9], fr +0.0 [-1.2, +1.2], de -1.1 [-3.1, +0.2]; by noise: clean -0.2 [-0.8, +0.3], medium -0.4 [-1.5, +0.4], hard +0.4 [-1.8, +2.9]. Every CI includes 0. The residual -0.2 (805 spans, 770 TP vs 808 / 773) is OCR only: on identical OCR words the rules reproduce the benchmark span for span, and the rest is the renderer difference of section 2.8 (a few re-read words on hard and medium pages, e.g. "-GroB," not joined by M2). The new M2 list guard changed no span on this sample (output equal to the first run with "pr" patched in).

**First run (diagnosis).** As first shipped, dF1 was -0.8 [-1.6, -0.0] (fr -3.3 [-5.5, -1.1]): `_TITLES` lacked the French "Pr", so "Pr Rogner" left a bare, untitled surname that Dsingle dropped (9 TP on 6 French documents; patching the list in the measuring process closed the gap). Without a hint, "M." was read as an initial because the title rule got the caller's `None` instead of the detected language ("M. Guyon", 11 names on 4 French documents). Both are fixed (`_TITLES` is now a superset of the benchmark list; `OcrPage.language` feeds the title rule): no-hint output now reads "Guyon", and the only "M. X" name left is the Portuguese initial "M. Azevedo".

Reported names keep inner commas ("Guyon, Guillaume"; the benchmark joins tokens without them); boxes and scores are unaffected.

**Timing** (same container, 16-thread host, sequential documents, OCR `workers=4`, NER `threads=2`): with hint OCR 0.72 + NER 0.55 = **1.27 s/page** (102 pages, 130 s wall); without hint 1.68 + 0.58 = 2.26 s/page (extra detection pass); model load ~5 s once.

**Verdict.** Accepted. The final chain matches the benchmark default on the same 60 documents: paired dF1 -0.2 [-0.8, +0.3] overall, every language and noise CI includes 0, trapFP and distinct-person recall unchanged; with and without hint the output is identical. Absolute F1 93.4 is above the 600-document interval [91.2-92.7] only because the sample is easier (the default scores 93.6 on it).

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

### 8.1 Implemented chain re-measurement (task 8.1 of `add-rag-pipeline`)

**Method.** `IngestDocument` with the real adapters at their defaults (`TesseractOcrEngine` without a language hint, as the use case calls it; `E5Embedder` e5-base revision d1287505, fp32 CPU; `QdrantVectorStore` on Qdrant v1.20.0, dedicated collection dropped afterwards) inside `docintel-test`. (a) the 60 seed-0 sample documents of section 2.8 (102 pages, 327 chunks) on their own; (b) **all 600 documents in one collection** (1,030 non-empty pages, 3,299 chunks; benchmark 3,196), the comparable run. Each query is `embed_query` + `search(top_k=10)`, the same calls as `AskQuestion`. Scoring uses the benchmark's own `metrics.py` (answer match, doc/page hit, parent-page hit = the answer is on one of the distinct pages of the top-3 chunks), **paired per question** with `e5-base__ocr__s800_o12` (Tesseract-paragraph chunks), with `e5-base__ocrflat__s800_o12` (one paragraph per page) and with the previous run of the chain (before page text kept the paragraphs); bootstrap over questions, 1,000 reps, seed 0. The page text is rebuilt from the OCR words with Tesseract's paragraphs (`PageWord.paragraph`, from `block_num`/`par_num`): a blank line between paragraphs, a line break between lines.

| 600-doc index, same-language (2,871) | chain | benchmark `ocr` | paired vs `ocr` | paired vs `ocrflat` | paired vs previous chain (61.8 R@5) |
|---|---|---|---|---|---|
| answer R@1 | 38.9 (37.2-40.8) | 38.4 (36.5-40.2) | +0.5 [-0.5, +1.6] | +1.7 [+0.3, +3.1] | +4.3 [+2.9, +5.7] |
| answer R@3 | 57.9 (56.0-59.7) | 58.2 (56.4-60.1) | -0.3 [-1.3, +0.5] | | +3.2 [+2.0, +4.4] |
| answer R@5 | **64.3 (62.6-66.0)** | 64.4 (62.9-66.1) | **-0.1 [-1.0, +0.7]** | +1.3 [+0.1, +2.4] | +2.5 [+1.3, +3.6] |
| doc R@1 | 67.9 (66.2-69.4) | 68.4 (66.8-70.0) | -0.6 [-1.4, +0.4] | | +0.3 [-0.9, +1.5] |
| page R@1 | 60.7 (59.0-62.6) | 60.1 (58.3-62.0) | +0.6 [-0.5, +1.8] | | +1.2 [-0.1, +2.5] |
| parent page, k=3 | **81.0 (79.5-82.3)** | 81.4 (80.0-82.8) | **-0.5 [-1.2, +0.3]** | +0.7 [-0.3, +1.8] | +0.4 [-0.6, +1.4] |
| kept at 0.808 | **94.7 (93.9-95.5)** | 94.8 (94.0-95.6) | **-0.1 [-0.6, +0.3]** | | -0.1 [-0.6, +0.4] |

**By language and noise (600-doc index, chain vs `ocr`).** Answer R@5: en 67.3 (-1.0 [-2.6, +0.5]), es 61.9 (-0.4), pt 66.6 (+1.4 [-0.7, +3.8]), fr 62.2 (+0.2), de 63.5 (-0.9 [-2.9, +1.2]); clean 68.6 (+1.1 [-0.1, +2.5]), medium 67.2 (+0.2), **hard 50.3 (-3.1 [-5.8, -0.5])**. Against the previous chain answer R@5 rose in every language (en +2.4, es +0.2, pt +4.5, fr +1.9, de +3.7) and on clean (+3.6) and medium (+3.3), not on hard (-1.0 [-3.6, +1.5]). Page R@1 vs `ocr`: every language and noise CI includes 0 except fr +4.1 [+1.6, +6.7] and medium +1.9 [+0.0, +3.8]. Parent page at k=3 vs `ocr`: every CI includes 0 (largest: hard -2.6 [-5.0, +0.2], de -2.0 [-3.9, +0.2]). **Cross-lingual (600):** answer R@5 49.0 vs 50.0 (-1.0 [-3.0, +1.2]; vs previous chain +3.3 [+0.5, +6.2]), doc R@1 50.0 vs 50.2, page R@1 43.8 vs 43.8, parent page 61.3 vs 60.3 (+1.0 [-0.8, +2.8]), kept 83.8 vs 84.3. **Refusal at 0.808 (300 unanswerable):** refused 48.7 % vs 48.3 (+0.3 [-1.3, +2.0]; previous chain 48.7), without cross-document 66.5 % vs 66.0 (+0.5 [-2.0, +3.0]); absent-entity 39 %, cross-document 13 %, off-topic 94 %; mean |top-1 cosine difference| vs the benchmark 0.0026 (previous chain 0.0046). Top-1 chunk from the same document as the benchmark for 89.2 % of the questions (previous chain 82.2 %).

**Chunker control.** The app chunker (`clean_page_text` + `chunk_page`) + `E5Embedder` on the benchmark's own Tesseract-paragraph page text (`ocr__page`, exact numpy search) against `ocr`: answer R@1 +0.3 [-0.4, +1.0], R@5 +0.3 [-0.3, +1.0], refused +0.3 [-1.3, +2.3]. On identical paragraph text the two chunkers retrieve equally well, although their boundary rules differ (the benchmark packs whole paragraphs and splits only paragraphs longer than the limit, the app ends at the furthest strongest boundary; 62.5 % of pages give identical chunk lists). A variant of the app chunker with the benchmark's rule scored R@5 -0.2 [-0.8, +0.3] against `ocr`, no better, so the chunker was left as specified. The -1.1 attributed to boundary rules in the previous run appeared only on one-paragraph pages.

**60 sample documents.** On the 600-doc index the sample's 290 same-language questions remain a noisy subset (see the previous run: a 60-document check is too small to separate a 3-point effect). In a 60-document collection retrieval is much easier and not comparable with the 600-document benchmark.

**Timing** (same container, 16-thread host; previous run, the re-run shared the host with another job). Sequential, defaults (OCR `workers=4`, embedder `threads=2`): **2.20 s/page** wall (OCR with detection pass 1.63 s/page; embedding 178 ms/chunk, 0.56 s/page); model load 6 s once. Bulk run (2 documents at a time, torch 6 threads): 1.02 s/page throughput, embedding 105 ms/chunk. Query: `embed_query` 44.9 ms (2 threads) / 27.3 ms (6 threads) + Qdrant search 2.3-2.7 ms (about 3,300 points, top 10). Truncation plays no part: chunks are at most 800 characters (about 200 tokens, under 512).

**Verdict.** Accepted. With the recognised paragraphs kept in the page text, every overall metric matches the benchmark (answer R@1/3/5, doc and page R@1, parent page at k=3, kept-answerable 94.7 % and refused-unanswerable 48.7 % at 0.808, paired differences all including 0), and answer R@5 rose 2.5 pp over the previous chain. The one remaining gap is answer R@5 on the hard noise level (-3.1 [-5.8, -0.5], parent page -2.6 [-5.0, +0.2]), where the chain's OCR text differs most from the benchmark's (renderer difference of section 2.8). The 0.808 threshold stays valid. Scripts and per-question outputs: `rag_calib/{run.py, ctrl.py, score.py, diag.py}` and `rag_fix/{run.py, ctrl2.py, vchunk.py, cmp.py, score.py, prev.py}` (scratch).

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
| `MAX_CONCURRENT_JOBS` | **12** (applied), one process, 16 logical CPUs | 15.2 (D7 rule: smallest limit within 5 % of the best throughput); before the NER pool: 8 (13.2) | with the NER pool, 75-request runs (two passes or more each): 87.9 docs/min at 12 / pool 4 and 91.2 at 12 / pool 8, against 78.9 at 8 (-13.5 % of the best, 91.2); 16 and 24 are no faster (88.6-89.3; 87.1 on the short set); host-specific, re-measure on the target hardware |
| `NER_POOL_SIZE` (`TORCH_THREADS` 2) | **8** (applied) | 14 (NER alone: identical spans, +70 MB), 15.2, 15.3 and 15.6 (D7 rule over both workloads) | short documents: pools 4 and 8 within 4 % of each other (87.9 and 91.2 docs/min at 12 jobs, run-to-run spread 1-3 %); 50-page documents at 12 jobs: 24 documents in 342-345 s with 8 against 367-370 s with 4 (pool 4 is 7 % slower, outside the 5 % band); under mixed load short-document p95 15 s against 27-33 s; server peak +40-90 MB |
| `BUSY_WAIT_S` | 10 s (unchanged; re-derived with the pool) | 13.2, 15.2 (derived from measured latency, not an optimum) | before the pool: one loaded job at limit 8 (p50 10.1 s, 13.2); with the pool at the recommended 12 jobs: a job holds its slot 7.4-7.7 s on average and p50 latency at 16 clients is 9.6-10.8 s, so 10 s is still about one loaded job; a 50-page job holds a slot 47-347 s, so a burst of long documents is shed after 10 s (intended; clients retry after `Retry-After`) |
| `TORCH_THREADS` | 2 (one process-wide value for the recogniser and the embedder) | 3 and 6 (CPU latency; both models were benchmarked at 2 threads), 14.2 (NER alone: threads 1 vs 2) | NER alone 65.7 docs/min at 2 threads vs 34.7 at 1 (pool 1); the former `NER_THREADS` / `EMBEDDING_THREADS` were merged because torch threads are process-wide (15.2 note); above 2 not measured end to end |
| `MALLOC_ARENA_MAX` | 2 (Dockerfile) | 13.3, 15.5 (resources, 200 mixed requests per round) | RSS flat after round 1 (2,454-2,462 MB) instead of growing 16-520 MB per round, about 840 MB lower, at the same throughput (91.1-92.1 docs/min) |
| `MAX_CONCURRENT_QUESTIONS` | 16 | not measured (judgement) | a question costs about 20 ms of embedding plus the language model's wait (I/O); 16 bounds threads and model calls while a slow model answers; to be measured with the real model in block 6 |
| `READINESS_TIMEOUT_S` | 2 s (at most 30) | not measured (judgement) | a healthy store answers its ping in milliseconds; 2 s is below the 3 s compose healthcheck timeout; the review measured a hung store's ping cut from 10.0 s to 2.0 s |
| Drain after an early 413 | up to 1 MiB over the route limit, at most 5 s | `add-http-api` 10.4 (`tests/api/test_uvicorn_edge.py`) | 900 KiB over: the client reads the 413 (reset without the drain); 5 s is nginx's `lingering_timeout`; far-over bodies are closed after about 1 MiB |
| Body limits | uploads `MAX_UPLOAD_MB` + 1 MiB + 64 KiB; other routes 64 KiB | `add-http-api` 10.6 and review finding 2 | 1 MiB is the multipart parser's per-field limit (no accepted `names` field is cut); a 2,000-character question fully escaped is about 24 KB |
| `MAX_UPLOAD_MB`, `MAX_PAGES`, `MAX_NAMES`, name length | 20 MB, 50 pages, 1,000 names, 100 characters | product limits agreed in the `add-http-api` proposal, not optimised; 15.3 checks the worst case | 24 concurrent 50-page jobs (11.6-15.7 MB files): all 200, server peak 2.6-2.9 GB, no OCR timeout |

---

## 12. Not adopted / future work

- **Hybrid BM25 + dense retrieval.** In the diagnostic BM25 over the same chunks beat all three embedders (whole page ans R@5 91.4 vs 79.7 for e5-base; `s800_o12` 67.6 vs 64.4), but the corpus is template-generated, near-duplicate, and its questions quote exact names and project terms, which favours lexical matching; the result is not evidence for real documents. Not adopted; worth re-testing on real questions (for example with rank fusion of BM25 and e5; a suggestion, not tested).
- **`Snowflake/snowflake-arctic-embed-m-v2.0` for cross-lingual traffic.** It is far better when questions are in a different language from the document (ans R@5 54.7 vs 50.0, doc R@1 62.0 vs 50.2, gap to same-language only +3.8 pp) and wins overall on public data (mean nDCG@10 82.2 vs 79.3); it loses 3.9 pp ans R@5 on same-language questions, needs `trust_remote_code`, and its same-language result improves by +4.1 pp ans R@5 only with a per-language index (oracle routing, section 6.2). Candidate if cross-lingual questions exceed about 17-46 % of traffic (break-even on ans R@1 / R@5, section 6.2) or if a per-language index is acceptable.
- **Model-complexity router** (a cheap model for easy pages, a larger one for hard pages). Not designed or measured; only the per-model latencies of section 4.4 and the public speed-quality table (section 3) exist.
- Other items measured and rejected: whole-page or 1,200/1,600-character embedding (higher raw recall but worse recall per context character once the parent page is passed, section 7); `wikineural_mbert` (best public NER score but CC-BY-NC-SA-4.0 licence and not evaluated on OCR text); spaCy, GLiNER and larger transformer NER models (section 3-4); particle-bridge, box-trim and regex-rescue span rules (section 4.3); `WRatio`, `token_set_ratio`, `partial_ratio`, plain `ratio` (section 5.2); Otsu / adaptive binarisation, 300-DPI rendering of 150-DPI scans, psm 4 and 6 (sections 2.4, 2.5); per-language refusal thresholds (section 8, noisy); the generic legal-form lexicon rule `Dtrap` (+0.3 F1, section 4.3).
- **Open measurements:** single-thread NER and embedding cost; GPU service cost; end-to-end cost of the two-pass language strategy; behaviour on real scans, handwriting, non-Latin scripts, letters that NFKD does not decompose (o-slash, l-stroke, d-stroke, ae); re-calibration of the refusal and fuzzy thresholds on real documents; the verification of the e5 and bge-m3 training-data statements (section 10).

---

## 13. HTTP edge (`add-http-api`)

Scripts: `benchmarks/http_edge/scripts/` (local, like every benchmark script; driver `run.sh`, see `benchmarks/README.md` step 8).

Tasks 8.1-8.3 of `add-http-api`: the FastAPI edge adds no model behaviour, so acceptance is parity with the use cases and cost. **Environment** (all subsections): image `docintel-evidence` built with `docker build --target test` from the tree of this change (Python 3.12.16, Tesseract 5.5.0, FastAPI 0.162.0, Starlette 1.10.0, Uvicorn 0.54.0, httpx 0.28.1), run through `docker-compose.test.yml` (project `block5ev`, Qdrant v1.20.0, collections with a unique prefix dropped afterwards); Docker Desktop 29.9.2 on the Windows 11 host of the preamble (AMD Ryzen 7 7700X); the container sees **16 CPUs** (`os.cpu_count()`, affinity 16, cgroup `cpu.max` unlimited) and 15.2 GiB of memory; settings at their defaults unless stated (so `OCR_WORKERS` = min(4, 16) = 4, `NER_THREADS` = `EMBEDDING_THREADS` = 2), no language model configured; nothing else ran on the host during the timing runs. Inputs: the seed-0 sample of section 2.8 (60 documents, 102 pages), name lists from `namelists.jsonl` where a document has one (19 documents), no language hint. Scripts and raw outputs are in the scratch folder of the session (`http_evidence/scripts/{p81,p82,p82_supp,p82_long,p82_arena,p83,p83_spool,ev_common}.py`, `run.sh`, outputs in `http_evidence/out/`; local, not committed). Each script runs as `TEST_IMAGE=docintel-evidence docker compose -p block5ev -f docker-compose.test.yml run --rm -v <scripts>:/ev:ro -v <corpus>:/data:ro -v <sample>:/calib:ro -v <out>:/out tests python /ev/<script>.py`, followed by `docker compose -p block5ev -f docker-compose.test.yml down -v` (`run.sh` does both).

### 13.1 Parity and overhead (task 8.1)

**Method.** One process builds the NER model and the e5 embedder once (shared adapter factories, as in `tests/integration/test_http_e2e.py`) and two containers with `build_container(load_settings(...))`, each with its own collection prefix: H behind `TestClient(create_app(settings, H'))`, where H' is H with its `ask` replaced by an `AskQuestion` on H's store and a scripted LLM (`FakeLLMClient`, fixed reply), and D for direct ingestion. (a) For every sample document `H.extract(pdf, names, None)` against `POST /api/extract` with the same PDF and names; names, boxes (exact float equality of the domain values with the JSON values) and matches compared, and the whole JSON body against `to_response(result).model_dump(mode="json")`. (b) `POST /api/ingest` (H's collection) against `D.ingest(pdf)` (D's collection); status, chunk count and document id compared, then both collections scrolled and every stored chunk compared (document id, sequence, page, text, offsets). (c) For the sample's 290 answerable questions, their 60 cross-lingual variants, the 22 unanswerable questions paired with sample documents and the 50 corpus-level off-topic ones: `POST /api/ask` against a direct `AskQuestion` with its own scripted LLM on the same store (decision, sources, answer text, prompt messages sent to the LLM) and against an `AskQuestion` on D's store (checks that the two ingestion paths stored equivalent data). Timing: one warm-up document and three warm-up questions first; each pair alternates which call goes first (even items direct first, odd items HTTP first).

| check | identical / total | detail |
|---|---|---|
| extract: names, boxes, matches | **60 / 60** documents | 805 name occurrences, 86 matches (19 documents with a name list); JSON body equals the response model's own serialisation on 60 / 60 |
| ingest: status, chunk count, document id | **60 / 60** | 327 chunks; every response `indexed` |
| ingest: stored chunks (two collections) | **327 / 327** | identical payloads |
| ask: refusal decision and sources | **422 / 422** | 89 refusals (answerable 18 / 290, cross-lingual 10 / 60, paired unanswerable 11 / 22, off-topic 50 / 50) |
| ask: answer text and prompt sent to the LLM | **422 / 422** | |
| ask: HTTP store against the directly ingested store | **422 / 422** | |

No difference was found, so there is no cause to report.

**Overhead** (wall time per call, 60 documents / 422 questions; paired = HTTP minus direct for the same item):

| endpoint | direct median / p95 | HTTP median / p95 | paired difference median / p95 | relative (paired median) | edge only (stub use case) median / p95 |
|---|---|---|---|---|---|
| `/api/extract` | 3.816 / 5.033 s | 3.785 / 4.985 s | -16.0 / +33.6 ms | -0.5 % | 2.41 / 2.84 ms |
| `/api/ingest` | 3.710 / 5.201 s | 3.750 / 5.241 s | -8.9 / +61.5 ms | -0.3 % | not measured |
| `/api/ask` | 44.2 / 56.3 ms | 47.2 / 58.7 ms | +2.6 / +9.3 ms | +5.7 % | 1.82 / 2.03 ms |

"Edge only" replaces the use case by a stub that returns the result already computed for the same input, so it is the cost of `TestClient` + multipart parsing of the upload (median 490 KB) + form validation + response serialisation alone (180 requests each, after one unmeasured round). For extraction and ingestion the edge costs about 2.4 ms against a 3.8 s OCR + NER call (**0.06 %**), well under the 1 % expected by the design; the paired difference is dominated by run-to-run OCR jitter (it was slightly negative in both orders, -15.0 and -17.8 ms; direct calls ran on the main thread and HTTP calls on an AnyIO worker thread, a cause that was not investigated further). For questions the edge adds about 2 ms to a 44 ms call (embedding the question dominates). `TestClient` calls the application in process, so the socket and Uvicorn cost is not in these figures; it is part of the real-server latencies of 13.2.

### 13.2 Concurrency default (task 8.2, design D7)

**Method.** Real Uvicorn (`uvicorn app.main:app --host 0.0.0.0 --port 8000`, one process) started inside the container for each limit with `MAX_CONCURRENT_JOBS` in {1, 2, 4} and `BUSY_WAIT_S=600` (nothing rejected); the client runs in the same container (threads, one `httpx.Client` each, pulling from a shared queue). Request set: 25 sample documents (45 pages; per language the 1st and 3rd clean, 1st and 3rd medium and 1st hard document of the sample, interleaved by language), `POST /api/extract` with the name list where one exists, no language hint. Two warm-up requests per server, then the set at 1, 2, 4, 8 and 16 concurrent clients. Two passes (limits 1, 2, 4 then 4, 2, 1); cells are the mean of both. docs/min = 25 / wall; latency is the client's (includes waiting for a slot); peak RSS is the server's `VmHWM` (reset per cell through `/proc/<pid>/clear_refs`; it matched the 0.25 s sampled maximum within 10 MB) and the sampled peak of the server plus its Tesseract children; CPU is the busy share of all 16 CPUs (`/proc/stat`). Every request of every cell returned 200.

| limit | clients | docs/min | p50 / p95 latency (s) | server peak RSS (MB) | server + Tesseract peak (MB) | CPU busy |
|---|---|---|---|---|---|---|
| 1 | 1 | 15.5 | 4.0 / 4.9 | 1,704 | 1,798 | 10 % |
| 1 | 2 | 15.6 | 7.5 / 8.9 | 1,766 | 1,871 | 10 % |
| 1 | 4 | 15.6 | 15.4 / 17.2 | 1,886 | 1,972 | 10 % |
| 1 | 8 | 15.5 | 29.8 / 33.0 | 2,068 | 2,162 | 10 % |
| 1 | 16 | 15.6 | 48.5 / 62.4 | 2,233 | 2,305 | 10 % |
| 2 | 1 | 15.5 | 4.0 / 4.9 | 1,699 | 1,770 | 10 % |
| 2 | 2 | 28.5 | 4.3 / 5.3 | 1,765 | 1,945 | 18 % |
| 2 | 4 | 27.8 | 8.3 / 10.2 | 1,850 | 2,026 | 17 % |
| 2 | 8 | 28.9 | 16.1 / 17.8 | 2,019 | 2,215 | 18 % |
| 2 | 16 | 27.8 | 29.2 / 34.2 | 2,338 | 2,530 | 17 % |
| 4 | 1 | 15.5 | 4.0 / 4.9 | 1,711 | 1,798 | 10 % |
| 4 | 2 | 28.1 | 4.4 / 5.3 | 1,771 | 1,982 | 17 % |
| 4 | 4 | 39.3 | 6.2 / 7.4 | 1,897 | 2,270 | 25 % |
| 4 | 8 | 40.4 | 10.6 / 13.3 | 2,058 | 2,432 | 26 % |
| 4 | 16 | 41.6 | 18.7 / 22.8 | 2,326 | 2,628 | 27 % |

Within D7's grid limit 4 is best (41.6 docs/min) and 1 and 2 are far below it, but throughput was still rising at 4 while only a quarter of the CPUs were busy, so the grid was extended (same method, two passes each, `p82_supp.py`) and the decisive pair re-run with three times the request set (75 requests, two passes, `p82_long.py`) to reduce the end-of-run tail:

| limit | clients | requests | docs/min (pass 1, pass 2) | p50 / p95 latency (s) | server peak RSS (MB) | server + Tesseract peak (MB) | CPU busy |
|---|---|---|---|---|---|---|---|
| 8 | 8 | 25 | 44.9, 45.3 | 10.1 / 12.6 | 2,043 | 2,607 | 30 % |
| 8 | 16 | 25 | 45.6, 45.5 | 17.7 / 23.2 | 2,391 | 2,905 | 30 % |
| 16 | 8 | 25 | 45.3, 45.1 | 10.1 / 12.8 | 1,974 | 2,567 | 30 % |
| 16 | 16 | 25 | 44.5, 44.7 | 18.3 / 25.8 | 2,288 | 5,228 | 32 % |
| 4 | 16 | 75 | 42.5, 42.0 | 21.3 / 25.9 | 2,294 | 2,658 | 26 % |
| **8** | 16 | 75 | **47.8, 47.7** | 19.4 / 23.4 | 2,332 | 2,709 | 31 % |

**Choice.** The best throughput is limit 8 (47.7 docs/min on the long run; 16 is not faster, 44.5-45.3 against 44.9-45.6 for 8 on the short set, and once peaked at 5.2 GB with up to 20 processes). Limit 4 reaches 42.3 docs/min, 11 % below it, so it is outside the 5 % band; limit 8 is the smallest limit within 5 % of the best. **Recommended `MAX_CONCURRENT_JOBS` = 8** for one process on this 16-CPU host (the initial value 2 gives 61 % of that throughput). At 8 jobs, a single request's latency is unchanged (4.0 s at one client), the server peak stays at 2.0-2.4 GB and the process tree at 2.6-2.9 GB.

**Why throughput stops at about 45-48 docs/min with 70 % of the CPUs idle.** The NER model runs under one lock per process (`TransformersNameRecognizer`, two torch threads): from the use case's own log, NER costs 0.58 s per page unloaded, so the 135 pages of the 75-request run need 78 s of NER, 83 % of the 94 s wall time at limit 8 (73 % at limit 4); its logged `ner_ms` (which includes waiting for the lock) rises from 0.58 s per page at limit 1 to 3.5 s at limit 8 and 5.1 s at limit 16, while OCR per page stays at 1.6-1.8 s (2.2 s at 16). More jobs per process therefore only queue behind the NER lock; more throughput on the same host needs more processes or replicas, each with its own models (~2 GB), as `DESIGN.md` plans (and `KNOWN_LIMITATIONS.md` P4). `NER_THREADS` above 2 was not measured. **Update:** the NER lock has since been replaced by a pool of pipelines (section 14); with it one process reaches 88-91 docs/min on the same request set, the job limit worth having rises to 12 and the bottleneck moves to the CPU, shared by Tesseract and NER (section 15, which also covers 50-page documents and shows that the occasional 5 GB process-tree peaks were most likely a sampling artefact). The documents are short (1-3 pages), so per request OCR uses at most 2-3 of its 4 workers: at 8 jobs at most 14 processes were seen (server + Tesseract); a 50-page document could have 4 Tesseract processes and 8 rendered pages in flight per job, i.e. up to 32 processes at limit 8 on 16 CPUs, which the measurement did not cover.

**Back-pressure check.** With limit 8 and a short `BUSY_WAIT_S=1`, 24 simultaneous `/api/extract` requests (released by a barrier, `p83.py`) gave **8 x 200 and 16 x 503 `service_busy`**, the 503s after 1.03-1.05 s with `Retry-After: 1` and the error body with its `request_id`; no other status. With `BUSY_WAIT_S=10` the same burst gave 10 x 200 (3.7-14.7 s) and 14 x 503 at 10.03-10.04 s with `Retry-After: 10`, nothing else.

**`BUSY_WAIT_S` (recommendation, derived rather than optimised): 10 s.** The unloaded median `/api/extract` latency of these 1-3-page documents is 4.0 s (p95 4.9 s) and a job at full load (8 jobs) takes about 10 s (p50 10.1 s at 8 clients). A 10 s wait (about 2.5 x the unloaded median, about one loaded job) lets a request wait for roughly one job's duration: under steady load, with jobs finishing at staggered times, a slot frees about every 10 s / 8 = 1.3 s, so about 8 queued requests (one more generation) are admitted before the service sheds load; it also bounds what a client waits before it hears 503 to 10 s, and a served request completes within about 25 s (p95 23.4 s at 16 clients on the long run), under the 30 s timeout of common proxies. A synchronised burst is shed harder, because all jobs start together: in the burst above the first slot freed at 3.7 s and the second at 7.4 s, so 5 s would have served 9 of the 24 requests and 10 s served 10. Longer documents (OCR plus NER about 2.3 s per page unloaded without a hint) hold a slot longer, so bursts of them are shed sooner; that is the intended behaviour, and clients retry after `Retry-After`. The current 5 s is also defensible; the measurement bounds the choice (a few seconds to about one loaded job) but does not single out one value.

### 13.3 Resources and start-up (task 8.3)

**Method** (`p83.py`). Uvicorn with `MAX_CONCURRENT_JOBS=8`, `BUSY_WAIT_S=600`, `MAX_UPLOAD_MB=1` and no language model. Warm-up (one valid request of each endpoint and two invalid ones), then two rounds of the same 200 requests from 4 client threads, deterministic shuffle (seed 0): 100 invalid (PNG and text renamed `.pdf`, empty file, four kinds of malformed `names`, uploads of 1 MiB + 1 byte and 2 MiB, a PDF truncated to a third, unsupported language, missing file field, blank, 2,001-character and non-JSON questions; 6-7 of each) and 100 valid (40 extractions, 30 ingestions of sample documents under 1 MiB, the repeats `already_indexed`; 30 questions, refused or 503 `llm_unavailable`). Snapshots of the server process: open descriptors (`/proc/<pid>/fd` and their targets), threads and RSS (`/proc/<pid>/status`), the `/tmp` listing (`TMPDIR` is unset, so Starlette's spooled uploads go to `/tmp`), sampled every second during the rounds. Because no upload under `MAX_UPLOAD_MB=1` exceeds Starlette's 1 MB in-memory spool, a complement (`p83_spool.py`, default `MAX_UPLOAD_MB=20`, two rounds of 40 requests: 20 valid 300-DPI PDFs of 1.5-2.9 MB, 10 of them truncated to 60 %, 10 bodies of 21 MiB) exercised the disk spool, sampled every 0.2 s.

| point | open fds | threads | RSS (MB) | `/tmp` entries | Tesseract processes |
|---|---|---|---|---|---|
| after start-up | 14 | 22 | 1,092 | 2 (pre-existing: `torchinductor_app`, a uv lock) | 0 |
| after warm-up (baseline) | 14 | 23 | 1,859 | 2 | 0 |
| round 1, maximum during | 30 | 34 | 2,375 | 2 (no descriptor on `/tmp`) | 5 |
| round 1, right after / 10 s later | 14 / 14 | 29 / 29 | 2,375 / 2,375 | 2 / 2 | 0 |
| round 2, maximum during | 30 | 32 | 2,399 | 2 | 5 |
| round 2, right after / 10 s later | 14 / 14 | 29 / 29 | 2,386 / 2,386 | 2 / 2 | 0 |
| spool complement, after each round | 14 | 29 | 1,810, 1,844 | 2 (at most one descriptor on an unlinked `/tmp` file seen during) | 0 |

Every invalid request got its documented status and code (400 `empty_file`, 413 `too_large`, 415 `not_a_pdf`, 422 `invalid_names` / `unsupported_language` / `pdf_unreadable` / `invalid_question` / `invalid_request`) and every valid one 200, except questions that would need the model (503 `llm_unavailable`, 6 in round 1 and 10 in round 2 as the index grew); no other 5xx in either run. The spool complement returned 20 x 200, 10 x 422 `pdf_unreadable` and 10 x 413 per round.

**Reading.** Descriptors return to 14 and the `/tmp` listing never changes (spooled uploads are unnamed temporary files, closed at the end of the request); no Tesseract process outlives its request. The thread count settles at 29 after the first round and stays there (idle AnyIO worker threads are kept for reuse; the pool is capped at 40). RSS rises once and then stays flat: +767 MB on the first requests (model working memory), +516 MB during the first loaded round, **+11 MB (0.5 %) during the second** identical round. The growth with concurrency is mostly allocator retention: glibc keeps per-thread malloc arenas and does not return their pages to the system. With `MALLOC_ARENA_MAX=2` (`p82_arena.py`, an environment variable of the C library, no code change) the server peak at limit 1 and 16 clients was 1,813 MB instead of 2,229 MB, and at limit 8 and 16 clients 1,956 MB instead of 2,391 MB, at unchanged throughput (15.5 and 45.4 docs/min); worth setting in the runtime image (block 6), measured on these runs only.

**Start-up** (17 server starts across 13.2 and 13.3): process start to the first `/health/ready` 200 **10.4 s** (median 10.41, range 10.30-10.45). `/health` answers at the same moment, since Uvicorn accepts connections only after the lifespan has built the container. Split: importing `app.main` (FastAPI, torch, transformers, the adapters) 3.47 s (three fresh interpreters, 3.45-3.49); building the container in the lifespan 6.54 s (median of the `service started` log line's `duration_ms`, 6.46-6.58), of which the NER model 5.0 s and e5 1.55 s (timed separately in the 8.1 process; building a second container with the same models took 13 ms); the remaining ~0.4 s is interpreter and Uvicorn start-up, the Qdrant readiness probe (~28 ms) and the 50 ms polling interval. All starts were warm (model files in the page cache after the first run); a cold first start was not measured.

**Verdict.** Accepted. HTTP and the direct use cases agree on every document, chunk and question of the sample; the edge costs about 2 ms per request; with the recommended limit a burst is shed with 503 `service_busy` and `Retry-After` only; descriptors, temporary files, processes and threads return to their baseline after 400 mixed requests, and RSS plateaus after the first loaded round. Uvicorn exits with the SIGTERM it received (status -15 / 143) after a clean shutdown (`service stopped` and `Application shutdown complete` are logged), which is its normal behaviour.

---

## 14. NER pipeline pool (task 10.1 of `add-http-api`, design D10)

Scripts: `benchmarks/ner_pool/scripts/` (local, like every benchmark script; driver `run.sh`, see `benchmarks/README.md` step 8).

`TransformersNameRecognizer` used to run its one pipeline under a lock, which capped one process at about 45-48 `/api/extract` documents per minute (13.2). It now keeps `NER_POOL_SIZE` token-classification pipelines that share one model and one tokenizer; a call takes a free pipeline, the forward passes run in parallel (eval mode, no gradients), and only the tokenizer steps (tokenising a text, grouping the predicted tokens) run under one lock shared by the pool. This section measures the recogniser alone; the end-to-end effect is in section 15.

**Environment.** Image `docintel-nerpool` built (target `test`) from the tree with the pool (torch 2.14.1+cpu, transformers 5.21.0, tokenizers 0.23.3), run through `docker-compose.test.yml` without Qdrant (`run --no-deps`, project `nerpool`), same host and container view as section 13 (16 CPUs, nothing else running). Model `Davlan/xlm-roberta-base-ner-hrl` from the image. Inputs: the 60 documents of the seed-0 sample (section 2.8), read once with the production OCR engine without a language hint (`ocr_texts.py`); each page's NER input is cached exactly as the use case builds it (`title_case_caps(PageText.from_words(words).text)`): **101 page texts** (one page of the 102 has no text), 34 of them longer than the model's 512 tokens (so the 64-token-stride windowing runs). Scripts and raw outputs are in the session scratch folder (`ner_pool/scripts/{ocr_texts,correct,bench_one,bench_all,supp,control_shared_tokenizer,tok_mem,tables}.py`, outputs in `ner_pool/out/`; local, not committed).

### 14.1 Identical output (`correct.py`)

**Method.** Expected spans: a pool of 1 called sequentially on 111 texts (the 101 pages, plus 10 long texts of six pages joined, 1,972-3,230 tokens each, which exercise windowing and batching over many windows). Then a pool of 4 called from 16 threads on the same texts, in three rounds with different shuffles (seeds 0-2); every returned span list is compared for equality with the expected one. The pool's structure is read from the object.

| check | result |
|---|---|
| spans, pool 4 under 16 threads vs pool 1 sequential | **111 / 111 texts identical in each of the 3 rounds** (333 calls, 1,590 spans per round), 0 exceptions; all 60 documents identical in every round |
| structure | 4 pipelines, 1 model object, 1 weight tensor (`data_ptr`), 1 tokenizer (and 1 Rust tokenizer), model in eval mode |
| wall time | pool 1 sequential 86.2 s; pool 4 at 16 threads 31.1, 31.7, 32.0 s (2.7-2.8 x) |
| after `close()` | `ProcessingFailed('ner_failed', 'NER model is closed')` |

The rejected design A (each pipeline with its own tokenizer, no lock) also gave 111 / 111 identical texts in all three rounds (`correct_designA.json`). A negative control (`control_shared_tokenizer.py`: 4 pipelines sharing one tokenizer **without** the lock, 16 threads, 3 rounds) returned 333 / 333 identical results with no exception as well: on these inputs a concurrent-use failure of the tokenizer was **not reproduced**, so the lock is a precaution resting on the tokenizer switching its truncation and padding settings in place (the reason given in the recogniser's docstring), not on a failure observed here. It costs little: the tokenizer steps are a small part of a call, and pool 4 with the lock is not slower than design A without it (14.3).

### 14.2 Throughput: pool size x threads x concurrency (`bench_one.py`, `bench_all.py`)

**Method.** Each configuration runs in a fresh process: the recogniser is built with `pool_size` P and `threads` T (`torch.set_num_threads(T)`), every pipeline is warmed up once (P concurrent calls), then the 101 page texts are recognised from 1, 2, 4 and 8 threads pulling from a shared queue. Two passes: pass 1 with configurations and concurrencies ascending, pass 2 with both descending; cells are the mean of the two. "docs/min" here is the NER rate alone: the 60 documents' 101 page texts per minute of wall time (not comparable with the end-to-end docs/min of sections 13 and 15, which include OCR). "Process CPUs" is the process's CPU time over wall time.

| pool | threads | concurrent calls | docs/min (pass 1, pass 2) | mean | process CPUs | errors |
|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 34.1, 35.3 | 34.7 | 1.00 | 0 |
| 1 | 1 | 2 | 34.1, 35.6 | 34.9 | 1.00 | 0 |
| 1 | 1 | 4 | 34.4, 35.5 | 35.0 | 1.00 | 0 |
| 1 | 1 | 8 | 34.6, 35.6 | 35.1 | 1.00 | 0 |
| 1 | 2 | 1 | 65.6, 65.8 | 65.7 | 1.99 | 0 |
| 1 | 2 | 2 | 66.4, 65.9 | 66.2 | 1.99 | 0 |
| 1 | 2 | 4 | 64.0, 65.4 | 64.7 | 1.99 | 0 |
| 1 | 2 | 8 | 63.8, 66.9 | 65.3 | 1.99 | 0 |
| 2 | 1 | 1 | 36.1, 35.2 | 35.7 | 1.00 | 0 |
| 2 | 1 | 2 | 69.4, 68.3 | 68.9 | 2.00 | 0 |
| 2 | 1 | 4 | 69.4, 67.8 | 68.6 | 1.99 | 0 |
| 2 | 1 | 8 | 68.9, 68.8 | 68.9 | 1.99 | 0 |
| 2 | 2 | 1 | 64.8, 65.5 | 65.2 | 1.99 | 0 |
| 2 | 2 | 2 | 118.9, 117.3 | 118.1 | 3.91 | 0 |
| 2 | 2 | 4 | 116.8, 117.8 | 117.3 | 3.90 | 0 |
| 2 | 2 | 8 | 117.5, 117.7 | 117.6 | 3.90 | 0 |
| 4 | 1 | 1 | 35.6, 35.8 | 35.7 | 1.00 | 0 |
| 4 | 1 | 2 | 68.8, 69.5 | 69.2 | 1.99 | 0 |
| 4 | 1 | 4 | 128.7, 127.7 | 128.2 | 3.96 | 0 |
| 4 | 1 | 8 | 125.7, 126.1 | 125.9 | 3.88 | 0 |
| 4 | 2 | 1 | 65.4, 64.8 | 65.1 | 1.99 | 0 |
| 4 | 2 | 2 | 118.2, 117.5 | 117.8 | 3.91 | 0 |
| **4** | **2** | 4 | 185.8, 186.6 | **186.2** | 7.60 | 0 |
| 4 | 2 | 8 | 182.7, 182.7 | 182.7 | 7.49 | 0 |

Best cell per configuration: pool 1 / 1 thread 35.1 (-81 % of the best), 1 / 2 66.2 (-64 %), 2 / 1 68.9 (-63 %), 2 / 2 118.1 (-37 %), 4 / 1 128.2 (-31 %), **4 / 2 186.2**. The rate scales with pool size x threads (the CPUs the recogniser can use) up to about 7.6 CPUs; concurrency beyond the pool size adds nothing. At equal CPUs, more pipelines with one thread each are slightly faster than fewer with two (4 / 1: 128.2 against 2 / 2: 118.1 on four CPUs).

**Supplement: pool 8** (`supp.py`, same method, 8 and 16 concurrent calls, two interleaved passes against 4 / 2):

| pool | threads | concurrent calls | docs/min (pass 1, pass 2) | mean | process CPUs | errors |
|---|---|---|---|---|---|---|
| 4 | 2 | 8 | 174.0, 185.6 | 179.8 | 7.51 | 0 |
| 4 | 2 | 16 | 172.5, 188.6 | 180.5 | 7.62 | 0 |
| 8 | 1 | 8 | 203.7, 207.5 | 205.6 | 7.70 | 0 |
| 8 | 1 | 16 | 191.5, 198.9 | 195.2 | 7.42 | 0 |
| 8 | 2 | 8 | 236.3, 236.4 | 236.4 | 15.33 | 0 |
| 8 | 2 | 16 | 227.5, 228.7 | 228.1 | 14.95 | 0 |

Alone on the host, pool 8 / 2 threads is 31 % faster than 4 / 2 but uses all 16 CPUs (15.3), which a server shares with Tesseract; pool 8 / 1 thread is 14 % faster than 4 / 2 on the same ~7.6 CPUs. Which one helps end to end depends on what OCR leaves free, so section 15 measures it in the server.

### 14.3 Memory (RSS of the process, MB)

| configuration | after import | after load | after warm-up | peak (`VmHWM`) over the cells, pass 1 / pass 2 |
|---|---|---|---|---|
| pool 1, 1 thread | 229 | 987 | 1,023 | 1,368 / 1,460 |
| pool 1, 2 threads | 228 | 987 | 1,026 | 1,432 / 1,437 |
| pool 2, 2 threads | 228 | 987 | 1,051 | 1,465 / 1,508 |
| pool 4, 1 thread | 229 | 988 | 1,081 | 1,521 / 1,611 |
| **pool 4, 2 threads** | 229 | 987 | **1,093** | **1,588 / 1,627** |
| pool 8, 1 thread (supplement) | 229 | 987 | 1,206-1,225 | 1,723 / 1,853 |
| pool 8, 2 threads (supplement) | 229 | 987 | 1,190-1,196 | 1,874 / 1,878 |
| design A, pool 1, 2 threads (rejected) | 229 | 654 | 1,023 | 1,389 |
| design A, pool 2, 2 threads (rejected) | 229 | 951 | 1,347 | 1,779 |
| **design A, pool 4, 2 threads (rejected)** | 229 | 1,547 | **1,983** | **2,478** |

"After load" of the final design includes the one warm-up call that `load_pipelines` makes. The shared pool costs **+70 MB** after warm-up and +160-190 MB at peak for pool 4 against pool 1 with the same 2 threads (1,093 against 1,026 MB; 1,588 / 1,627 against 1,432 / 1,437 MB: the working memory of four parallel forward passes). Design A, a tokenizer per pipeline, costs about **300 MB per extra tokenizer** (654 -> 951 -> 1,547 MB after loading 1, 2 and 4; this model's tokenizer has a 250,002-token vocabulary): at pool 4, **+890 MB** after warm-up (1,983 against 1,093 MB; +893 MB against the 1,090 MB of the supplement's second pass) and +850 MB at peak (2,478 against 1,627 MB), for no gain in throughput (design A pool 4 / 2 threads: 171.5-174.3 docs/min at 4-8 calls, single pass, against 182.7-186.2) or in correctness (14.1). It was rejected for that reason.

**Choice (NER alone).** `NER_POOL_SIZE` = 4 with `NER_THREADS` = 2 was the default implemented: in the grid it is the fastest configuration (186.2 docs/min, **2.8 x** the 66.2 of the former single pipeline with 2 threads), with output identical to pool 1 and +70 MB resident. Pool 8 is faster when the recogniser has the host to itself (14.2 supplement). End to end, section 15 recommends pool 8: equivalent to 4 for short documents, 7 % faster for 50-page documents (15.6).

**Caveats.** NER alone on cached text: the end-to-end rate is lower, since OCR shares the CPUs (section 15). `torch.set_num_threads` is process-wide (recogniser docstring), so every torch computation in the process, the embedder's included, uses the last value set. Design A was measured in one pass only. The negative control did not reproduce the tokenizer failure (14.1). Host-specific: re-measure on the target hardware.

---

## 15. Concurrency with the NER pool (tasks 10.1 and 10.7 of `add-http-api`, design D7)

Scripts: `benchmarks/http_pool/scripts/` (local, like every benchmark script; driver `run.sh`, see `benchmarks/README.md` step 8).

With the NER lock gone (section 14) the job limit of 13.2 had to be re-measured, together with the pool size, on real Uvicorn; task 10.7 adds 50-page documents. **Environment**: as in section 13, except the image: `docintel-test` built fresh with `docker compose -p block5ev2 -f docker-compose.test.yml build tests` from the tree with the pool, the job cancellation on disconnect and the drain after an early 413 (settings `MAX_CONCURRENT_QUESTIONS`, `READINESS_TIMEOUT_S`, `LOG_EXCEPTION_MESSAGES` at their defaults), compose project `block5ev2`, torn down with `down -v --remove-orphans` after every run; 16 CPUs in the container (`os.cpu_count()` and affinity 16; the host's AMD Ryzen 7 7700X has 8 cores with two hardware threads each); nothing else ran on the host. Scripts and raw outputs are in the session scratch folder (`http_evidence2/scripts/{p15,p107,p15res,ev2,tables15}.py` with `ev_common.py` and `run.sh` reused from section 13; outputs and server logs in `http_evidence2/out/`; local, not committed).

### 15.1 Method

Same request set and client as 13.2 (`p15.py`, derived from `p82.py`): one Uvicorn process per configuration with `MAX_CONCURRENT_JOBS` J, `NER_POOL_SIZE` P (optionally `OCR_WORKERS`, `NER_THREADS`) and `BUSY_WAIT_S=300` (the new maximum; nothing was rejected); the 25 sample documents (45 pages) of 13.2 sent as `POST /api/extract` with their name lists and no language hint, from c client threads (one `httpx.Client` each) pulling from a shared queue. Warm-up per server: 2 sequential requests, then 8 concurrent ones (so every pipeline of the pool has run once), not counted. Clients 8 and 16, and 24 for J >= 12. Two passes, configurations in reverse order in the second; the decisive configurations again with the set three times (75 requests). Per cell: docs/min = requests / wall; client latency p50 / p95 (it includes waiting for a slot); server peak RSS (`VmHWM`, reset per cell, and the 0.25 s sampled maximum, whichever is larger); peak of the server plus its Tesseract children; most Tesseract processes at once; CPU busy share of all 16 CPUs (`/proc/stat`); and from the use case's own `extraction completed` lines of the cell: `ocr_ms` and `ner_ms` summed over the jobs and divided by their pages (`ner_ms` includes waiting for a free pipeline), and the mean `duration_ms` (time a job holds its slot). Every request of every cell returned 200.

**Sampling artefact in the process-tree peak.** Starting a Tesseract process forks the 2-3 GB server; for a moment the child (still named `uvicorn`, or already `tesseract` in the middle of `exec`) reports the parent's shared pages as its own RSS (2.1-2.7 GB). A sample that catches it adds that to the tree: such samples gave 4.6-6.2 GB "peaks" in about a third of the busy cells, while a running Tesseract never exceeded 127 MB. The tree peaks below leave those samples out (server RSS + RSS of the processes named `tesseract` below 500 MB); where only maxima were recorded, the bound server peak + peak Tesseract sum is used (marked †). The 5,228 MB peak of 13.2 (limit 16) was most likely the same artefact, which the 13.2 sampler could not tell apart.

### 15.2 Job limit x pool size (short documents)

25-request set, mean of two passes (docs/min at 8 / 16 / 24 clients; the other columns at 16 clients; peaks are the maxima over all clients of the configuration):

| jobs | pool | docs/min at 8 / 16 / 24 clients | CPU busy | OCR ms / page | NER ms / page | server peak (MB) | server + Tesseract peak (MB) |
|---|---|---|---|---|---|---|---|
| 4 | 1 | 43.2 / 44.2 / - | 28 % | 1,710 | 1,103 | 1,878 | 2,238 |
| 8 | 1 | 45.3 / 45.5 / - | 30 % | 1,840 | 3,209 | 2,055 | 2,845 |
| 12 | 1 | 45.0 / 45.9 / 46.1 | 31 % | 2,114 | 4,835 | 2,267 | 3,451 |
| 16 | 1 | 45.1 / 45.0 / 45.0 | 32 % | 2,628 | 6,313 | 2,456 | 3,877 |
| 4 | 4 | 50.7 / 50.3 / - | 33 % | 1,750 | 734 | 1,903 | 2,269 |
| 8 | 4 | 72.6 / 73.3 / - | 58 % | 2,166 | 1,042 | 2,062 | 2,825 |
| 12 | 4 | 73.1 / 81.0 / 82.5 | 67 % | 2,490 | 1,617 | 2,276 | 3,364 |
| 16 | 4 | 72.3 / 84.2 / 84.3 | 70 % | 2,865 | 2,381 | 2,449 | 3,915 |
| 12 | 8 | 74.1 / 79.6 / 82.8 | 69 % | 2,588 | 1,538 | 2,373 | 3,695 |
| 16 | 8 | 74.1 / 85.7 / 85.5 | 77 % | 3,116 | 2,130 | 2,461 | 4,027 |
| 16 | 8, `NER_THREADS` 1 (invalid: see note) | 73.9 / 85.7 / 85.9 | 77 % | 3,081 | 2,102 | 2,464 | 3,997 |
| 24 | 8 | 74.4 / 84.4 / 87.1 | 76 % | 3,042 | 2,106 | 2,756 | 4,426 |

Pool 1 (the former single pipeline) reproduces 13.2: 43-46 docs/min at any job limit, NER per page (lock wait included) growing from 1.1 to 6.3 s while CPUs stay 70 % idle. With the pool, throughput rises to 82-87 docs/min. With 8 clients every limit from 8 up gives 72-74 docs/min (only 8 requests are in flight); the limits differ from 16 clients on. The pool-8 run repeated J 12 and 16 with pool 4 as controls (72.8 / 81.9 / 82.0 and 72.5 / 83.5 / 82.5), within 1-2 % of the grid. The `NER_THREADS` 1 row is invalid: the review of task 10.x found that `torch.set_num_threads` is process-wide and the embedder, built after the recogniser, set it to `EMBEDDING_THREADS` (2), so that run used 2 threads like the others; the settings were then merged into one `TORCH_THREADS`. Thread counts other than 2 were measured only for NER alone (section 14).

**75-request runs** (the decisive pairs; 16 and 24 clients; † = bound, see 15.1):

| jobs | pool | OCR workers | clients | docs/min (passes) | mean | p50 / p95 (s) | mean job in slot (s) | server peak (MB) | server + Tesseract peak (MB) | max Tesseract | CPU busy | OCR ms / page | NER ms / page |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 4 | 4 | 16 | 78.9, 79.0 | 78.9 | 11.4 / 13.8 | 5.8 | 2,100 | 2,691 | 11 | 63 % | 2,140 | 1,076 |
| 8 | 4 | 4 | 24 | 78.6, 78.8 | 78.7 | 17.0 / 19.4 | 5.8 | 2,104 | 2,837 | 11 | 63 % | 2,142 | 1,081 |
| 12 | 4 | 4 | 16 | 88.2, 84.9, 88.3, 90.1 | 87.9 | 10.2 / 13.0 | 7.7 | 2,260 | 3,533 † | 15 | 76 % | 2,484 | 1,810 |
| 12 | 4 | 4 | 24 | 88.3, 85.5, 88.5, 89.5 | 87.9 | 15.1 / 18.2 | 7.7 | 2,265 | 3,426 | 16 | 76 % | 2,473 | 1,817 |
| 12 | 4 | 3 | 16 | 88.3, 89.3 | 88.8 | 10.1 / 12.6 | 7.6 | 2,260 | 3,126 | 14 | 76 % | 2,439 | 1,791 |
| 12 | 4 | 3 | 24 | 82.0, 89.5 | 85.7 | 15.6 / 18.5 | 8.0 | 2,289 | 3,375 | 15 | 76 % | 2,540 | 1,883 |
| 12 | 4 | 5 | 16 | 86.6, 89.1 | 87.8 | 10.0 / 13.3 | 7.7 | 2,317 | 3,245 | 15 | 76 % | 2,486 | 1,817 |
| 12 | 4 | 5 | 24 | 88.6, 88.8 | 88.7 | 14.9 / 18.1 | 7.7 | 2,300 | 3,451 | 14 | 76 % | 2,439 | 1,820 |
| **12** | **8** | 4 | 16 | 91.0, 91.5 | **91.2** | 9.8 / 13.0 | 7.5 | 2,303 | 3,180 | 15 | 79 % | 2,477 | 1,662 |
| 12 | 8 | 4 | 24 | 90.6, 90.8 | 90.7 | 14.6 / 17.6 | 7.5 | 2,318 | 3,474 | 16 | 78 % | 2,476 | 1,665 |
| 16 | 4 | 4 | 16 | 88.8, 89.4 | 89.1 | 9.7 / 14.8 | 10.1 | 2,377 | 3,957 † | 18 | 77 % | 2,596 | 3,018 |
| 16 | 4 | 4 | 24 | 89.5, 88.3 | 88.9 | 14.3 / 21.4 | 10.2 | 2,415 | 3,849 | 18 | 77 % | 2,628 | 3,022 |
| 16 | 8 | 4 | 16 | 89.2, 89.3 | 89.3 | 10.2 / 14.3 | 10.0 | 2,504 | 4,103 † | 19 | 82 % | 2,955 | 2,589 |
| 16 | 8 | 4 | 24 | 89.0, 88.3 | 88.6 | 14.4 / 19.6 | 10.1 | 2,568 | 4,047 † | 19 | 82 % | 2,972 | 2,620 |
| 12 | 8, `MALLOC_ARENA_MAX=2` | 4 | 16 | 91.9, 92.3 | 92.1 | 10.0 / 12.4 | 7.4 | 2,208 | 3,131 | 14 | 79 % | 2,467 | 1,636 |
| 12 | 8, `MALLOC_ARENA_MAX=2` | 4 | 24 | 91.0, 91.2 | 91.1 | 14.8 / 17.2 | 7.5 | 2,194 | 3,472 | 14 | 79 % | 2,492 | 1,652 |

The runs were made in four sessions (J 12 / 16 with pools 4 and 8; J 8 with J 12 / pool 4 and the `OCR_WORKERS` variants; J 12 / pool 8; the same with `MALLOC_ARENA_MAX=2`), each with two interleaved passes; J 12 / pool 4 appears in two sessions (88.2-90.1 except one pass at 84.9-85.5), so differences of 1-3 % between sessions are within run-to-run variation. A single request is as fast as before: the first warm-up requests of every server (2-page documents) took 3.5-3.8 s, as in 13.2. Start-up to readiness: 10.2-10.7 s with pool 4 or 8 (56 starts, the first, cold start of the session at 12.6 s excluded), as in 13.3.

**`OCR_WORKERS`** (default `min(4, cpu_count)` = 4), tried one step down and up at J 12 / pool 4 because the CPUs were not saturated: 3 gives 88.8 / 85.7 and 5 gives 87.8 / 88.7 docs/min against 87.9 / 87.9, inside the pass-to-pass spread (one pass of 3 at 24 clients at 82.0). Short documents use at most 2-3 workers per job anyway. No change is recommended.

### 15.3 50-page documents (task 10.7)

**Documents.** One 50-page PDF per language (`p107.py`), made of the language's sample documents' pages in sample order, cycled, with the original page images copied unchanged (150-DPI grey JPEG scans, PyMuPDF `insert_pdf`), and the union of the included documents' name lists (16-45 names). Sizes 11.6-15.7 MB (en 11.6, es 13.2, pt 13.9, de 14.4, fr 15.7 MB), under `MAX_UPLOAD_MB` (20). The 1-2 MB first envisaged would need about 30 KB per page, i.e. re-encoding to about 75 DPI or bilevel images, unlike the corpus scans; the larger files are the stricter case for memory (each job holds its upload in memory). **Server**: `MAX_CONCURRENT_JOBS=12` (the recommended limit), `BUSY_WAIT_S=300`, pool 4 or 8, after the same warm-up. **Parts**: (a) 12 clients, each sending two long documents one after the other (24 requests, 1,200 pages); (b) 24 clients sending one long document each at once (12 wait for a slot); (c) mixed: 6 clients each send one long document while 6 clients send the short 25-document set in a loop until the long ones are done. Samples every 0.25 s as in 15.1.

| part | pool | wall (s) | requests | latency long p50 / p95 / max (s) | short served, p50 / p95 / max (s) | server peak (MB) | server + Tesseract peak (MB) | max Tesseract | CPU busy | long: OCR / NER ms per page |
|---|---|---|---|---|---|---|---|---|---|---|
| a | 4 | 367.5 (369.9 in an earlier run) | 24 x 200 | 159 / 251 / 301 | - | 2,627 | 3,408 | 16 | 60 % | 1,050 / 2,172 |
| a | 8 | 344.9, 342.2 | 24 x 200, 24 x 200 | 146-152 / 244-251 / 260-295 | - | 2,700-2,715 | 3,442-3,491 | 15 | 75-76 % | 1,188-1,204 / 1,861-1,923 |
| b | 4 | 375.5 | 24 x 200 | 271 / 366 / 375 | - | 2,860 | 3,857 | 16 | 59 % | 1,074 / 2,186 |
| c | 4 | 140.6, 137.5 | 6 + 74, 6 + 72 x 200 | 97-99 / 133-135 / 133-136 | 74 and 72; 7.6-8.7 / 26.5-33.1 / 44 | 2,428-2,862 | 3,336-3,846 | 16 | 63-64 % | 1,105-1,167 / 956-972 |
| c | 8 | 140.7, 141.3 | 6 + 89, 6 + 92 x 200 | 129-131 / 136-137 / 137 | 89 and 92; 8.6-8.9 / 14.8-15.0 / 17.5 | 2,498-2,806 | 3,246 (one pass †: about 3.0 GB) | 15 | 78 % | 1,143-1,171 / 1,392-1,393 |

The "earlier run" of part a is the first attempt of the same script, which completed every request but crashed in the sampler afterwards (fixed); its wall time is taken from the server log (first job start to last job end, 369.9 s), so it has no memory figures.

**Findings.** Every request returned 200; no error, warning other than the request-failure lines, or timeout was logged: each 50-page job's whole OCR took 18-83 s with 4 workers, far from `OCR_TIMEOUT_S` (60 s per Tesseract call). A job holds its slot 47-347 s (mean 152-163 s in parts a and b), so with the default `BUSY_WAIT_S` of 10 s, part b's 12 waiting requests would have been answered 503 `service_busy` after 10 s (here they waited, under the 300 s used for the measurement). **Memory stays bounded by the job limit**: the server peaked at 2.6-2.9 GB and server + Tesseract at **3.9 GB at most**; in part b, 12 requests waiting with their 12-16 MB uploads did not raise the peak beyond part a's (part b started at 2,616 MB, retained from part a, and peaked at 2,860 MB; part a rose from 2,221 to 2,627 MB). Tesseract never exceeded **16 processes** (127 MB each at most, 1.3 GB together), although 12 jobs x 4 workers would allow 48: each job reads all its pages first and then runs NER page by page, so at any time part of the jobs are in their NER phase (no Tesseract process ran at all in 30 % of part a's samples), and pages reach Tesseract only as fast as they are rendered, one at a time under the process-wide PDFium lock (a likely contributor, not measured separately). Threads peaked at 90.

With long documents, **NER is the bottleneck again**: 2.2 s of `ner_ms` per page (waiting for one of 4 pipelines included) against 1.05 s of OCR, with 40 % of the CPUs idle. Pool 8 brings it to 1.9 s per page, uses 75 % of the CPUs and finishes part a **7 % sooner** (342-345 s against 367-370 s). In the mixed run it serves **24 % more short documents** in the same time (89-92 against 72-74) and **halves their p95** (15 s against 27-33 s; max 17.5 s against 44 s), while the long documents take longer (p50 129-131 s against 97-99 s): with 8 pipelines, short documents' pages wait less behind long documents' 50 pages. Pages per minute in part c: 196-198 with pool 8, 185-188 with pool 4. The server peak was no higher with pool 8 (2.5-2.8 GB against 2.4-2.9 GB; section 14 measured +100-250 MB for the NER part alone).

### 15.4 Where the bottleneck is now

| configuration | OCR ms / page | NER ms / page | CPU busy | docs/min |
|---|---|---|---|---|
| one request, unloaded (warm-up, 2-page documents) | 1,240-1,330 | 530-580 | - | - |
| pool 1, J 8, 16 clients (former behaviour) | 1,840 | 3,209 | 30 % | 45.5 |
| pool 4, J 8, 16 clients, 75 requests | 2,140 | 1,076 | 63 % | 78.9 |
| pool 4, J 12, 16 clients, 75 requests | 2,484 | 1,810 | 76 % | 87.9 |
| pool 8, J 12, 16 clients, 75 requests | 2,477 | 1,662 | 79 % | 91.2 |
| pool 8, J 16, 16 clients, 75 requests | 2,955 | 2,589 | 82 % | 89.3 |
| 50 pages, pool 4, J 12 (part a) | 1,050 | 2,172 | 60 % | 3.9 |
| 50 pages, pool 8, J 12 (part a) | 1,188-1,204 | 1,861-1,923 | 75-76 % | 4.2 |

For short documents the NER lock is gone as a bottleneck: per-page NER time no longer explodes (1.7-1.8 s at J 12, against 3.2-6.3 s with one pipeline) and throughput doubles (45.5 -> 87.9-91.2 docs/min). What limits it now is the **CPU shared by Tesseract and NER**: beyond J 12 both stages only get slower per page (OCR 2.5 -> 3.0 s, NER 1.7 -> 2.6 s from J 12 to J 16 with pool 8) at the same throughput, and more jobs (16, 24), more pipelines (8), more or fewer OCR workers (3, 5) or one NER thread per pipeline all stay within 88-92 docs/min. The CPUs are 76-82 % busy over a run, ramp-up and tail included, i.e. 12-13 of 16 logical CPUs; the host has 8 physical cores with two hardware threads each, so the remaining share is mostly second hardware threads of busy cores (an inference; per-core load was not measured). For long documents the bottleneck is NER, because a job does its NER page by page after its OCR and the pool is shared by all jobs (15.3). More throughput on this host would need less CPU per page (a faster NER runtime, ONNX or quantisation; fewer Tesseract passes, e.g. a language hint, which skips the detection read) or more hosts.

### 15.5 Resources and cancellation at the recommended settings (`p15res.py`)

**Method.** As 13.3, at `MAX_CONCURRENT_JOBS=12`, `NER_POOL_SIZE` 8 (and 4 for comparison), `MAX_UPLOAD_MB=1`, no language model, with two changes: 20 of the 100 valid requests of the 13.3 mix are replaced by **clients that disconnect mid-job** (10 `/api/extract`, 10 `/api/ingest`, 2-3-page sample documents: a raw socket sends the whole request, waits 0.3, 0.8, 1.5, 2.2 or 3.0 s and closes without reading), and `BUSY_WAIT_S=2`, so that a job slot never freed would show up as 503 `service_busy` (the 4 client threads keep at most 4 jobs plus cancelled ones still finishing a step). After each round: snapshots right after and after 10 s idle (descriptors, threads, RSS, `/tmp`, Tesseract processes anywhere in the container), then a **slot check**: 12 simultaneous extractions released by a barrier must all get 200 (one leaked slot would make one of them wait more than 2 s, then 503). Four rounds per run (the first run had two).

| run | RSS after warm-up | RSS after rounds 1 / 2 / 3 / 4 (10 s idle) | max during rounds: RSS, server + Tesseract, fds, threads, Tesseract | after each round: fds / threads / `/tmp` entries / Tesseract |
|---|---|---|---|---|
| pool 8 | 1,865 | 2,497 / 2,856 / 3,068 / 3,299 | 3,457, 3,763, 32, 60, 5 | 15 / 38 / 2 / 0 |
| pool 8, earlier 2-round run | 1,865 | 2,458 / 2,925 | 2,925, 3,373, 34, 61, 6 | 15 / 38 / 2 / 0 |
| pool 4 | 1,855 | 2,456 / 2,845 / 3,113 / 3,129 | 3,335, 3,711, 33, 60, 5 | 15 / 38 / 2 / 0 |
| pool 8, no disconnecting clients | 1,864 | 2,366 / 2,884 / 3,040 / 3,060 | 3,282, 3,667, 33, 61, 6 | 15 / 36 / 2 / 0 |
| pool 8, `MALLOC_ARENA_MAX=2` | 1,815 | **2,302 / 2,461 / 2,454 / 2,462** | 2,462, 2,921, 32, 60, 5 | 15 / 38 / 2 / 0 |

**Cancellation.** In each of the 14 rounds with disconnecting clients, 19 of the 20 were cancelled: **19 access lines with status 499** (10 `/api/extract`, 9 `/api/ingest`), 9-10 `extraction cancelled` and 9 `ingestion cancelled` lines (a 499 without a use-case line is a request whose job had not started, which the use case never sees; not checked further); the 20th had already received its 200 when its client closed (a job finished within 3.0 s). The request ended 0.04-0.07 s (median) and at most 0.47 s after the client closed (499 duration minus the client's waiting time, matched in sorted order). **No Tesseract process outlived its job** (0 in the container after every round and after shutdown), and **every slot check returned 12 x 200** (4.6-9.0 s) with no 503 `service_busy` anywhere, so cancelled jobs free their slots. Every other request got the status and code of 13.3 (400, 413, 415, 422 as documented; 200; 503 `llm_unavailable` only for questions that need the model, 6 then 10 per round); no other 5xx.

**Resources.** Descriptors return to 15 (14 right after start-up, one more socket after the warm-up) and the `/tmp` listing never changes (the two pre-existing entries); threads settle at 36-38 when idle (the 12 job threads are kept; 52 right after a slot check). **RSS does not plateau as in 13.3** (where the second round added 11 MB at limit 8 with one pipeline): it grows by 360-520 MB in round 2 and by 16-270 MB per round after that, about the same with or without disconnecting clients (so it is not the cancellation path) and with pool 4 or 8 (pool 4 had nearly stopped: +16 MB in round 4). With `MALLOC_ARENA_MAX=2` it stops after the first round (2,461 -> 2,454 -> 2,462 MB), about 840 MB lower after four rounds, so the growth is glibc keeping per-thread malloc arenas for the 12 job threads and the NER threads, not a leak. The cap also cost nothing in throughput (J 12 / pool 8: 91.1-92.1 docs/min against 90.7-91.2, server peak 2.2 GB against 2.3 GB, 15.2). Setting it in the runtime image (block 6), already suggested in 13.3, is now clearly worth it.

### 15.6 Choice

D7's rule, the smallest setting within 5 % of the best throughput, applied to both workloads:

- **Short documents** (75-request runs): best 91.2 docs/min (J 12 / pool 8); the 5 % band starts at 86.6. J 8 reaches 78.9 (-13.5 %), outside; **J 12 is the smallest limit inside** (87.9 with pool 4, 91.2 with pool 8); 16 and 24 add nothing (88.6-89.3; 87.1 on the short set). Both pools are inside at J 12.
- **50-page documents** (J 12): pool 4 needs 367-370 s for part a against 342-345 s with pool 8, i.e. **6.8 % less throughput, outside the band**; pool 8 is inside on both workloads, and under mixed load it serves 24 % more short documents with half the p95.

**Applied: `MAX_CONCURRENT_JOBS` = 12, `NER_POOL_SIZE` = 8, `TORCH_THREADS` = 2 (the former `NER_THREADS` = `EMBEDDING_THREADS` = 2), `OCR_WORKERS` unset (min(4, CPUs) = 4, unchanged), `BUSY_WAIT_S` = 10 s (unchanged).** If the traffic is short documents only, pool 4 is equivalent (within 1-4 %) and saves the extra pipelines' memory (+100-250 MB for NER alone in section 14; not visible in the server peaks here). `BUSY_WAIT_S`, by the reasoning of 13.2 (about one loaded job's duration): at J 12 a job holds its slot 7.4-7.7 s on average and a request at 16 clients waits and runs 9.6-10.8 s (p50), so 10 s is still about one loaded job; with jobs finishing at staggered times a slot frees about every 7.5 s / 12 = 0.6 s, so about 16 queued requests are admitted before the service sheds load; a served short request completes within about 18 s (p95 at 24 clients), under common 30 s proxy timeouts. 50-page jobs hold a slot 47-347 s, so a burst of long documents is shed after 10 s, the intended behaviour (clients retry after `Retry-After`). **Expected envelope** at these settings on this host: 88-92 short docs/min per process; in the throughput runs the server peaks at 2.2-2.3 GB and server + Tesseract at 3.5 GB with short documents, and at 2.6-2.9 GB and 3.9 GB with 50-page documents; without `MALLOC_ARENA_MAX=2` the server's RSS keeps growing by allocator retention (3.3-3.5 GB after four resource rounds, 15.5), with it it stayed at 2.5 GB.

**Caveats.** One host, synthetic scans, no language hint (detection reads the first pages twice); the client ran in the server's container and shares its CPUs. Configurations measured in different sessions differ by up to 3 % for the same setting; the 50-page comparison rests on two passes per pool (part a) and two passes of part c, part b on one pass (pool 4). Pool 8 at J 8 and J 12 with 50-page documents at other limits were not measured. The tree peaks marked † are upper bounds. The PDFium lock as a limit on concurrent Tesseract processes and the reading of the idle CPU share as second hardware threads are inferences, not measurements. `KNOWN_LIMITATIONS.md` P4 (one process tops out at 45-48 docs/min behind the NER lock) describes the state before the pool.
