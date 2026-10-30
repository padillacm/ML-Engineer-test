# Benchmark evidence

Reference record of the measurements behind the defaults of the service (OCR, NER, fuzzy matching, embeddings, chunking, refusal threshold) and of the capacity estimate. All figures are copied from the result files of the original benchmark session; nothing is re-estimated here except where a derivation is stated explicitly. The scripts that produced the numbers are kept outside this repository; the `benchmarks/...` paths below name them as they were run. The capacity estimate is recomputed by `docs/capacity_estimate.py` (standard library only).

Conventions: percentages unless stated; `[a-b]` or `(a-b)` is the 95 % percentile-bootstrap confidence interval reported by the source (resampling documents for the 600-document corpus, 200 documents for the name lists, sentences for the public NER sets, queries for the retrieval sets; 1,000 resamples, seed 0, 500 for the refusal tables). A cell without an interval means the source reports none. "Source" names the result file in the original session folder (`bench/...`); neither those folders nor the scripts are part of this repository.

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
16. Language model choice (`add-deployment` tasks 5.1 and 5.2): candidates, context length, answer quality, thinking mode, audit, concurrency, hosted reference
17. Deployed stack (`add-deployment` task 6.1): image sizes, extract and ingest throughput with CPU limits, sizing, questions, cold start, parity, smoke suite
18. Embedder pool and concurrency derived from the CPUs (`add-deployment` task 6.2): identical vectors, pool size, questions during ingestion, ingestion through HTTP under CPU quotas, the rule, FIFO hand-off of the NER and embedder slots
19. Deployed stack after task 6.2 and the FIFO hand-off (`add-deployment` task 6.3): derived defaults at 16, 8 and 4 CPUs, questions during ingestion, cold start with `start_interval`, parity, smoke suite
20. Audit viewer in headless Chromium (`add-audit-viewer` tasks 5.1-6.2): browser suite, frame tests, ink-check calibration on the 60-document sample with every failure classified, pdf.js against PDFium, viewer performance for 1, 10 and 50 pages, image sizes
21. Low-confidence OCR names (`add-audit-viewer` task 9): word filter before recognition against rules on candidate names on the 600-document corpus, the adopted rule end to end
22. Defensive hardening (`add-defensive-hardening` task 7.2): answer quality, support share, detector precision and recall, cost per page and per question, before and after
23. In-process models on the GPU (`add-gpu-inference` task 4.1): VRAM, per-stage times, per-component impact, GPU pool calibration, questions with Gemma on the shared GPU, parity, images and start-up

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
| `MAX_CONCURRENT_JOBS` | **12** on 16 logical CPUs, one process (derived from the CPUs when unset since task 6.2, 18.5) | 15.2 (D7 rule: smallest limit within 5 % of the best throughput); before the NER pool: 8 (13.2) | with the NER pool, 75-request runs (two passes or more each): 87.9 docs/min at 12 / pool 4 and 91.2 at 12 / pool 8, against 78.9 at 8 (-13.5 % of the best, 91.2); 16 and 24 are no faster (88.6-89.3; 87.1 on the short set); host-specific, re-measure on the target hardware |
| `NER_POOL_SIZE` (`TORCH_THREADS` 2) | **8** on 16 CPUs (derived when unset, 18.5); free pipelines go to waiting calls in arrival order (18.6) | 14 (NER alone: identical spans, +70 MB), 15.2, 15.3 and 15.6 (D7 rule over both workloads) | short documents: pools 4 and 8 within 4 % of each other (87.9 and 91.2 docs/min at 12 jobs, run-to-run spread 1-3 %); 50-page documents at 12 jobs: 24 documents in 342-345 s with 8 against 367-370 s with 4 (pool 4 is 7 % slower, outside the 5 % band); under mixed load short-document p95 15 s against 27-33 s; server peak +40-90 MB |
| `MAX_CONCURRENT_JOBS` / `NER_POOL_SIZE` on fewer CPUs | derived when unset since task 6.2 (18.5), following the sizing table (16 CPUs 12 / 8, 8 CPUs 6 / 4, 4 CPUs 3 / 2) | 17.3, 17.4 (deployed runtime image, `cpus:` quota), 19.2 (derived, deployed) | defaults confirmed on 16 CPUs (85.3 docs/min, test-image control 85.4); with `cpus: 8` the former fixed defaults logged the oversubscription warning and gave 54.0 docs/min against 61.4 for 6 / 4; with `cpus: 4` 27.9 against 37.5 for 3 / 2; derived and unset (19.2): 84.0 / 62.0 / 37.3 docs/min extraction at 16 / 8 / 4 CPUs, no warning |
| `EMBEDDING_POOL_SIZE` (`TORCH_THREADS` 2) | derived when unset: CPUs / (2 x `TORCH_THREADS`), at most 4 (**4** on 16 CPUs); one copy of the model, one more slot reserved for queries, which also go before waiting passages; queries and passages each served in arrival order (18.6) | 18.1-18.5 (D7 rule on `/api/ingest` docs/min) | vectors bit-identical to a pool of 1 (780 / 780 calls); 16 CPUs: 86.3 docs/min at pool 4 against 45.8 for the former single embedder, 70.4 for pool 2 and 84.2-85.3 for 6 and 8; deployed with the derived pools (19.2): 90.7 / 65.9 / 38.1 docs/min ingestion at 16 / 8 / 4 CPUs against 50.9 / 49.8 / 39.2 before the pool (17.3); 8 CPUs: pool 2 63.8 (pool 1 47.7, pool 4 62.4); 4 CPUs: pool 1 37.1 (pool 2 37.3); a question during ingestion embeds in 53-81 ms with the reserved slot against 0.3-1.2 s (p50) with priority alone, and is starved without priority; +110 MB after warm-up at pool 4 |
| Derived concurrency (`MAX_CONCURRENT_JOBS`, `NER_POOL_SIZE`, `EMBEDDING_POOL_SIZE`, `OCR_WORKERS` unset) | `concurrency_for(cpus, T)` in `app/bootstrap.py`: jobs = 3/4 of the CPUs rounded up, at most 12; NER pool = CPUs / T, at most 8; embedder pool = CPUs / (2 T), at most 4; OCR workers = CPUs, at most 4; each at least 1; CPUs = affinity lowered to the cgroup v2 quota; a set value wins; logged once at start-up (`concurrency in use`) | 17.3, 17.4, 18.5, 19.1-19.2 | 16 CPUs 12 / 8 / 4 / 4, 8 CPUs 6 / 4 / 2 / 4, 4 CPUs 3 / 2 / 1 / 4 (the measured points; other counts follow the rule, 18.5 table), read from the deployed stack's `concurrency in use` line at all three (19.1); no derived pool logs the oversubscription warning |
| `OCR_WORKERS` | derived when unset: min(4, CPUs available), the quota included (formerly the adapter's min(4, `os.cpu_count()`), which ignores a quota) | 15.2, 17.3, 18.5 | 3, 4 and 5 workers within the run-to-run spread on 16 CPUs (15.2); 4 was used at the 8- and 4-CPU quotas of 17.3 and 18.4 (the container sees 16 CPUs), which the rule keeps; fewer than 4 CPUs not measured |
| `BUSY_WAIT_S` | 10 s (unchanged; re-derived with the pool) | 13.2, 15.2 (derived from measured latency, not an optimum) | before the pool: one loaded job at limit 8 (p50 10.1 s, 13.2); with the pool at the recommended 12 jobs: a job holds its slot 7.4-7.7 s on average and p50 latency at 16 clients is 9.6-10.8 s, so 10 s is still about one loaded job; a 50-page job holds a slot 47-347 s, so a burst of long documents is shed after 10 s (intended; clients retry after `Retry-After`) |
| `TORCH_THREADS` | 2 (one process-wide value for the recogniser and the embedder) | 3 and 6 (CPU latency; both models were benchmarked at 2 threads), 14.2 (NER alone: threads 1 vs 2) | NER alone 65.7 docs/min at 2 threads vs 34.7 at 1 (pool 1); the former `NER_THREADS` / `EMBEDDING_THREADS` were merged because torch threads are process-wide (15.2 note); above 2 not measured end to end |
| `MALLOC_ARENA_MAX` | 2 (Dockerfile) | 13.3, 15.5 (resources, 200 mixed requests per round) | RSS flat after round 1 (2,454-2,462 MB) instead of growing 16-520 MB per round, about 840 MB lower, at the same throughput (91.1-92.1 docs/min) |
| `MAX_CONCURRENT_QUESTIONS` | 16 (unchanged) | 16.7 (proxy, test image), 17.5 (deployed stack, `gpu` profile, e4b), 19.3 | 16 clients: 123-125 questions/min (16.7) and 128-136 on the deployed stack, p95 9.6-9.7 s, maximum 12.2 s, every request 200; a question costs about 50 ms of embedding plus the model's wait (I/O); during a 16-client ingestion load 0.2-0.5 s of embedding at 16 CPUs (client p95 2.3 s) and 78-174 ms at 8 (19.3) |
| `LLM_READ_TIMEOUT_S` / `LLM_TOTAL_TIMEOUT_S` | 60 s / 90 s (unchanged) | 16.7, 17.5, 17.6, 19.4 | loaded maximum of a question 12.2 s at 16 clients (a fifth of 60 s); first question after a start 0.9-1.4 s (0.93-1.20 s in 19.4), including a first-ever start, thanks to the `ollama-pull` warm-up and `OLLAMA_KEEP_ALIVE=-1` (65 s without them, 16.7) |
| Ollama serving (`gpu` profile) | `OLLAMA_NUM_PARALLEL` 2, `OLLAMA_KEEP_ALIVE` -1, `OLLAMA_CONTEXT_LENGTH` 8192, `OLLAMA_MAX_LOADED_MODELS` 1, warm-up prompt in `ollama-pull`, CUDA kernel cache in the volume | 16.3, 16.7 (choice), 17.5, 17.6 (confirmed deployed), 19.4 | read from the running stack; 128-136 questions/min at 16 clients, VRAM 5.9-6.5 GB device total; start with fresh containers 15.8-19.4 s with `start_interval` (19.7-23.4 s before), first-ever start 85 s |
| Compose healthchecks (Qdrant, Ollama) | `interval: 5s`, `start_interval: 1s` during a 30 s `start_period` | 17.6, 19.4 | the app starts 1.7 s after Qdrant instead of 5.7 s; a start without a model 13.4-13.6 s instead of 17.4-20.1 s; Ollama healthy 2.8-3.5 s after its start instead of 5.9-6.1 s |
| `READINESS_TIMEOUT_S` | 2 s (at most 30) | not measured (judgement) | a healthy store answers its ping in milliseconds; 2 s is below the 3 s compose healthcheck timeout; the review measured a hung store's ping cut from 10.0 s to 2.0 s |
| Drain after an early 413 | up to 1 MiB over the route limit, at most 5 s | `add-http-api` 10.4 (`tests/api/test_uvicorn_edge.py`) | 900 KiB over: the client reads the 413 (reset without the drain); 5 s is nginx's `lingering_timeout`; far-over bodies are closed after about 1 MiB |
| Body limits | uploads `MAX_UPLOAD_MB` + 1 MiB + 64 KiB; other routes 64 KiB | `add-http-api` 10.6 and review finding 2 | 1 MiB is the multipart parser's per-field limit (no accepted `names` field is cut); a 2,000-character question fully escaped is about 24 KB |
| `MAX_UPLOAD_MB`, `MAX_PAGES`, `MAX_NAMES`, name length | 20 MB, 50 pages, 1,000 names, 100 characters | product limits agreed in the `add-http-api` proposal, not optimised; 15.3 checks the worst case | 24 concurrent 50-page jobs (11.6-15.7 MB files): all 200, server peak 2.6-2.9 GB, no OCR timeout |
| Ink check of the audit viewer's browser tests (`tests/e2e/ink.py`) | a box passes when at least **16 %** of its pixels are ink and the ink share inside is at least **2.1** times that of a ring of equal area around it; ink = 40 grey levels below the median of the box's neighbourhood, table rules excluded | 20.3 (calibration on the 60-document sample, wrong-frame controls) | 763 / 770 boxes matching a truth mention pass (99.1 %), 793 / 805 of all boxes (98.5 %); boxes moved by a wrong frame pass 0.9 % (pixels read as points), 20.3 % (y-flip), 31.4 % (one line down) |
| Minimum confidence of a name (Dconf) | a candidate name whose every word has a Tesseract confidence below **20** is not reported; the recogniser's text is unchanged | 21 (A: word filter before recognition; B: rules on names; C: end to end) | precision +0.2 [+0.1, +0.3], recall -0.0 [-0.1, +0.0], persons +0.0 on the 600 documents; 14 spurious names and 3 true positives removed, none added; a word filter before recognition costs recall from T = 10 (-0.8 [-1.1, -0.4]) |
| Browser of the `e2e` image | Playwright 1.64.0's Chrome Headless Shell only (`playwright install --with-deps --only-shell chromium`) | 20.5 | the full Chromium is never launched by the headless tests; leaving it out saves 403 MB (`/ms-playwright` 681 -> 278 MB; image 7.41 -> 6.80 GB) |

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

**Note (18.6).** Since `add-deployment` the pipelines are handed to waiting calls in arrival order (formerly in no particular order). Re-measured with the request set of 15.2 at J 12 and 16 clients: 84.7 docs/min against 84.0 with pool 8 and 84.8 against 85.8 with pool 4 (within the run-to-run spread); with pool 4 the per-job NER time per page is more even (p95 2.5-2.8 s against 3.0-3.3 s, maximum 3.2-3.4 s against 3.7-4.7 s). The figures of this section are otherwise unaffected.

---

## 16. Language model choice (`add-deployment` tasks 5.1 and 5.2, design D5 and D6)

Scripts: `benchmarks/llm_eval/` (local, like every benchmark script; driver `scripts/run.sh`, see `benchmarks/README.md` step 9). Raw outputs (per-question answers, recorded model calls, server and Ollama logs, VRAM samples) are in `benchmarks/llm_eval/out/` (local, not committed).

### 16.1 Method

**Stack.** The `docintel-test` image (built 2026-10-28 13:41 UTC, image id `sha256:3ea3c136...`; its `app/` is byte-identical to the working tree apart from line endings) runs the real service with Uvicorn and its **default settings**, plus `LLM_BASE_URL` and `LLM_MODEL`. Qdrant v1.20.0 (the digest of the tests) keeps its data in an external volume, so the 600 documents were ingested once and every candidate used the same index. Ollama `ollama/ollama:0.40.12@sha256:20d1bbfb...` ran with one GPU (RTX 4070 Ti 12 GB, Docker Desktop WSL2), `OLLAMA_CONTEXT_LENGTH=8192`, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=1`, so one model at a time. Compose projects `llmeval*` (override `benchmarks/llm_eval/compose.llmeval.yml` of `docker-compose.test.yml`), torn down with `down -v --remove-orphans` after every run. The service calls a **pass-through recording proxy** in the harness container (`common.Proxy`). It forwards the exact request body to the model server and returns the reply unchanged, and it records the messages, the reply, `usage` and the upstream time, which the service does not expose. The proxy stores no headers. Refusal reasons, top-1 score and `llm_ms` come from the service's own `question completed` log line, matched by `X-Request-ID`.

**Index.** All 600 PDFs through `POST /api/ingest` with 12 client threads (default `MAX_CONCURRENT_JOBS` 12): 600 x 200, **3,299 chunks** (the same count as the chain re-measurement of section 8.1), 1,045 pages in 1,010 s = **35.6 docs/min, 62 pages/min**. The model download (about 21 GB) ran on the host at the same time, so this is not a clean throughput figure (task 6.1 measures throughput). 11 of the first 12 simultaneous requests on the empty store returned 503 `vector_store_unavailable` and succeeded when resent: concurrent first `QdrantVectorStore.ensure_collection` calls hit a short window in which the new collection exists but is not yet readable (finding 1 in 16.10, fixed).

**Sample** (`scripts/sample.py`, seed 20261028, `out/sample.json`). 600 of the 2,871 answerable same-language questions, stratified by language x document type (35 strata, proportional allocation by largest remainder, 14-19 per stratum). n = 600 bounds the 95 % half-width at 1.96 x sqrt(0.25/600) = 4.0 points (3.6 with the finite-population correction). Also 150 of the 300 unanswerable questions, 10 per language x category (absent entity, cross-document, off-topic). Every candidate and the hosted reference got the same 750 questions, in one fixed shuffled order (seed 20261029), sequentially (one client), after 3 warm-up questions that were not counted.

**Scores** (`scripts/metrics.py`, `score.py`), as in D6:
- **Accuracy.** The normalised gold answer is contained in the reply on token boundaries. Normalisation is NFKD, casefold, combining marks dropped, every Unicode punctuation character replaced by a space, whitespace collapsed. A refusal counts as wrong. Token-F1 (SQuAD-style, same normalisation) is the secondary score.
- **Accuracy when gold in context.** The same accuracy, restricted to the questions whose normalised gold answer is contained in the context the model received. This separates the model's reading from retrieval. It undercounts slightly, because OCR errors on the gold page make the gold string absent.
- **Refusals.** Wrong refusals on answerable questions and correct refusals on unanswerable ones, each split into threshold refusals (top-1 cosine below 0.808, model not called) and model refusals (`NOT_IN_DOCUMENTS` or a blank reply).
- **Answer language.** The service's stop-word detector (`detect_language`, `min_words=3`, since replies are short) compared with the question's language. Undetermined replies (too few stop words) are counted separately.
- **Latency.** The client's time for `POST /api/ask` (all requests, and the requests that called the model), plus the service's `llm_ms`.
- **Intervals.** Percentile bootstrap over questions, 2,000 resamples, seed 0, paired over the same questions for differences.
- **Tokens per second, and token counts checked against the API.** A side probe (`scripts/probe.py`) replays 100 recorded calls per candidate (40 for the 12B thinking runs) to Ollama's native `/api/chat` with the same messages, temperature 0 and `num_predict` 512, and reads `eval_count / eval_duration` (generation) and `prompt_eval_count / prompt_eval_duration` (prefill). `prompt_eval_count` equalled the OpenAI-compatible `usage.prompt_tokens` in 100 / 100 replays for every candidate. KV-prefix caching does not shrink the count.
- **VRAM.** `nvidia-smi` on the host sampled every second. The figures are the device total, including about 0.9 GB used by the desktop at idle.

**Thinking.** All three Gemma 4 models declare the `thinking` capability with default `true` (`ollama show`). Through Ollama's OpenAI-compatible endpoint, which the service's client uses, they therefore reason before answering. The reasoning comes back in a separate `reasoning` field, and its tokens count against `max_tokens` (512, `LLM_MAX_TOKENS`). Of the request fields tried (`scripts/think_check.py`), only `reasoning_effort: "none"` turns thinking off on this endpoint: the same German question took 22 completion tokens and 0.2-0.5 s, against 336-413 tokens and 3-12 s. `think: false` is ignored by the OpenAI-compatible endpoint (it works only on the native `/api/chat`), and `reasoning_effort: "low"` still thinks. The service's client sends neither field. Each candidate was therefore measured twice:
- **thinking off (`_nt`).** The proxy adds `reasoning_effort: "none"` (`--rewrite reasoning_none`). This is what the client would send with a new setting (16.10). Full sample for all three models.
- **as configured (`_t`).** The unchanged request, so thinking is on. The full sample for e4b. For the 12B models, a random subsample: the first 300 (12b-qat) and 150 (12b-q4_K_M) questions of the fixed shuffled order, because a 12B call takes about 8.5 s in this mode.

**Hosted reference** (approved by the user): `gpt-6-luna` through `https://api.openai.com/v1`, model id confirmed from every response's `model` field (660 / 660). The key was passed only through compose interpolation from the repository `.env` (`compose.hosted.yml`), never written or printed; the outputs were checked afterwards for key-like strings (none). Two parameters of the service's unchanged client are rejected by this model, so the run used side-harness rewrites (`--rewrite max_completion_tokens,drop_temperature`; finding 3 in 16.10):
- **`max_tokens`:** HTTP 400 `unsupported_parameter`, "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead." The proxy renamed it to `max_completion_tokens`.
- **`temperature` 0:** HTTP 400 `unsupported_value`, "Unsupported value: 'temperature' does not support 0.0 with this model. Only the default (1) value is supported." The proxy dropped the field, so the reference ran at the model's default temperature 1 and is not deterministic.

Both errors reached the API client as 503 `llm_unavailable`. An earlier attempt with the first key failed with 401 `expired_secret_key` (no tokens billed).

### 16.2 Candidates (task 5.1)

Pulled 2026-10-28 with `ollama pull` (`out/models.txt`: `ollama list`, `ollama show`, manifest files):

| tag | ollama list ID / manifest digest | size | parameters | quantisation | weights blob | other | trained context | thinking |
|---|---|---|---|---|---|---|---|---|
| `gemma4:e4b-it-qat` | `ee6656371218` / `sha256:ee665637121887cf3befff38abbb1be4ee117c7db867d97a67e29049ecd7e15f` | 6.1 GB | 7.5B | Q4_0 (QAT) | `sha256:e8b6a059ba86...` (5.15 GB) | vision/audio projector 0.99 GB | 131,072 | default on |
| `gemma4:12b-it-qat` | `38044be4f923` / `sha256:38044be4f923e5a55264ed7df4eaac2676651a905f735197c504045140c02bd3` | 7.2 GB | 11.9B | Q4_0 (QAT) | `sha256:faff1a63667f...` (6.98 GB) | projector 0.18 GB | 262,144 | default on |
| `gemma4:12b-it-q4_K_M` | `6114515d63c1` / `sha256:6114515d63c17436a7c0417d82820ac65ad643e2806c5a3c89cb62846436ed0b` | 8.0 GB | 11.9B | Q4_K_M (imatrix) | `sha256:bb722270d543...` (7.38 GB) | projector 0.18 GB, **draft model** 0.47 GB (`draft_num_predict 3`, speculative decoding) | 262,144 | default on |

All three ship sampling defaults of temperature 1, top_k 64, top_p 0.95; the service sends `temperature` 0. Loading a model into VRAM took 33-42 s (`llama-server started in` in the Ollama log) on the first request after the container started (the weights are read from the Docker volume). That first request took 65 s, longer than `LLM_READ_TIMEOUT_S` (60 s); the client timed out and its retry succeeded (warm-up lines in `console.log`, one `BrokenPipeError` in the proxy per run). Later requests in the same server found the model loaded (`OLLAMA_KEEP_ALIVE` 60 min here).

### 16.3 Context length (task 5.1)

**Observed** over 659 model calls per candidate (the full sample) plus the warm-up:
- **Largest prompt:** 2,661 tokens (e4b) and 2,665 tokens (both 12B models, which share a tokenizer), at 6,122 characters of messages.
- **Mean prompt:** 1,493 / 1,497 tokens.
- **Densest prompt:** 0.457 tokens per character, from invoice pages full of amounts and identifiers. German and Portuguese invoices are the densest.

**Constructed:** the probe packs the densest recorded pages with the service's own `select_context` (3 pages, budget 6,000) and the longest sample question (147 characters). This gave 1,973 tokens (e4b, 5,821 context characters) and 2,549 tokens (12B, 4,808 characters; the three densest pages were short).

**Upper bound:** 0.46 tokens per character x about 6,700 characters (6,000 of context, the 535-character system prompt, the framing and a 150-character question) = **about 3,100 prompt tokens**. Generation adds up to `max_tokens` 512, so a request needs **at most about 3,600 tokens of context**; 25 % margin makes about 4,500.

| setting | context | margin over 3,600 | verdict |
|---|---|---|---|
| Ollama's own default on this GPU (log: `vram-based default context ... default_num_ctx=4096`) | 4,096 | 14 % | too small for the 25 % rule |
| **`OLLAMA_CONTEXT_LENGTH=8192`** (measured) | 8,192 | 128 % | **recommended** |

VRAM at 8,192 (device maximum during each run, desktop included): e4b 5.7 GB, 12b-qat 9.2 GB, 12b-q4_K_M 10.2 GB.

**No truncation at 8,192.**
- No `truncating input prompt` warning in any Ollama log of the quality runs.
- `prompt_eval_count` equalled `usage.prompt_tokens` in every replay.
- `n_tokens` of the llama-server slot lines stayed under 3,100.
- The detector works: a deliberate request with `options.num_ctx` 1024 (12B) logged `level=WARN msg="truncating input prompt" limit=1027 prompt=2549 keep=5 new=1027` and reported `prompt_eval_count` 1,027. The llama-server line said `truncated = 0` for the same request, so truncation must be checked on Ollama's own WARN line, not on llama-server's counter.
- For e4b the per-request `num_ctx` 1024 did not reload the model and the full 1,973 tokens were evaluated. The service never sends `num_ctx`, so the server setting is what counts.

### 16.4 Answer quality (task 5.2)

Full sample (600 answerable, 150 unanswerable; `out/score/tables.md`). Thinking off for the local models; the hosted model with its own default reasoning (`completion_tokens_details.reasoning_tokens` 16,895 in total, 0 for most replies):

| model (thinking) | accuracy (containment) | token-F1 | accuracy, gold in context (n = 448) | wrong refusals: all / threshold / model | correct refusals: all / threshold / model | language mismatch (undetermined of answered) |
|---|---|---|---|---|---|---|
| e4b (off) | 72.3 (68.7-76.0) | 27.5 (25.4-29.7) | 95.5 (93.5-97.3) | 10.5 (8.2-13.0) / 3.5 / 7.0 (5.2-9.0) | 72.0 (64.7-78.7) / 46.7 / 25.3 (18.7-32.7) | 0.3 (0.0-1.0) (226 of 537) |
| 12b-qat (off) | **72.8 (69.2-76.5)** | 20.6 (19.3-21.9) | 95.5 (93.5-97.3) | 12.5 (10.0-15.0) / 3.5 / 9.0 (6.8-11.2) | 72.7 (66.0-79.3) / 46.7 / 26.0 (19.3-33.3) | 0.6 (0.0-1.6) (216 of 525) |
| 12b-q4_K_M (off) | 72.3 (68.7-76.0) | 21.1 (19.9-22.5) | 95.1 (92.9-96.9) | 12.3 (9.8-15.0) / 3.5 / 8.8 (6.7-11.0) | 72.7 (66.0-79.3) / 46.7 / 26.0 (19.3-33.3) | 0.0 (0.0-0.0) (219 of 526) |
| e4b (on, as configured) | 61.5 (57.7-65.5) | 32.3 (29.7-35.0) | 81.0 (77.5-84.6) | 24.7 (21.3-28.0) / 3.5 / 21.2 (18.0-24.3) | 74.0 (66.7-80.7) / 46.7 / 27.3 (20.7-34.7) | 0.6 (0.0-1.9) (296 of 452) |
| 12b-qat (on), subsample 241 / 59 | 61.8 (55.6-67.6) | 25.3 (22.4-28.3) | 78.3 (72.5-84.1) (n = 189) | 32.4 (26.6-38.2) / 3.7 / 28.6 (22.8-34.4) | 76.3 (64.4-86.4) / 47.5 / 28.8 | 0.0 (104 of 163) |
| 12b-q4_K_M (on), subsample 125 / 25 | 64.0 (55.2-72.0) | 28.0 (23.5-32.3) | 80.6 (71.4-87.8) (n = 98) | 28.8 (20.8-36.8) / 4.8 / 24.0 (16.8-32.0) | 80.0 (64.0-92.0) / 44.0 / 36.0 | 0.0 (60 of 89) |
| gpt-6-luna (hosted reference) | 69.5 (65.5-73.3) | **48.7 (45.7-51.7)** | 91.1 (88.4-93.8) | 12.0 (9.5-14.5) / 3.5 / 8.5 (6.3-10.7) | 71.3 (64.0-78.0) / 46.7 / 24.7 (18.0-32.0) | 0.0 (431 of 528) |

Every request returned 200 (0 errors in 4,950 questions). The threshold columns carry no interval: retrieval is the same for every model, so threshold refusals are identical.

**Accuracy by language**, thinking off (en / es / pt / fr / de; each about +-8 points). German is the lowest for every model.
- e4b: 72 / 70 / 77 / 76 / 66.
- 12b-qat: 73 / 71 / 79 / 76 / 66.
- 12b-q4_K_M: 73 / 71 / 78 / 76 / 64.
- gpt-6-luna: 70 / 68 / 74 / 75 / 60.

**Refused by category**, all models with thinking off: absent entity 100 %, off-topic 100 %, cross-document 14-18 %.

**Paired differences in accuracy** against the most accurate local model (12b-qat, thinking off), in points:
- e4b: -0.5 [-1.7, +0.7].
- 12b-q4_K_M: -0.5 [-1.2, +0.0].
- e4b with thinking on: -11.3 [-14.2, -8.8].
- gpt-6-luna: -3.3 [-5.0, -1.8].

**Paired differences in wrong refusals**, same reference:
- e4b: -2.0 [-3.3, -0.8].
- 12b-q4_K_M: -0.2 [-0.8, +0.3].
- e4b with thinking on: +12.2 [+9.5, +15.0].
- gpt-6-luna: -0.5 [-1.7, +0.7].

**Retrieval is the same for everyone.** The context is identical for all models (same index, same questions): the gold answer was in it for 448 of 600 questions (74.7 %). Threshold refusals are identical: 21 answerable (3.5 %) and 70 unanswerable (46.7 %) were refused before any model call. The 1-2 point spread between the models is far below the +-4-point interval of each.

**When the gold answer is in the context, the local models with thinking off answer it in 95 %.** Their model refusals of answerable questions are mostly questions whose gold answer was not in the context: 36 of e4b's 42, 44 of 54 (12b-qat) and 43 of 53 (12b-q4_K_M). That is the intended behaviour; only 1.3-2.2 % of the gold-in-context questions were refused.

**Where the accuracy is lost.** Most of the remaining loss is retrieval. The corpus has many near-identical documents (the same template, with the same project names), so when the gold document is not retrieved the models answer from a similar document's values, or list each retrieved document's value; both are scored wrong. Language mismatch is 0-0.6 % of the determinable replies for every model. Refusal of unanswerable questions is the same for all: 100 % of absent-entity and off-topic questions (two thirds of absent-entity refusals by the model, the rest by the threshold), 14-22 % of cross-document questions (most of them are answerable in a filter-free index, section 8).

**Thinking on (the service's current client).**
- **e4b, full sample.** Accuracy falls from 72.3 to 61.5, and wrong refusals rise from 10.5 to 24.7 % (model refusals 21.2 %). In **136 of 659 calls (21 %)** the reasoning used the whole 512-token budget and the call ended with `finish_reason: length` and an **empty reply**, which `is_refusal` turns into the fixed refusal. With the gold answer in the context, 14.3 % were refused (1.3 % with thinking off). Among the questions it did answer it is as accurate as without thinking (81.6 against 80.8 %).
- **Cost of thinking.** Completion tokens are 257,175 against 26,408, and the median latency of a model call is 4.05 s against 0.56 s.
- **12B subsamples:** on the same questions with thinking off:
  - **Accuracy:** 12b-qat falls by 14.1 [-18.7, -9.5] points (61.8 against 75.9), 12b-q4_K_M by 10.4 [-16.0, -4.8] (64.0 against 74.4).
  - **Wrong refusals:** rise by 19.1 [+14.1, +24.1] and 15.2 [+8.8, +21.6] points.
  - **Token limit:** 62 of 263 and 27 of 133 calls ended at the 512-token limit.
  - **Latency:** the median model call takes 7.94 s and 3.97 s, against 1.50 s and 1.42 s with thinking off on the same questions.
  - **Unanswerable questions:** thinking refuses slightly more of them correctly (e4b 111 against 108 of 150), not enough to offset the loss.

### 16.5 Latency, speed and memory

Sequential, one question at a time, on the full sample. 659 of the 750 questions call the model; the others are threshold refusals answered in about 55 ms.

| model (thinking) | requests p50 / p95 (s) | with a model call p50 / p95 / p99 / max (s) | generation tokens/s (p50) | prefill tokens/s (p50) | completion tokens (p50) | VRAM, device max (GB) | prompt / completion tokens, whole sample |
|---|---|---|---|---|---|---|---|
| e4b (off) | 0.52 / 1.74 | **0.56 / 1.79** / 2.26 / 6.57 | 113 | 7,183 | 27 | 5.7 | 983,283 / 26,408 |
| 12b-qat (off) | 1.44 / 4.09 | 1.51 / 4.33 / 4.89 / 7.63 | 55 | 3,228 | 35 | 9.2 | 985,919 / 33,970 |
| 12b-q4_K_M (off) | 1.37 / 2.20 | 1.42 / 2.24 / 2.82 / 6.43 | 126 | 2,196 | 29 | 10.2 | 985,919 / 28,996 |
| e4b (on) | 3.87 / 5.37 | 4.05 / 5.41 / 5.74 / 8.99 | 106 | 6,775 | 371 | 6.0 | 984,601 / 257,175 |
| 12b-qat (on), 300 | 7.61 / 11.12 | 7.94 / 11.12 / 11.55 / 14.83 | 53 | 3,135 | 308 | 9.7 | 387,349 / 99,099 |
| 12b-q4_K_M (on), 150 | 3.80 / 5.52 | 3.97 / 5.53 / 6.11 / 9.37 | 129 | 2,104 | 340 | 10.7 | 199,848 / 49,152 |
| gpt-6-luna | 1.19 / 2.62 | 1.25 / 2.72 / 3.66 / 4.40 | - | - | 18 (mean 44, reasoning included) | - | 862,017 / 29,076 (OpenAI tokenizer) |

Tokens per second come from the side probe (100 replays per model; 40 for the 12B thinking runs). VRAM is the device total sampled on the host, with the idle desktop taking about 0.9 GB. The models themselves therefore take about 4.8 / 8.3 / 9.3 GB at `OLLAMA_CONTEXT_LENGTH` 8192; the q4_K_M figure includes its draft model.

The 12b-q4_K_M generates faster than 12b-qat (126 against 55 tokens/s) because of its bundled draft model (speculative decoding); its prefill is slower (2,196 against 3,228 tokens/s). With thinking off, replies are short (median 27-35 completion tokens), so latency is mostly prefill plus about 30 generated tokens. Retrieval in the service adds a median 48-50 ms (`embed_ms`) plus 3 ms (`search_ms`). The hosted model's latency includes the internet round trip (1.0-1.7 s per call in the 5-question probe). Its run shared the host with the 12B thinking run on the GPU, so its VRAM column is not its own.

### 16.6 Manual audit of the containment metric

50 replies scored wrong by containment, answered (not refused), from the three local models with thinking off: 17 / 17 / 16, round-robin over the five languages, seeded (`scripts/audit.py`, labels in `out/score/audit_labels.json`). Each was checked against its question, the gold answer, the gold sentence and the context the model received, including where the gold page was in the context and how OCR rendered it:

| label | count | what it was |
|---|---|---|
| truly wrong | 40 | 34 answered from another, similar document (the gold document was not in the context, or only another page of it was); 4 picked the rank-1 document's value while the gold page was at rank 2-3; 1 gave the KYC reference instead of the document number; 1 the gold page was too garbled by OCR to read |
| correct, format (OCR copy) | 6 | the model copied the OCR text faithfully: `Salle CS` for C5 (twice), `ler fevrier` for 1er, `8hoo`, `10188862` for 1018886Z, `204380571` for 29438057H |
| correct paraphrase | 3 | `la oficina de Bogota, Sala 8A` for `nuestra oficina...`; the sender's company name in place of "our office" (twice) |
| partially correct | 1 | the right (OCR-garbled) author plus the author of a second memo |

No number or date in another format (e.g. 1.000 vs 1,000, 2024-12-10 vs 10 December 2024) appeared among the 50, so no extra normalisation was added for the local models. **Correction.** 9 of 50 (18 %) are correct, 3 of 50 (6 %) if OCR copies count as wrong for the user. Applying these pooled rates to each model's answered-but-not-contained replies (88-103) gives:

| model | accuracy (containment) | corrected, OCR copies count as correct | corrected, paraphrases only |
|---|---|---|---|
| e4b | 72.3 | 75.4 | 73.4 |
| 12b-qat | 72.8 | 75.5 | 73.7 |
| 12b-q4_K_M | 72.3 | 75.1 | 73.3 |

The containment metric therefore underestimates the local models by about 1-3 points, about equally for all three; the ranking does not change.

**Supplementary audit of the hosted model** (46 replies; `out/score/audit_labels_luna.json`). The hosted model answers tersely and rephrases more, which containment penalises: its token-F1 is the highest, 48.7, while its containment is the lowest. It was therefore audited separately. Of its 111 answered-but-not-contained replies, all 31 with the gold answer in the context were checked:
- 11 correct paraphrases ("Na Sala A5, em Campinas", "Im Buero in Berlin, Raum 12").
- 5 format variants: "19 %" for "19,0 %", "7%" for "7,0%", German case endings "30 Tagen" for "30 Tage", "24 Monaten", "12 Monaten".
- 2 partial.
- 13 truly wrong.

Of the other 80, 15 sampled were 13 wrong, 1 partial and 1 OCR copy. Corrected accuracy: 72.2 (paraphrases and format variants) to 73.1 (OCR copies as well), against 69.5 strict. These number and inflection variants occur only in the hosted model's replies. A metric that folded them would raise its score by about 1 point; it was not changed, to keep D6's definition.

### 16.7 Concurrency (D7 questions part, chosen candidate)

Chosen candidate (16.8): e4b. Thinking off (side-harness `reasoning_effort: "none"`), and also as configured (thinking on), so the effect of the current client is visible. Method (`scripts/chain_conc.sh`, `conc_table.py`):
- A fresh compose project per cell with `OLLAMA_NUM_PARALLEL` 1, 2 or 4, `MAX_CONCURRENT_QUESTIONS` 16 (default) and 16 client threads.
- The first 320 answerable questions of the fixed order with thinking off (96 with thinking on), after 4 sequential warm-up questions.
- `LLM_READ_TIMEOUT_S` 600 and `LLM_TOTAL_TIMEOUT_S` 900 for these runs only, so that the measured latency is not cut by the client.
- Thinking off: two passes, the second in reverse order. Thinking on: one pass.
- The machine was otherwise idle: no other compose project was running, and the other agent's `deploy` stack had stopped before 17:30 UTC.

| thinking | `OLLAMA_NUM_PARALLEL` | questions/min (passes) | client p50 / p95 / p99 / max (s) | `llm_ms` p50 / p95 / p99 (s) | VRAM, device max (GB) |
|---|---|---|---|---|---|
| off | 1 | 93.9, 94.2 | 10.0 / 15.0 / 17.1 / 17.7 | 10.1 / 15.2 / 17.0 | 5.8-5.9 |
| off | **2** | **124.6, 123.3** | 7.5 / 12.3-12.4 / 15.0 / 15.8 | 7.5 / 12.7 / 15.0 | 6.1 |
| off | 4 | 127.5, 128.3 | 6.9-7.1 / 11.7-13.0 / 13.6-13.9 / 20.7 | 6.9-7.2 / 11.6-13.0 / 12.9-13.9 | 6.1 |
| on | 1 | 16.3 | 60.7 / 65.3 / 66.3 / 66.3 | 61.2 / 65.3 / 66.2 | 6.1 |
| on | 2 | 26.7 | 36.2 / 40.0 / 41.3 / 41.5 | 36.3 / 39.9 / 41.2 | 6.0 |
| on | 4 | 36.3 | 24.9 / 32.0 / 34.7 / 35.1 | 24.9 / 32.0 / 33.9 | 6.2 |

Every request returned 200. `llm_ms` includes the wait in Ollama's queue, so with 16 questions in flight it is about 16 / `NUM_PARALLEL` times one call. With thinking off, a second slot adds 32 % throughput and a fourth only 3 % more (one GPU; batching two sequences is nearly free, four are limited by compute). Parallel slots cost only about 0.2-0.3 GB of VRAM at this context (`ollama ps` reports the same 8,192 context per slot). The one 20.7 s maximum at `NUM_PARALLEL` 4 is a single outlier of the second pass.

**Thinking on with the current defaults breaks under load.** With one slot and 16 questions in flight, the median model call takes 61 s and the p99 66 s. That is above the default `LLM_READ_TIMEOUT_S` of 60 s, so with the service's defaults about half of these questions would time out and be retried: the client makes 3 attempts within the 90 s `LLM_TOTAL_TIMEOUT_S` and adds load while it does.

**Recommendations** (the D7 rule of section 15.6, the smallest setting within 5 % of the best throughput):
- **`OLLAMA_NUM_PARALLEL=2`.** 123-125 questions/min against the best 128; NUM_PARALLEL 1 is 27 % below the best.
- **`MAX_CONCURRENT_QUESTIONS` 16, unchanged.** At 16 questions in flight the p99 of a model call is 15 s with two slots, a quarter of the read timeout.
- **`LLM_READ_TIMEOUT_S` 60, unchanged.** This is 4.0 times the loaded p99 (15.0 s at NUM_PARALLEL 2) and 3.4 times the loaded maximum at NUM_PARALLEL 1 (17.7 s). Two conditions apply.
  - **The model must already be in VRAM.** The first question after Ollama starts took 65 s: 33-42 s of model load plus the call. It succeeded only through the client's retry. Preloading the model when the stack starts, for example a one-token request from the pull step with `keep_alive`, and a long `OLLAMA_KEEP_ALIVE` remove this. Without a preload, use `LLM_READ_TIMEOUT_S=90` with `LLM_TOTAL_TIMEOUT_S` 180.
  - **Thinking must be off.** With thinking on and two slots, the p99 is 41 s (66 s with one slot). That would need `LLM_READ_TIMEOUT_S` of at least 90 s and fewer concurrent questions.

### 16.8 Choice (D5 rule)

D5: among the local models whose accuracy lies within the 95 % interval of the best local model, the one with the lowest p95 latency; a candidate clearly worse on wrong refusals or answer language is excluded.

1. **Best local model.** 12b-qat with thinking off, 72.8 (69.2-76.5).
2. **Inside its interval.** e4b at 72.3 and 12b-q4_K_M at 72.3, both with thinking off (paired differences -0.5 [-1.7, +0.7] and -0.5 [-1.2, +0.0]). Every thinking-on configuration lies outside it (61.5-64.0, paired -10 to -14 points).
3. **Exclusions.** None. Wrong refusals are 10.5 % for e4b, which is lower than 12.3-12.5 % for the 12B models (paired -2.0 [-3.3, -0.8]). Answer-language mismatch is at most 0.6 % for all three.
4. **Lowest p95 latency.** e4b: **1.79 s** for a request that calls the model. The 12B models take 2.24 s (q4_K_M) and 4.33 s (qat).
5. **e4b also leads on other counts.** It uses the least VRAM (about 4.8 GB, so it may fit an 8 GB card; not tested), it was the only candidate measured under load (16.7), and its audit-corrected accuracy equals the others' (75.4 against 75.1-75.5).

**Recommended: `LLM_MODEL=gemma4:e4b-it-qat`, with thinking off.** The service's client does not turn thinking off today. With the unchanged client, the rule still picks e4b, since all thinking-on runs are equally poor, but at 61.5 % accuracy, 24.7 % wrong refusals and 4 s per question. So the recommendation depends on the client change in 16.10 (finding 2).

**Ollama options:**
- **`OLLAMA_CONTEXT_LENGTH=8192`** (16.3).
- **`OLLAMA_NUM_PARALLEL=2`** (16.7).
- **`OLLAMA_MAX_LOADED_MODELS=1`.**
- **`OLLAMA_KEEP_ALIVE` long** (for example `24h`, or `-1` to keep the model loaded), plus a preload at start (16.7).

The hosted reference, gpt-6-luna, is not more accurate on this corpus: 69.5 strict and 72.2-73.1 after its audit, against 72.3-72.8 strict and 73.3-75.5 corrected for the local models. Its latency (p95 2.72 s) is higher than e4b's. The hosted option is therefore a deployment choice (no GPU), not a quality upgrade.

### 16.9 Hosted reference: tokens and cost

Usage fields of the 660 responses (659 questions plus 1 warm-up; the 5-question probe and the two failed probes are separate):

| | prompt tokens | of which cached | completion tokens | of which reasoning | cost |
|---|---|---|---|---|---|
| full sample (750 questions, 659 model calls) | 862,017 | 57,188 | 29,076 | 16,895 | **$0.0956** |
| 5-question probe | 7,084 | 0 | 104 | 36 | $0.0008 |

Prices are $0.10 per 1M input tokens, $0.01 per 1M cached input tokens and $0.50 per 1M output tokens. The cost is computed as (prompt - cached) x 0.10 + cached x 0.01 + completion x 0.50, per 1M tokens. The responses also report `cache_write_tokens` (708,244 in total); no separate price was given for them, so they are counted as ordinary input. The failed probes (401 and 400) report no usage.

**Cost per 1,000 questions:** $0.13 for this mix of questions (12 % are refused before any model call), or $0.145 per 1,000 model calls. The prompt is about 1,310 tokens per call with the OpenAI tokenizer, against about 1,495 with the Gemma tokenizer (0.87 times). The projection made from the probe was $0.10 for the sample, which matches.

### 16.10 Findings for the main session (fixed in tasks 5.3 and 5.4 of `add-deployment`)

1. **Concurrent first `QdrantVectorStore.ensure_collection` calls.** 12 simultaneous first ingestions into an empty store gave 11 x 503 `vector_store_unavailable`. Reproduced against Qdrant v1.20.0: the duplicate create's 409 was already handled, but for about 15-30 ms after a create is accepted `collection_exists` is true while `get_collection` answers HTTP 500 "0 of 0 read operations failed" (shards not yet readable), and the retrier does not retry a 500, so `_verify` failed. Fixed in this block (task 5.4): first binding serialised by a lock, and the `get_collection` calls retry only that 500 (up to 10 tries, 0.05-0.5 s backoff, within the call deadline). After the fix: 12 simultaneous first ingestions on the runtime image, 12 x 200, one collection.
2. **Thinking models through the OpenAI-compatible endpoint (llm-client delta).** Gemma 4 on Ollama thinks by default. In this mode it is 10-14 points less accurate, and 21 % of calls (e4b) end at `max_tokens` with an empty reply that the service turns into a refusal. It is also slower (median model call 7.2 times longer with e4b, 2.8-5.3 times with the 12B models; 16.5), and under load it exceeds the default read timeout. Proposed client change:
   - An optional setting, for example `LLM_REASONING_EFFORT` (unset, `none`, `low`, ...), sent as `reasoning_effort`. With Ollama, `none` turns thinking off; this was verified, while `think: false` is ignored on this endpoint.
   - Log `finish_reason: length` with an empty content as its own outcome instead of a silent refusal.
3. **Hosted models that reject the current request (llm-client delta).** `gpt-6-luna` answers 400 to `max_tokens` (`unsupported_parameter`, wants `max_completion_tokens`) and to `temperature` 0 (`unsupported_value`, only the default 1). Proposed change:
   - A setting for the token-limit field name (`max_tokens` or `max_completion_tokens`), or always sending `max_completion_tokens` to OpenAI.
   - A way to omit `temperature` (for example `LLM_TEMPERATURE` unset means not sent).
   - Both errors currently surface as 503 `llm_unavailable` with no hint. Logging the upstream error `code` (not the message, which can quote input) would make them diagnosable.
4. **Cold start exceeds `LLM_READ_TIMEOUT_S`.** The first question after Ollama starts took 65 s (33-42 s model load) and was saved only by the retry. Preload the model in the stack (D2's pull step) or document a longer first-call timeout.
5. **Default Ollama context is too small for the margin.** On this 12 GB GPU, Ollama picks 4,096 when `OLLAMA_CONTEXT_LENGTH` is unset, which leaves 14 % over the worst request. The stack must set 8192.
6. **Truncation is visible only in Ollama's own log.** The WARN line `truncating input prompt` is the signal; llama-server's `truncated = 0` and the API response give no sign of it. `prompt_eval_count` and `usage.prompt_tokens` show the truncated count.

**Caveats.**
- **Corpus.** One synthetic corpus whose near-duplicate templates make retrieval, not the model, the main source of error (section 1). Real documents may separate the models more.
- **Determinism.** Temperature 0 on the local models is not fully deterministic: replaying the same messages to the native API gave the same reply in 72-81 % of cases.
- **Hosted reference.** It ran at temperature 1 with renamed parameters, through the side harness only, so it does not measure the service's current client. It shared the host with a GPU run.
- **12B thinking runs.** These are random subsamples of 300 and 150 questions, so their intervals are wider.
- **VRAM.** VRAM is the device total, including the desktop.
- **Audits.** The audits were done by one person (this evaluation) without a second rater.

---

## 17. Deployed stack (`add-deployment` task 6.1, design D7 and evidence plan items 3 and 4)

Scripts: `benchmarks/deploy_perf/` (local, like every benchmark script; driver `scripts/run.sh`, see `benchmarks/README.md` step 10). Raw outputs (client results per request, cgroup counters, `docker stats` samples, app, Qdrant and Ollama logs, `nvidia-smi` samples) are in `benchmarks/deploy_perf/out/` (local, not committed).

### 17.1 Method and environment

**Stack.** The repository's `docker-compose.yml` and `docker-compose.smoke.yml`, unchanged except `build.context` (rewritten to the repository path; `out/stack/compose.diff` shows that one line), started from a copy in `out/stack/` with its own `.env`. Compose reads `.env` from the project directory, both for interpolation and as the app's `env_file`, so the repository-root `.env` (which can hold a key) took no part in any run; the settings of a run were written into that copy's `.env`, exactly as a user configures the stack, and every run's `.env` is kept next to its results. The app's server process had no `LLM_API_KEY` (checked in `/proc/<pid>/environ`, 17.5), and no hosted model was called. Images built from the tree of this change with `docker build --target runtime -t docintel .` (`sha256:ffda6d7a...`) and `docker build --target test -t docintel-test .` (`sha256:018edb07...`), both on 2026-10-28. Compose projects `perf_cold`, `perf_par`, `perf_tp` and `perf_gpu`, each torn down with `down -v`; the evaluation volumes of section 16 were only mounted read-only (to copy them) and are unchanged.

**Host.** As in sections 13-16: Windows 11, Docker Desktop 29.9.2 (Compose 5.6.1, WSL2 VM with 16 CPUs and 15.2 GiB), AMD Ryzen 7 7700X (8 cores, 16 threads), RTX 4070 Ti 12 GB; nothing else ran during the timing runs.

**Client.** A separate container (`compose.perf.yml`: the test image on the stack's internal `backend` network, `http://app:8000`), so the client never shares the app container's CPU quota (in sections 13-15 it ran inside the server's container). Same request sets and client as block 5 (`client.py`): threads with one `httpx.Client` each pulling from a shared queue; `/api/extract` with the 25-document set of 13.2 three times (75 requests, 135 pages) and its name lists, no language hint; `/api/ingest` with 75 distinct documents (the 60 of the seed-0 sample and, per language, the first three corpus documents outside it; 125 pages). Each run is a fresh stack (`down -v`, so every ingestion is `indexed`), warmed up as in 15.1 (2 sequential extractions, 8 concurrent ones, 2 ingestions of other documents), then the extract cell and the ingest cell at 16 clients. `BUSY_WAIT_S=300` (nothing rejected, as in block 5); every other setting at its default unless stated. Two rounds per configuration, the second in reverse order (`chain_tp.sh`); the 4-CPU configurations one round.

**Counters.** docs/min = requests / wall time; latency is the client's. Mean CPUs used = the app container's cgroup `cpu.stat` `usage_usec` over the cell divided by its wall time (server and Tesseract processes); throttled = share of CFS periods throttled; server `VmHWM` reset before each cell; OCR, NER and embedding ms per page summed from the use cases' outcome lines of the cell's requests (request ids `<cell>-<i>`), including any wait for a pipeline or the embedder. CPU limit: `cpus: N` on the app service (`compose.cpus.yml`), a cgroup quota; the container still sees 16 CPUs, so `OCR_WORKERS` stays min(4, 16) = 4.

### 17.2 Image sizes (`sizes.txt`)

| image | size (`docker image inspect`) | venv | of which bytecode | models | `/app` | dev tools |
|---|---|---|---|---|---|---|
| `docintel` (runtime) | 5,840,400,350 B (5.84 GB) | 1,246 MB | 228 MB (10,085 `.pyc`) | 2,146 MB | 1 MB | none |
| `docintel-test` (test) | 5,749,571,612 B (5.75 GB) | 1,171 MB | 0 | 2,146 MB | 3 MB | pytest, pymupdf, mypy, ruff |

The runtime image is 91 MB **larger** than the test image: the development dependencies cost about 150 MB of the test venv (1,171 MB against 1,018 MB for the runtime venv without its bytecode), and the runtime venv is compiled to bytecode (228 MB), which D2 adopted because the non-root user cannot write `.pyc` files and imports then take 4.2-4.9 s instead of 8.6-9.5 s. Models (2.1 GB) and torch dominate both images.

### 17.3 Extract and ingest throughput, with and without a CPU limit

Means of the rounds (per-round values in `out/tables_tp.md`); J = `MAX_CONCURRENT_JOBS`, P = `NER_POOL_SIZE`, `TORCH_THREADS` 2; every request of every cell returned 200:

| app CPUs | J / P | rounds | extract docs/min (rounds) | extract pages/min | extract p50 / p95 (s) | CPUs used | throttled | OCR / NER ms per page | ingest docs/min (rounds) | ingest pages/min | ingest p50 / p95 (s) | server peak (MB) | warning |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 (unlimited) | **12 / 8** (defaults) | 2 | 85.1, 85.6 = **85.3** | 153.6 | 10.5 / 13.4 | 12.5 | 0 % | 2,633-2,681 / 1,737-1,761 | 50.9, 50.8 = **50.9** | 84.8 | 18.0 / 20.4 | 2,466-2,735 | no |
| 16, test image (control) | 12 / 8 | 2 | 84.6, 86.1 = 85.4 | 153.6 | 10.3 / 14.2 | 12.6 | 0 % | 2,651-2,691 / 1,727-1,790 | 51.0, 50.8 = 50.9 | 84.8 | 18.1 / 20.5 | 2,503-2,748 | no |
| 8 | 12 / 8 (defaults) | 2 | 53.6, 54.4 = 54.0 | 97.2 | 16.6 / 22.3 | 7.8 | 92-94 % | 4,142-4,226 / 2,876-2,931 | 49.9, 49.7 = 49.8 | 83.0 | 18.2 / 20.6 | 2,484-2,758 | **yes** |
| 8 | 8 / 4 | 2 | 59.1, 58.8 = 59.0 | 106.1 | 15.4 / 18.8 | 7.6 | 82-85 % | 2,817-2,821 / 1,504-1,520 | 49.9, 50.0 = 50.0 | 83.3 | 18.2 / 20.9 | 2,310-2,621 | no |
| 8 | **6 / 4** | 2 | 61.2, 61.5 = **61.4** | 110.5 | 14.9 / 17.1 | 7.3 | 55-56 % | 2,112-2,113 / 1,027-1,036 | 50.0, 50.5 = **50.3** | 83.8 | 18.4 / 20.8 | 2,243-2,566 | no |
| 8 | 4 / 4 | 2 | 51.3, 49.8 = 50.6 | 91.0 | 18.3 / 20.3 | 5.5 | 0-1 % | 1,769-1,820 / 771-793 | 47.8, 46.8 = 47.3 | 78.8 | 19.3 / 22.4 | 2,171-2,564 | no |
| 4 | 12 / 8 (defaults) | 1 | 27.9 | 50.3 | 32.2 / 47.0 | 4.0 | 98 % | 8,258 / 5,617 | 39.2 | 65.3 | 22.9 / 28.4 | 2,561-2,825 | **yes** |
| 4 | 4 / 2 | 1 | 36.2 | 65.1 | 25.7 / 28.5 | 3.9 | 92 % | 2,443 / 1,170 | 40.0 | 66.6 | 23.2 / 26.6 | 2,171-2,504 | no |
| 4 | **3 / 2** | 1 | **37.5** | 67.5 | 24.8 / 27.1 | 3.8 | 61 % | 1,829 / 783 | 37.6 | 62.7 | 24.6 / 27.9 | 2,161-2,495 | no |
| 4 | 2 / 2 | 1 | 29.3 | 52.7 | 32.4 / 34.8 | 2.9 | 4 % | 1,633 / 635 | 29.3 | 48.8 | 31.6 / 37.0 | 2,092-2,404 | no |

"Server peak" is the server's `VmHWM` over the extract and ingest cells; the container's sampled memory (`docker stats`, page cache excluded) stayed between 1.3 and 2.5 GB.

**Unlimited, against block 5.** The deployed runtime image reaches 85.3 docs/min (153.6 pages/min) for extraction at the defaults, against 91.2 docs/min (91.1-92.1 with `MALLOC_ARENA_MAX=2`) for the same configuration and request set in 15.2. The test image measured the same way, in the same session, gives 85.4: the 6-7 % gap is the method (client in its own container over the compose network instead of inside the server's container, another session), not the image. Latency (p50 10.5 s, p95 13.4 s against 9.8-10.0 / 12.4-13.0 s), CPUs used (12.5 of 16, 78 %, against 79 %), per-page stage times (OCR 2.6-2.7 s, NER 1.7-1.8 s against 2.5 / 1.6-1.7 s) and the server peak (2.5 GB against 2.2-2.3 GB) match 15.2.

**Ingestion is limited by the embedder, not by the CPUs.** Ingestion runs at 50.9 docs/min (84.8 pages/min) using only 4.9 of 16 CPUs, and the same 47-50 docs/min with 8 CPUs. Its embedding time is 5.8-5.9 s per page at J 12 against 0.7 s per page when 2 jobs share 4 CPUs: the e5 embedder is one model under one lock with 2 torch threads (`app/adapters/embeddings/embedder.py`), so concurrent ingestions queue for it, as NER did before its pool (13.2). 125 pages at about 0.7 s each is about 87 s, the 88 s wall time of the cell. This was not measured in block 5 (sections 13-15 timed extraction only; 16.1 indexed 600 documents at 35.6 docs/min while a model download ran). It is not a setting: more ingestion throughput per process needs a pool of embedder instances (or more torch threads per call, untested), a code change outside this task (17.9).

**With `cpus: 8`.** The oversubscription warning is logged once at start with the defaults (`NER_POOL_SIZE x TORCH_THREADS exceeds the CPUs available to the process`, `cpus_requested` 16, `cpus_available` 8; also at 4 CPUs with `cpus_available` 4), and not with pool 4 at 8 CPUs or pool 2 at 4 CPUs: the cgroup quota detection works in the deployed container. The defaults do hurt there: extraction falls to 54.0 docs/min with the quota throttled in 92-94 % of the periods and OCR and NER both about 1.6 times slower per page, while 6 / 4 gives 61.4 (+14 %) and 8 / 4 gives 59.0. At 4 CPUs the defaults give 27.9 against 37.5 for 3 / 2 (+34 %, one round). Too few jobs leave CPUs idle (4 / 4 at 8 CPUs uses 5.5 CPUs and reaches 50.6; 2 / 2 at 4 CPUs uses 2.9 and reaches 29.3). Ingestion barely moves with the settings (47-50 docs/min at 8 CPUs, 29-40 at 4), because the embedder queue sets it.

### 17.4 Sizing (D7 rule: the smallest setting within 5 % of the best extraction throughput)

| CPUs available to the app (quota or host) | `MAX_CONCURRENT_JOBS` | `NER_POOL_SIZE` (`TORCH_THREADS` 2) | extract docs/min (pages/min) | ingest docs/min | evidence |
|---|---|---|---|---|---|
| 16 | 12 | 8 | 85.3 (153.6) here; 88-92 in 15.6 | 50.9 | 15.6 (rule applied on this host), 17.3 (confirmed on the deployed image) |
| 8 | 6 | 4 | 61.4 (110.5) | 50.3 | 17.3: best 61.4, band from 58.3; 8 / 4 (59.0) is also inside, 4 / 4 (50.6) and the defaults (54.0, -12 %) are outside |
| 4 | 3 | 2 | 37.5 (67.5) | 37.6 | 17.3, one round: band from 35.6; 4 / 2 (36.2) is also inside, 2 / 2 (29.3) and the defaults (27.9, -26 %) are outside |
| other | about 3/4 of the CPUs (at least 2) | CPUs / `TORCH_THREADS` | not measured | not measured | the three measured points follow this rule; it is a rule of thumb between and beyond them |

The defaults (12 / 8) remain right for the 16-CPU host they were measured on and are not changed; on a smaller host or under a CPU limit they cost 12 % (8 CPUs) to 26 % (4 CPUs) of extraction throughput, and the start-up warning flags exactly that case. Pool = CPUs / `TORCH_THREADS` is the largest pool without the warning. Memory at the smaller settings is lower (server peak 2.2-2.6 GB against 2.5-2.8 GB).

### 17.5 Questions on the deployed stack (`gpu` profile)

**Stack.** `docker compose --profile gpu up -d --wait` of the copy (project `perf_gpu`), with the project's own volumes filled from read-only mounts of section 16's: the Qdrant index of the 600 documents (3,299 chunks, collection `chunks_eb486d86e2d5`) and, from the models volume, only `gemma4:e4b-it-qat`'s manifest, blobs and metadata (5.9 GB). Defaults throughout (`LLM_READ_TIMEOUT_S` 60, `LLM_TOTAL_TIMEOUT_S` 90, `MAX_CONCURRENT_QUESTIONS` 16), no harness rewrite and no proxy: the service's own client talks to Ollama. Checked in the running stack (`gpu_env.txt`): the app's server process has `LLM_BASE_URL=http://ollama:11434/v1`, `LLM_MODEL=gemma4:e4b-it-qat`, `LLM_REASONING_EFFORT=none` (set by the compose start command) and no `LLM_API_KEY`; Ollama 0.40.12 has `OLLAMA_NUM_PARALLEL=2`, `OLLAMA_KEEP_ALIVE=-1`, `OLLAMA_CONTEXT_LENGTH=8192`, `OLLAMA_MAX_LOADED_MODELS=1`, `CUDA_CACHE_PATH=/root/.ollama/cuda-cache`; `ollama ps` shows the model 100 % on the GPU, context 8192, until "Forever". Questions: the first N answerable questions of section 16's fixed order (`benchmarks/llm_eval/out/sample.json`, seed 20261029), after 3 warm-up questions; two rounds.

| clients | questions per round | questions/min (rounds) | client p50 / p95 / p99 / max (s) | `llm_ms` p50 / p95 / p99 (s) | statuses | refused: threshold / model | VRAM, device max (GB) |
|---|---|---|---|---|---|---|---|
| 1 | 150 | 81.1, 100.0 | 0.54-0.62 / 1.29-1.50 / 1.77-2.38 / 1.90-7.17 | 0.49-0.56 / 1.27-1.44 / 1.72-2.34 | 300 x 200 | 6 / 10 per round | 5.9-6.5 |
| 16 | 320 | **127.8, 136.3** | 7.0-7.5 / **9.6-9.7** / 10.7-11.5 / 11.9-12.2 | 7.0-7.5 / 9.6 / 10.7-11.5 | 640 x 200 | 13 / 22-23 per round | 5.9-6.4 |

This confirms 16.5 and 16.7 on the deployed stack: one client gets p50 0.54-0.62 s and p95 1.29-1.50 s (16.5: 0.56 / 1.79 s for questions that call the model); 16 clients get 128-136 questions/min with p95 9.6-9.7 s (16.7 at `OLLAMA_NUM_PARALLEL` 2: 123-125 questions/min, p95 12.3-12.4 s, through the proxy and with raised timeouts). The first one-client round had one 7.2 s outlier and a slower start (81 against 100 questions/min); the second round's maximum was 1.9 s. VRAM is the device total including about 1.2 GB used by the desktop at idle. The loaded maximum (12.2 s) is a fifth of `LLM_READ_TIMEOUT_S`, so **`MAX_CONCURRENT_QUESTIONS` 16 and `LLM_READ_TIMEOUT_S` 60 stay** (D9); `OLLAMA_NUM_PARALLEL` 1 and 4 were not re-measured on the deployed stack (16.7 has them).

### 17.6 Cold start, end to end

Time of `docker compose [--profile gpu] up -d --wait` until it returns (every service healthy, the one-shot pull completed), with the container timestamps from `docker inspect`:

| case | runs | `up -d --wait` (s) | breakdown | first question (s) |
|---|---|---|---|---|
| (a) no profile, new containers and new Qdrant volume | 1 | 20.1 | Qdrant start to app start 5.7 s; app start to healthy 13.4 s | - |
| (a) no profile, from a stopped stack (`stop`, then `up`) | 3 | 17.4-17.9 | Qdrant start to app start 5.7 s; app start to healthy 11.2 s | - |
| (a) no profile, recreated containers (`down` without `-v`) | 2 | 17.6, 17.6 | as above | - |
| (a) no profile, fresh stack in each throughput run (17.3, runtime image) | 14 | 17.6-18.3 (median 17.6) | lifespan (`service started` `duration_ms`) 4.1-4.2 s | - |
| (b) `gpu`, fresh containers, model and compiled CUDA kernels in the volume (`down`, then `up`) | 3 | 19.7, 20.5, 23.4 | Ollama start to healthy 5.4-5.8 s; `ollama-pull` 3.5-6.6 s (pull check 0.74-0.77 s, warm-up prompt 2.6-5.6 s); app start to healthy 9.0-9.1 s | **0.92, 0.99, 1.07** (model call 0.75-0.94 s) |
| (c) `gpu`, first-ever start of the project (model just copied into the new volume, no CUDA kernel cache) | 1 | **85.0** | `ollama-pull` 68.6 s: pull check 1.0 s, warm-up prompt 60 s (model load 36 s, then 31 s to the first token); app start to healthy 9.0 s | 1.42 (model call 1.22 s) |

The first question after a start is answered in about 1 s in every case, far below `LLM_READ_TIMEOUT_S` (60 s) and `LLM_TOTAL_TIMEOUT_S` (90 s): the warm-up in `ollama-pull` and `OLLAMA_KEEP_ALIVE=-1` take the model load (and, on a first start, the CUDA kernel compilation) out of the request path, which in section 16.7 made a first question take 65 s. On a first-ever start the stack is healthy only after about 85 s (D2 measured 33-50 s load plus 33-46 s compilation on a 1B model; here 36 + 31 s), and a model download, when the volume is empty, comes on top (not measured here: about 6 GB). A cold page cache could not be forced: Docker Desktop's VM page cache cannot be evicted from the host without a privileged container writing `drop_caches`, which was not done; so "first-ever" here means new containers and volumes and no CUDA kernel cache, with the model blobs probably still in the page cache from the copy.

**Where the 17.6 s without a model go.** Qdrant listens 0.16 s after its container starts (its own log), but its healthcheck (`interval: 5s`, no `start_interval`) first passes 5.0-5.2 s after start, so the app starts 5.7 s after Qdrant in every run. The app then needs about 10.3 s to `Application startup complete` (interpreter start and imports about 6 s, lifespan 4.1-4.2 s; 9.0-9.1 s in total in the `gpu` runs) and is marked healthy at the next 2 s probe (`start_interval` 2 s of the image's healthcheck). The Ollama healthcheck has the same 5 s interval (5.4-5.8 s from start to healthy). A `start_interval` of about 1 s on the Qdrant and Ollama healthchecks would most likely take about 4 s off each of these starts (not measured: the compose file was not changed, 17.9).

### 17.7 Parity of the deployed runtime image with the test image (evidence plan item 4)

The 60 documents of the seed-0 sample through the deployed stack twice, once with the runtime image and once with the test image as the app service (`APP_IMAGE=docintel-test`, the same compose copy and the same client, `parity.py`): `/api/extract` with the documents' name lists, then `/api/ingest` into the empty store, 4 clients:

| check | identical / total | detail |
|---|---|---|
| extract: whole JSON body (names, boxes, matches, their order) | **60 / 60** | 805 names, each with its box, 86 matches; every status 200 |
| ingest: HTTP status, result status, `chunks_stored` | **60 / 60** | 327 chunks; every result `indexed` |
| against block 5's run (`p81.json` of 13.1, test image of `add-http-api`): names, matches and chunks per document | **60 / 60** each | same totals: 805 names, 86 matches, 327 chunks |

No difference, so no document is reported. Block 5's raw outputs keep only per-document counts and equality flags, not the bodies, which is why the exact comparison was regenerated with the current test image. Wall times: extraction 66.1 s (runtime) and 65.8 s (test), ingestion 78.5 and 78.3 s.

### 17.8 Smoke suite and hygiene against the `gpu` stack

`docker compose -f docker-compose.yml -f docker-compose.smoke.yml --profile gpu run --rm smoke` from the copy (`gpu_smoke.txt`): **12 passed, 2 skipped** (the two "without a model" variants, skipped because readiness reports a configured model), exit 0. The answerable smoke questions were answered by e4b (two `question completed` lines with `refused: false`, model calls of 0.53 and 0.28 s); the off-topic question was refused below the threshold. `... run --rm hygiene`: all 13 checks `ok` (uid 10001, tini as PID 1, no pytest, pymupdf or development tools, no tests, sample PDFs or `.env` in the image, Hugging Face offline, `MALLOC_ARENA_MAX=2`, read-only code and models, the service imports), exit 0. The smoke suite ingested the sample PDFs into the project's copy of the index after the question runs.

### 17.9 Findings and caveats

**Defaults.** No default is changed. Confirmed on the deployed stack: `MAX_CONCURRENT_JOBS` 12 / `NER_POOL_SIZE` 8 / `TORCH_THREADS` 2 on 16 CPUs (17.3), `MAX_CONCURRENT_QUESTIONS` 16, `LLM_READ_TIMEOUT_S` 60, `LLM_TOTAL_TIMEOUT_S` 90, `OLLAMA_NUM_PARALLEL` 2, `OLLAMA_KEEP_ALIVE` -1, `OLLAMA_CONTEXT_LENGTH` 8192 and the `ollama-pull` warm-up (17.5, 17.6). For smaller hosts, the sizing table of 17.4 instead of a new default.

**Findings for the main session** (not fixed here; this task changes no code or configuration):
1. **Ingestion throughput is capped by the single e5 embedder** (one instance under a lock): about 50 docs/min (85 pages/min) per process on 16 CPUs with 4.9 CPUs busy, unchanged at 8 CPUs. A pool of embedders (as for NER, section 14) or measured `TORCH_THREADS` above 2 would be the levers; a candidate for `KNOWN_LIMITATIONS.md` if not addressed.
2. **Healthcheck intervals cost about 5 s per dependency at start** (Qdrant and Ollama: `interval: 5s` without `start_interval`): about a third of the 17.6 s start without a model. A `start_interval` (Docker Engine 25 or later) is the likely fix; not measured.
3. **The job defaults lose 12-26 % of extraction throughput under a CPU limit of 8 or 4** (17.3); the start-up warning appears in exactly those cases. Documenting the sizing table (task 7.1) is the measured remedy; deriving the defaults from the available CPUs would be a code change and was not measured beyond the three points.
4. **The runtime image is not smaller than the test image** (5.84 against 5.75 GB, 17.2), because of its bytecode; worth stating in the README rather than claiming a slimmer image.

**Caveats.** One host, one session per measurement; the client in its own container makes the absolute figures 6-7 % lower than block 5's (shown by the test-image control), so compare within this section. The 4-CPU configurations ran once. A `cpus:` quota is not a smaller machine: the container still sees 16 CPUs, so `OCR_WORKERS` stayed 4 and torch and Tesseract threads were scheduled on all 16 under the quota. Short documents only (the 50-page workload of 15.3 was not repeated under a limit). The question runs use only answerable questions of section 16's order with the default `OLLAMA_NUM_PARALLEL` 2, and the 1-client figures are sensitive to a single slow call (7.2 s in one round). The first-ever start ran with the model blobs probably in the page cache, and with no download (17.6).

---

## 18. Embedder pool and concurrency derived from the CPUs (`add-deployment` task 6.2, design D10)

Scripts: `benchmarks/embed_pool/scripts/` (local, like every benchmark script; driver `run.sh`, timing runs in `chain.sh`, see `benchmarks/README.md` step 11).

**Root cause (17.3, 17.9 finding 1).** `E5Embedder` ran every call (passage batches and question queries alike) under one `threading.Lock` around both the tokenizer and the forward pass, so one process embedded one batch at a time with `TORCH_THREADS` (2) CPUs: 5.3 chunks/s however many jobs waited (18.2), and an ingestion job held its job slot mostly waiting for that lock (5.8-7.1 s of `embed_ms` per page at 12 jobs against 0.7 s alone). The lock was wider than needed: the model only runs forward in `eval` mode under `torch.inference_mode`, which torch allows from several threads, and the only shared mutable state is the fast tokenizer's truncation, padding and special-token settings, which `transformers` (5.21) sets in place on a call only when they differ from the requested ones (`set_truncation_and_padding`), and which are the same on every call of the embedder. **Design (as built):** `EMBEDDING_POOL_SIZE` passage batches run at once with the one model and the one tokenizer; tokenising runs under one lock (0.26 % of a call, 18.1), the forward pass and pooling outside it; one warm-up call at load settles the tokenizer settings, which are then only read; one extra slot is reserved for queries, and a waiting query takes the next free slot before any waiting passage (18.3); `close()` releases the model and fails waiting and later calls with `embedding_failed`. Errors are mapped as before (fixed messages, cause kept).

**Environment.** As in sections 13-17 (16 CPUs, AMD Ryzen 7 7700X, Docker Desktop WSL2, nothing else running during the timing runs). Image `docintel-test` built with `docker build --target test -t docintel-test .` from the tree of this task (torch 2.14.1+cpu, transformers 5.21.0, tokenizers 0.23.3); the confirmation runs of 18.4 with the image rebuilt after the reserved slot was added (`sha256:bd880278...`). Every script ran through `docker-compose.test.yml` (project `embpoolev`, with its Qdrant, `down -v` after each script); CPU limits by a `cpus:` quota on the measuring container (`compose.cpus.yml`; `cpu.max` read `800000 100000` and `400000 100000`, and the server's `concurrency in use` line reported `cpus_available` 8 and 4). Inputs: the 60 documents of the seed-0 sample (section 2.8) read once with the production OCR engine and chunked exactly as `IngestDocument` does (`chunks.py`): **327 chunks** (102 pages, mean 617 characters, at most 9 chunks per document), one `embed_passages` call per document as the use case makes them (batches of at most 16); questions from the corpus QA set.

### 18.1 Identical vectors (`correct.py`)

**Method.** Expected vectors: a pool of 1 called sequentially on the 60 passage calls and 200 questions (260 calls). Then a pool of 4 called from 16 threads on the same calls, three rounds in different shuffles; every vector compared component by component.

| check | result |
|---|---|
| vectors, pool 4 under 16 threads vs pool 1 sequential | **780 / 780 calls bit-identical** (3 rounds of 260), maximum absolute difference **0.0**; 0 exceptions |
| structure | one encoder, one model (eval mode), one fast tokenizer; after the warm-up the tokenizer's truncation is `{max_length 512, longest_first, right}` and padding is set |
| share of a call spent tokenising (under the lock) | **0.26 %** (40 passage batches: tokenising against forward pass) |
| wall time | pool 1 sequential 70.7 s; pool 4 at 16 threads 25.5-27.2 s (2.6-2.8 x) |
| after `close()` | `ProcessingFailed('embedding_failed', 'embedding model is closed')` |
| negative control: 16 threads on one fresh tokenizer, no lock, no warm-up | 320 / 320 calls without error with constant settings, and also with `max_length` alternating 512 / 256 between calls |

The tolerance stated by the spec and the tests (1e-6 per component) is therefore not even used on this host: concurrency does not change a single bit, because each call computes the same shapes with the same kernels. As in 14.1, the tokenizer failure the lock guards against (a concurrent in-place change of its settings, `RuntimeError: Already borrowed` in the Rust binding) was not reproduced; the lock rests on the code path, costs 0.26 % of a call and was kept for consistency with the NER pool.

### 18.2 Embedding alone: pool size x concurrency (`bench_one.py`, `bench_all.py`)

**Method.** A fresh process per configuration (pool P, `TORCH_THREADS` 2, P concurrent warm-up calls), then the 60 passage calls pulled from a shared queue by 1 and by 16 threads; two passes (pass 2 in reverse order of pools and concurrencies). "docs/min" is the embedding rate alone (the 60 documents' chunks per minute), not comparable with the end-to-end figures of 18.4. Call latency includes the wait for a slot.

| pool | concurrent calls | chunks/s (passes) | mean | docs/min | call p50 / p95 (s) | process CPUs | errors |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 5.1, 5.3 | 5.2 | 57.4 | 1.03 / 1.83 | 2.00 | 0 |
| 1 | 16 | 5.3, 5.3 | 5.3 | 58.8 | 1.26 / 58.20 | 2.00 | 0 |
| 2 | 16 | 9.5, 9.3 | 9.4 | 103.6 | 1.39 / 32.96 | 3.88 | 0 |
| 4 | 16 | 14.8, 14.8 | 14.8 | 163.1 | 1.67 / 21.19 | 7.49 | 0 |
| 6 | 16 | 17.9, 17.7 | 17.8 | 196.1 | 1.99 / 17.56 | 11.34 | 0 |
| 8 | 16 | 18.9, 18.3 | 18.6 | 205.0 | 2.55 / 16.70 | 14.83 | 0 |

At one call at a time every pool gives 5.2-5.5 chunks/s (call p50 0.97-1.03 s for a document's batch). With 16 callers the rate scales with the CPUs the pool can use (P x 2) up to pool 6 (11.3 CPUs, 3.4 x the former embedder) and adds 4 % more at pool 8 with all 16 CPUs busy; each batch gets slower as more run (p50 1.26 s at pool 1, 2.55 s at pool 8). With pool 1, the p95 of 58 s is a call waiting behind the other 15 (and some calls waited far longer than others: the slot was not handed out in arrival order, see 18.3; fixed by the FIFO hand-off of 18.6).

| pool | RSS after import (MB) | after load | after warm-up | peak (`VmHWM`) over the cells, pass 1 / pass 2 |
|---|---|---|---|---|
| 1 | 229 | 1,004-1,005 | 1,032-1,054 | 1,468 / 1,485 |
| 2 | 229 | 1,005 | 1,049-1,064 | 1,550 / 1,610 |
| 4 | 229 | 1,005 | 1,146-1,174 | 1,777 / 1,664 |
| 6 | 229 | 1,005-1,006 | 1,193-1,234 | 1,827 / 1,818 |
| 8 | 229 | 1,004-1,006 | 1,240-1,275 | 1,990 / 2,006 |

The weights are loaded once whatever the pool (1,005 MB after load, the warm-up call included); a pool of 4 costs about +110 MB after warm-up and +180-310 MB at peak against pool 1 (the working memory of four forward passes of up to 16 x 512 tokens), pool 8 about +500 MB at peak. A copy of the model per slot would cost about 1 GB each.

### 18.3 Questions during ingestion: slot policy (`query_latency.py`)

**Method.** In one process, 12 threads (the 12 ingestion jobs of the 16-CPU default) loop over the 60 passage calls while one thread asks 60 corpus questions one after the other, 0.1 s apart (`embed_query` latency, slot wait included); "idle" is the same 60 questions without load. Policies: **none** (a query competes with the passages for the next free slot), **priority** (a waiting query takes the next free slot before any waiting passage batch), **reserved** (priority plus one slot only queries may use: `pool_size` passage batches and `pool_size` + 1 calls at most). The `none` cells stop the loaders after 120 s.

| pool | policy | idle p50 / p95 (ms) | loaded: questions answered | loaded p50 / p95 / p99 / max (ms) | passage batches/s |
|---|---|---|---|---|---|
| 2 | none | 47.9 / 58.5 | **1, starved**: still waiting at 125 s | 129,835 (the one question; it ran when the loaders stopped) | 1.90 |
| 4 | none | 47.1 / 56.9 | **1, starved**: still waiting at 125 s | 126,756 (as above) | 2.59 |
| 1 | priority | 50.0 / 61.2 | 60 | 1,188 / 2,025 / 2,104 / 2,150 | 0.81 |
| 2 | priority | 48.8 / 57.8 | 60 | 573 / 1,278 / 1,548 / 1,811 | 1.59 |
| 4 | priority | 49.3 / 61.3 | 60 | 344 / 886 / 1,039 / 1,151 | 2.51 |
| 1 | **reserved** | 47.0 / 61.7 | 60 | **53 / 67 / 70 / 71** | 1.08 |
| 2 | **reserved** | 48.2 / 57.9 | 60 | **62 / 82 / 88 / 89** | 1.73 |
| 4 | **reserved** | 49.3 / 58.2 | 60 | **81 / 102 / 106 / 106** | 2.38 |

Without priority a question is **starved**: a thread that has just finished a batch takes the freed slot again before a woken waiter can (the condition is not fair, nor was the former lock), so the question waited for minutes and ran only when the passage threads stopped. Priority bounds the wait by the remaining time of a running batch (p50 0.3-1.2 s, p95 0.9-2.0 s, growing as the pool shrinks). The reserved slot removes the wait: loaded questions take 53-81 ms (p95 67-102 ms) against 47-49 ms idle, and the passages' rate does not drop (1.08 / 1.73 / 2.38 batches/s against 0.81 / 1.59 / 2.51 with priority alone; these 7-10 s windows are short and noisy, and a query uses 2 threads for about 50 ms). **Decision: priority plus one reserved slot for queries** (as built). The passage batches themselves still took the slot in no particular order, as the NER pipelines did (a 50-page ingestion makes 4-5 calls of 16 chunks and a short one usually one); both are handed out in arrival order since 18.6.

### 18.4 End to end `/api/ingest` through HTTP (`http_ingest.py`)

**Method.** Block 5's method (15.1): one Uvicorn process per configuration in the test container, `BUSY_WAIT_S=300` (nothing rejected), a fresh collection prefix per server (every ingestion `indexed`), warm-up 2 sequential and 8 concurrent ingestions of other documents; then the 75-document ingest set of 17.1 (the 60 sample documents and per language the first three corpus documents outside the sample; 125 pages) from 16 client threads, while one more client asks a corpus question every 0.5 s (`RAG_THRESHOLD=1.0`, so every question is embedded and searched, then refused below the threshold; no language model is needed). Two passes per configuration, the second in reverse order. The client ran in the server's container (as in 13-15, unlike 17), so absolute figures are not those of 17.3. "CPUs used" is from `/proc/stat` over the cell (host-wide, client included). The grid ran with **priority only** (before the reserved slot was added); the rows marked "reserved" are the confirmation runs with the code as built, in a later session.

| CPUs | J / NER / emb pool | docs/min (passes) | mean | pages/min | client p50 / p95 (s) | CPUs used | OCR / embed ms per page | question p50 / p95 / max (s) | question `embed_ms` p50 / p95 | server peak (MB) |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 12 / 8 / 1 (former) | 43.8, 47.8 | 45.8 | 76.4 | 17.1 / 36.1 | 5.0 | 2,144 / 6,739 | 0.76 / 1.64 / 2.30 | 753 / 1,631 | 2,472-2,516 |
| 16 | 12 / 8 / 2 | 69.4, 71.4 | 70.4 | 117.3 | 11.5 / 20.1 | 8.6 | 2,390 / 3,389 | 0.58 / 1.25 / 1.71 | 523 / 1,215 | 2,498-2,566 |
| 16 | **12 / 8 / 4** | 86.8, 85.8 | **86.3** | 143.9 | 10.3 / 13.7 | 12.7 | 2,868 / 1,804 | 0.48 / 1.35 / 2.12 | 434 / 1,324 | 2,630-2,635 |
| 16 | 12 / 8 / 6 | 84.7, 83.8 | 84.2 | 140.4 | 10.4 / 14.3 | 13.0 | 3,022 / 1,739 | 0.41 / 1.09 / 1.47 | 388 / 1,065 | 2,781-2,801 |
| 16 | 12 / 8 / 8 | 85.2, 85.4 | 85.3 | 142.2 | 10.1 / 14.4 | 13.1 | 2,993 / 1,739 | 0.34 / 0.83 / 1.19 | 297 / 767 | 2,784-2,824 |
| 16 | 12 / 8 / 4, reserved | 81.5, 83.7 | 82.6 | 137.6 | 10.5 / 14.5 | 12.8 | 2,946 / 1,946 | 0.30 / 0.65 / 1.08 | 277 / 627 | 2,690-2,757 |
| 8 | 6 / 4 / 1 | 47.7, 47.6 | 47.7 | 79.4 | 18.7 / 25.1 | 4.9 | 1,903 / 2,474 | 0.65 / 1.46 / 1.90 | 643 / 1,454 | 2,323-2,354 |
| 8 | **6 / 4 / 2** | 62.8, 64.8 | **63.8** | 106.3 | 14.2 / 17.4 | 7.3 | 2,193 / 1,056 | 0.26 / 1.23 / 1.88 | 258 / 1,229 | 2,453-2,495 |
| 8 | 6 / 4 / 4 | 62.3, 62.6 | 62.4 | 104.0 | 14.5 / 17.7 | 7.5 | 2,299 / 1,015 | 0.10 / 0.27 / 0.92 | 86 / 259 | 2,464-2,515 |
| 8 | 6 / 4 / 2, reserved | 62.0, 61.5 | 61.8 | 103.0 | 14.8 / 17.8 | 7.3 | 2,257 / 1,092 | 0.09 / 0.20 / 0.30 | 80 / 166 | 2,424-2,441 |
| 4 | **3 / 2 / 1** | 37.2, 37.1 | **37.1** | 61.9 | 24.9 / 28.8 | 3.8 | 1,918 / 925 | 0.36 / 1.54 / 2.11 | 359 / 1,536 | 2,274-2,361 |
| 4 | 3 / 2 / 2 | 36.9, 37.7 | 37.3 | 62.2 | 25.1 / 28.6 | 4.0 | 2,016 / 827 | 0.09 / 0.65 / 1.70 | 74 / 642 | 2,349-2,373 |
| 4 | 3 / 2 / 1, reserved | 35.9, 36.2 | 36.1 | 60.2 | 25.6 / 29.8 | 3.8 | 1,980 / 948 | 0.08 / 0.16 / 0.24 | 66 / 139 | 2,359-2,376 |

Every request of every cell returned 200 (75 ingestions and 49-214 questions per cell); no warning or error was logged; servers were ready in 10.7-12.1 s.

**Findings.** On 16 CPUs the former single embedder gives 45.8 docs/min with 5.0 CPUs busy and 6.7 s of embedding per page (the 17.3 picture: 50.9 docs/min, 4.9 CPUs, 5.8 s); **pool 4 gives 86.3 docs/min (1.9 x)** with 12.7 CPUs busy and 1.8 s of embedding per page, and the client p95 falls from 36 to 14 s. Beyond 4 nothing is gained (84.2 and 85.3 at 6 and 8): OCR becomes the limit again (2.9-3.0 s per page) as in 15.4. With a quota of 8 CPUs pool 2 gives 63.8 docs/min against 47.7 for pool 1 (+34 %) and 62.4 for pool 4; with 4 CPUs pools 1 and 2 are equal (37.1 and 37.3). Ingestion now reaches the extraction rate of the same CPUs (17.3: 85.3, 61.4 and 37.5 docs/min at 16, 8 and 4 CPUs with the sizing table). The reserved slot shortens the question's wait for a slot without a measurable cost: question `embed_ms` p95 0.63 s against 1.32 s on 16 CPUs (the rest is CPU contention: 13 of 16 CPUs busy), 0.17 s against 1.23 s on 8 and 0.14 s against 1.54 s on 4, at 82.6, 61.8 and 36.1 docs/min, 2.7-4.3 % below the grid's 86.3, 63.8 and 37.1 (a later session; the faster answers also let the question client ask 29-86 % more questions in a cell). Memory: the server peak rises by about 120-160 MB from pool 1 to pool 4 (2.47-2.52 against 2.63 GB) and by 300 MB to pool 8.

### 18.5 Choice and the derived defaults

D7's rule (the smallest setting within 5 % of the best throughput), on ingestion docs/min:

- **16 CPUs:** best 86.3 (pool 4), band from 82.0; pool 2 (70.4, -18 %) is outside, **pool 4 is the smallest inside** (6 and 8 are inside too).
- **8 CPUs (J 6):** best 63.8 (pool 2), band from 60.6; pool 1 (47.7) is outside, **pool 2** is the smallest inside (4: 62.4).
- **4 CPUs (J 3):** best 37.3 (pool 2), band from 35.4; **pool 1** (37.1) is inside.

So the embedder pool is CPUs / (2 x `TORCH_THREADS`), at most 4: half the CPUs a pool of `CPUs / TORCH_THREADS` would use, because OCR (one thread per Tesseract process) needs the rest. **Applied:** `EMBEDDING_POOL_SIZE` unset = derived, `E5EmbedderConfig.pool_size` = 4 (the 16-CPU value, for standalone use), one reserved query slot. The other three values follow the measured sizing table of 17.4 and keep the existing OCR default; `concurrency_for(cpus, torch_threads)` in `app/bootstrap.py` is the one rule:

- `MAX_CONCURRENT_JOBS` = three quarters of the CPUs rounded up, at most 12 (rounding up because one job too few cost more than one too many where measured: at 4 CPUs 2 jobs -22 %, 4 jobs -3 %; at 8 CPUs 4 jobs -18 %, 8 jobs -4 %, 17.3);
- `NER_POOL_SIZE` = CPUs / `TORCH_THREADS` rounded down, at most 8 (the largest pool without the oversubscription warning, 17.4);
- `EMBEDDING_POOL_SIZE` = CPUs / (2 x `TORCH_THREADS`) rounded down, at most 4 (above);
- `OCR_WORKERS` = the CPUs, at most 4 (the former `min(4, cpu_count)`, now with the quota: 15.2 found 3, 4 and 5 equivalent on 16 CPUs; not re-measured under a quota);
- each at least 1; a set value always wins; the values in use are logged once at start-up (`concurrency in use`, with `cpus_available`) and a warning is logged per pool whose size x `TORCH_THREADS` exceeds the CPUs (the derived pools never do).

| CPUs available | 1 | 2 | 4 | 6 | 8 | 12 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|
| `MAX_CONCURRENT_JOBS` | 1 | 2 | **3** | 5 | **6** | 9 | **12** | 12 |
| `NER_POOL_SIZE` (T 2) | 1 | 1 | **2** | 3 | **4** | 6 | **8** | 8 |
| `EMBEDDING_POOL_SIZE` (T 2) | 1 | 1 | **1** | 1 | **2** | 3 | **4** | 4 |
| `OCR_WORKERS` | 1 | 2 | **4** | 4 | **4** | 4 | **4** | 4 |

Bold columns are measured (17.3 / 17.4 for jobs and NER pool, 18.4 for the embedder pool, 15.2 and 17.3 for OCR workers at 4); the others follow the rule and were not measured. Above 16 CPUs the caps keep the 16-CPU values, so a larger host is not used fully by one process (replicas, `KNOWN_LIMITATIONS.md` P4).

**Caveats.** One host, synthetic scans, short documents; the 50-page workload and mixed extract + ingest load were not repeated with the embedder pool (they share the CPUs, and the job limit bounds both pools together). The client ran in the server's container. The grid and the confirmation runs are different sessions (2.7-4.3 % apart for the same pool), so the reserved slot's throughput cost is bounded, not measured exactly; its question latency gain is large and consistent (18.3, 18.4). The 8- and 4-CPU points are CPU quotas on a 16-CPU host (threads still spread over 16 CPUs), not smaller machines. A deployed-stack re-measurement with the derived defaults is task 6.3 (section 19).

### 18.6 FIFO hand-off of the NER pipelines and the embedder slots (`chain_fifo.sh`, `fifo_tables.py`)

**Problem.** Both pools admitted callers through a plain `threading.Condition`: a freed slot went to whichever thread took the lock next, which is usually the thread that has just released it and calls again before a woken waiter runs (18.2: with pool 1 some calls waited 58 s while others took 1 s; 18.3: a query without priority was starved for minutes). This was `KNOWN_LIMITATIONS.md` P11. **Change (as built).** `app/adapters/slots.py` (`FifoSlots`, shared by both adapters) hands a freed slot directly to the first waiter of its queue (one condition per waiter on one lock), so a caller that arrives, or comes back, while others wait queues behind them. Lanes keep the embedder's policy: queries (lane 0) may use every slot including the reserved one and are served first, passage batches (lane 1) hold at most `EMBEDDING_POOL_SIZE` slots; each lane is FIFO. The NER pool is one lane of `NER_POOL_SIZE` slots. `close()`, the error codes and messages, and the vectors and spans are unchanged (the models' calls are the same; only the order of admission changes). Unit tests: `tests/adapters/test_slots.py` (arrival order, a releasing thread does not overtake a waiter, lane priority and limits, close, interrupted waits) and ordering tests in the recogniser and embedder tests (a job calling page after page, or a call with several batches, does not overtake a waiting call; waiting queries before waiting passages, each in arrival order).

**Method.** The scripts of 18.2 (`bench_one.py`, now also recording every call's latency and the calls served per worker thread), 18.4 (`http_ingest.py`, now also recording each job's `embed_ms` and `duration_ms`) and 15.2 (`p15.py`, each job's `ner_ms` and `duration_ms` from the outcome lines), run with the test image before the change (`docintel-test:prefifo`, the tree of task 6.2) and after it (`docintel-test:fifo`), one pass per run in the order before, after, after, before (the extract cells also once more before, and pool 4 four more passes, after, before, after, before); compose project `fifoev`, `down -v` after every script; otherwise as in 18.2 and 18.4 (client in the server's container, 16 CPUs or a `cpus: 8` quota). "CV" is the standard deviation over the mean. Per-job stage times are per page of the job, waiting for a slot included.

**Embedding alone** (16 threads pulling the 60 passage calls from a queue):

| pool | image | chunks/s (passes) | call p05 / p50 / p95 / max (s) | CV | calls served per thread |
|---|---|---|---|---|---|
| 1 | before | 5.49, 5.39 | 0.51-0.52 / 1.23-1.25 / 57.0-58.1 / 59.6-60.7 | 1.61 | 1-45 |
| 1 | **after** | 5.40, 5.45 | 3.46-3.56 / 15.4-15.6 / 18.1-18.2 / 18.7-18.8 | 0.33-0.34 | **3-4** |
| 4 | before | 15.14, 15.20 | 0.73-0.77 / 1.63 / 20.3-20.5 / 21.5-21.6 | 1.44 | 1-14 |
| 4 | **after** | 15.54, 15.57 | 1.35-1.38 / 5.08 / 6.38-6.48 / 7.18-7.38 | 0.32-0.33 | **3-5** |

Before, a few threads kept the slot for most of the run (one served 45 of the 60 calls with pool 1) and the others waited up to a minute; after, every thread gets 3-5 calls and the slowest call waits 18.8 s instead of 60.7 s (pool 1) or 7.4 s instead of 21.6 s (pool 4). The median rises because it now includes the fair share of waiting (15 calls ahead with pool 1). Throughput is unchanged (pool 1: 5.40-5.45 against 5.39-5.49 chunks/s; pool 4: 15.5-15.6 against 15.1-15.2).

**End to end** (75 requests from 16 clients, one pass per run; per-job values over the 75 jobs of a run):

| workload, CPUs, J / NER / emb | image | docs/min (passes) | mean | client p50 / p95 / max (s) | stage ms per page per job: p50 / p95 / max (s) | CV | job s per page p95 / max |
|---|---|---|---|---|---|---|---|
| ingest, 16, 12 / 8 / 4 | before | 87.4, 89.7 | 88.6 | 9.9-10.0 / 12.6-13.7 / 14.3-16.9 | embed 1.72-1.74 / 2.75-3.26 / 3.39-7.72 | 0.30-0.49 | 7.13-7.57 / 7.98-11.28 |
| ingest, 16, 12 / 8 / 4 | after | 89.2, 89.3 | **89.3** | 9.9-10.0 / 13.3-13.5 / 14.4-16.0 | embed 1.75-1.81 / 2.73-2.80 / 2.97-3.33 | 0.27-0.29 | 6.81-7.15 / 8.25-8.37 |
| ingest, `cpus: 8`, 6 / 4 / 2 | before | 62.2, 64.6 | 63.4 | 13.9-14.7 / 17.2-17.8 / 17.8-19.0 | embed 1.01-1.02 / 1.83-1.94 / 2.41-2.45 | 0.37-0.40 | 5.05-5.26 / 5.59-6.60 |
| ingest, `cpus: 8`, 6 / 4 / 2 | after | 64.8, 64.6 | **64.7** | 13.9-14.1 / 16.9-17.1 / 17.7-17.8 | embed 0.97-1.00 / 1.80-2.09 / 2.20-2.25 | 0.36-0.38 | 5.17-5.26 / 5.81-6.07 |
| extract, 16, 12 / 8 | before | 83.6, 84.4 | 84.0 | 10.5-10.6 / 13.3-13.5 / 15.0-16.9 | NER 1.83-1.93 / 2.69-2.73 / 3.22-3.32 | 0.31 | 5.90 / 7.65-8.10 |
| extract, 16, 12 / 8 | after | 84.7, 84.7 | **84.7** | 10.6-10.9 / 13.8-13.9 / 16.3-17.1 | NER 1.86-1.89 / 2.59-2.71 / 3.03-3.31 | 0.30-0.31 | 5.95-5.97 / 7.91-7.93 |
| extract, 16, 12 / 4 | before | 85.1, 86.2, 85.6, 86.1 | 85.8 | 10.4-10.6 / 13.0-13.9 / 14.7-17.2 | NER 1.77-1.87 / 2.96-3.28 / 3.72-4.72 | 0.33-0.39 | 6.42-7.40 / 8.23-9.46 |
| extract, 16, 12 / 4 | after | 83.7, 84.3, 85.9, 85.2 | **84.8** | 10.6-10.8 / 13.2-13.7 / 14.8-16.7 | NER 1.88-1.99 / 2.50-2.83 / 3.22-3.38 | 0.27-0.30 | 5.77-6.24 / 7.52-7.93 |

Every request of every cell returned 200. **Throughput is unchanged within the run-to-run spread**: ingestion +0.8 % (16 CPUs) and +2.1 % (8 CPUs), extraction +0.8 % at the default pool 8 and -1.1 % at pool 4 (two passes each, four at pool 4; the passes of one image differ by up to 3.9 %). **The spread between jobs narrows where jobs compete for a pool**: with 12 extraction jobs on 4 NER pipelines the slowest job's NER time per page falls from 3.7-4.7 s to 3.2-3.4 s, the 95th percentile from 3.0-3.3 to 2.5-2.8 s and the coefficient of variation from 0.33-0.39 to 0.27-0.30 (job time per page p95 6.4-7.4 s against 5.8-6.2 s); on the 16-CPU ingestion runs the worst job's embedding time per page was 7.7 s in one before-run (3.4 s in the other) against at most 3.3 s after. With the default 8 pipelines for 12 jobs, and with 2 passage slots for 6 jobs at 8 CPUs, the per-job spread is the same before and after (CV 0.30-0.31 and 0.36-0.40); client latency (dominated by the job limit and the queue of 16 clients for 12 jobs) is the same. Questions keep the reserved slot and their priority: question `embed_ms` p50 / p95 during ingestion 226-272 / 592-608 ms after against 217-230 / 606-634 ms before on 16 CPUs, and 69-72 / 131-170 against 72-75 / 167-215 ms on 8 (18.4: p95 627 and 166 ms).

**Caveats.** Short documents only: the case P11 named (a 50-page ingestion's 4-5 batches overtaking shorter jobs) is covered by the unit tests and by the embedding-alone runs, not by a 50-page HTTP run. One host, client in the server's container, one pass per run (four for extraction at pool 4).

---

## 19. Deployed stack after the embedder pool, the derived defaults and the FIFO hand-off (`add-deployment` task 6.3)

Scripts: `benchmarks/deploy_perf/scripts/` as in section 17, driven by `chain19.sh` (new commands `gpu-askload` and `smoke` in `run.sh`, `askload` in `client.py`; tables by `tables.py`, `askload_table.py`, `cold_table.py` and `parity.py`). Raw outputs in `benchmarks/deploy_perf/out19/` (local, not committed).

### 19.1 Method and environment

As 17.1, with the tree after task 6.2 and the FIFO hand-off of 18.6: images `docker build --target runtime -t docintel .` (`sha256:f76a09ff...`, 5.84 GB) and the test image from `docker compose -f docker-compose.test.yml build tests` (`sha256:fe0db350...`, 5.75 GB), both on 2026-10-28; same host, Docker Desktop 29.9.2, Compose 5.6.1, nothing else running. The stack is the repository's `docker-compose.yml` and `docker-compose.smoke.yml`, started from a copy with its own `.env` (`out19/stack/`, `compose.diff` shows only `build.context` changed), so the repository-root `.env` took no part; no hosted model was called. **No concurrency setting was set**: the copy's `.env` held only `BUSY_WAIT_S=300` for the timed cells (as in 17, so nothing is shed) and nothing for the cold-start, parity and no-model smoke runs, so `MAX_CONCURRENT_JOBS`, `NER_POOL_SIZE`, `EMBEDDING_POOL_SIZE` and `OCR_WORKERS` were derived at start-up from the CPUs (18.5). CPU limits by `cpus:` on the app service. Compose projects `perf_tp`, `perf_cold`, `perf_par`, `perf_smoke` and `perf_gpu`, each torn down with `down -v`; the evaluation volumes of section 16 were only mounted read-only (copied by `gpu-setup`).

**Derived values in use**, from every run's `concurrency in use` line (`cpus_available`, jobs / NER pool / embedder pool / OCR workers): 16 -> 12 / 8 / 4 / 4, `cpus: 8` -> 8 -> 6 / 4 / 2 / 4, `cpus: 4` -> 4 -> 3 / 2 / 1 / 4, the values of the 18.5 table, in every run (throughput, cold start, parity, smoke, both question runs); no oversubscription warning in any run.

### 19.2 Extract and ingest throughput with the derived defaults

Same cells as 17.3 (75 extractions and 75 ingestions from 16 clients in their own container, fresh stack per run, two rounds, the second in reverse order of CPUs); every request returned 200:

| app CPUs | 17.3, defaults then (J / P) | 17.3 extract docs/min | 17.3 ingest docs/min | derived now (J / NER / emb / OCR) | extract docs/min (rounds) | extract p50 / p95 (s) | ingest docs/min (rounds) | ingest p50 / p95 (s) | CPUs used: extract / ingest | throttled: extract / ingest | server peak (MB): extract / ingest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 12 / 8 | 85.3 | 50.9 | 12 / 8 / 4 / 4 | 83.5, 84.5 = **84.0** | 10.6-10.7 / 13.3-14.0 | 90.3, 91.0 = **90.7** | 9.4-9.6 / 12.9-13.2 | 12.9-13.0 / 12.5 | 0 / 0 % | 2,552-2,595 / 2,919-2,973 |
| 8 | 12 / 8 (warning) | 54.0 (6 / 4: 61.4) | 49.8 (6 / 4: 50.3) | 6 / 4 / 2 / 4 | 61.9, 62.2 = **62.0** | 14.5-14.9 / 16.9 | 65.5, 66.3 = **65.9** | 13.7-13.8 / 16.4-16.7 | 7.3 / 6.9-7.0 | 56-58 / 30-33 % | 2,295-2,305 / 2,671-2,732 |
| 4 | 12 / 8 (warning) | 27.9 (3 / 2: 37.5) | 39.2 (3 / 2: 37.6) | 3 / 2 / 1 / 4 | 37.3, 37.3 = **37.3** | 24.9-25.1 / 26.9-27.0 | 38.1, 38.1 = **38.1** | 24.1-24.3 / 27.9 | 3.8 / 3.5 | 58 / 34-38 % | 2,152-2,176 / 2,523-2,540 |

Per-page stage times (outcome lines): extraction OCR / NER 2.69-2.71 / 1.76-1.80 s at 16 CPUs, 2.09 / 1.01-1.02 s at 8, 1.83-1.84 / 0.78-0.79 s at 4; ingestion OCR / embedding 2.77 / 1.63-1.67 s, 2.11 / 1.00-1.03 s and 1.86-1.88 / 0.89-0.90 s.

**Against 17.3.** Without any setting the stack now gets what 17.3 needed hand-picked settings for: at `cpus: 8` extraction runs at 62.0 docs/min against 54.0 with the old fixed defaults (+15 %; 61.4 with the best 6 / 4), at `cpus: 4` 37.3 against 27.9 (+34 %; 37.5 with 3 / 2), each within 1 % of the best setting of 17.4, without the oversubscription warning. At 16 CPUs extraction is 84.0 against 85.3 (-1.5 %, within the spread between sessions; the configuration is the same apart from the FIFO hand-off, which 18.6 measured as neutral at pool 8). **Ingestion is no longer capped by the embedder**: 90.7 docs/min (151 pages/min) at 16 CPUs against 50.9, with 12.5 CPUs busy instead of 4.9 and 1.6-1.7 s of embedding per page instead of 5.8-5.9 s; 65.9 against 49.8-50.3 at 8 CPUs; at 4 CPUs 38.1 against 37.6-39.2 (one embedder slot was enough there already). Ingestion now matches or exceeds extraction on the same CPUs (17.3's finding 1 resolved), and the client p95 of an ingestion at 16 CPUs is 12.9-13.2 s against 20.4 s. The server peak during ingestion is 2.9-3.0 GB at 16 CPUs against 2.5-2.7 GB (the working memory of four concurrent forward passes, 18.2); the container's sampled memory stayed at 1.4-2.7 GB.

### 19.3 Questions during ingestion (`gpu` profile)

**Method.** `gpu` stack as in 17.5 (own volumes with a copy of the 600-document index and `gemma4:e4b-it-qat`, defaults, thinking off by the compose start command; checked in `gpu_env.txt`: `LLM_BASE_URL=http://ollama:11434/v1`, `LLM_REASONING_EFFORT=none`, no `LLM_API_KEY`, `OLLAMA_NUM_PARALLEL=2`). After a restart of the stack (17.6 (b)), 3 warm-up questions, then 60 questions from one client with no load ("idle"), then the 75-document ingest set from 16 clients while one more client asks the next answerable question of section 16's order every 0.5 s until the ingestion ends (`client.py askload`). Each ingested file had a comment line appended after `%%EOF`, which changes its sha256 and therefore its document id but not its content, so every ingestion OCRed, embedded and stored (75 / 75 `indexed`) although the index already held the originals. Repeated at `cpus: 8` (stack recreated with the quota). Service times from the questions' `question completed` lines.

| CPUs | load | questions (all 200) | client p50 / p95 / max (s) | `embed_ms` p50 / p95 / max | `search_ms` p50 / p95 | `llm_ms` p50 / p95 / max (s) | ingestion during the run |
|---|---|---|---|---|---|---|---|
| 16 | none | 60 | 0.60 / 1.84 / 2.14 | 53 / 69 / 82 | 4 / 5 | 0.53 / 1.78 / 2.09 | - |
| 16 | ingestion, 16 clients | 37 | **0.84 / 2.31 / 2.48** | **228 / 538 / 688** | 11 / 118 | 0.55 / 1.91 / 2.13 | 80.9 docs/min, 75 x 200 |
| 8 | none | 60 | 0.56 / 1.86 / 2.14 | 52 / 65 / 72 | 3 / 12 | 0.50 / 1.80 / 2.10 | - |
| 8 | ingestion, 16 clients | 55 | **0.55 / 1.99 / 8.79** | **78 / 174 / 391** | 4 / 14 | 0.45 / 1.86 / 8.71 | 60.1 docs/min, 75 x 200 |

A question during a full ingestion load costs 0.2-0.5 s more at 16 CPUs (embedding p50 228 ms against 53 ms idle: the query has its reserved slot and priority, 18.3, but its forward pass shares the CPUs with 12 jobs, the app container using 11.6 CPUs; 18.4 measured p50 277, p95 627 ms the same way) and little at 8 CPUs (embedding 78 / 174 ms against 52 / 65 ms idle, client latency unchanged; the quota most likely leaves the other containers' share of the host free, not measured). The language model dominates in both cases (`llm_ms` p50 0.45-0.55 s, unchanged by the load; GPU-bound). The one slow question at 8 CPUs (8.8 s) was slow in the model call (`llm_ms` 8.7 s), not in the service; single occurrence, like the 7.2 s outlier of 17.5. **Before the embedder pool** (17.3 ingestion: one embedder under one lock, 5.8 s of embedding per page) a question's query waited behind the ingestion's passage batches (18.3: 0.3-1.2 s p50 with priority alone, starved without). The ingestion rates (80.9 and 60.1 docs/min) are below 19.2's (90.7 and 65.9) because this run is a different method: no ingestion warm-up after the restart, and Ollama, its CPU-side work and the question client share the host (OCR per page 2.98 against 2.77 s, embedding 2.02 against 1.63-1.67 s, the app container used 11.6 CPUs against 12.5; not separated further).

### 19.4 Cold start with `start_interval`

Same commands as 17.6 (`run.sh cold`, `gpu-first`, `gpu-restart`); breakdown from `docker inspect` (`cold_table.py`; the Ollama figure is its start to the start of `ollama-pull`, which waits for Ollama's healthcheck, because the inspect keeps only the last five checks):

| case | runs | `up -d --wait` (s), 17.6 | `up -d --wait` (s), now | Qdrant start to app start (s), 17.6 -> now | Ollama start to healthy (s), 17.6 -> now | app start to healthy (s), 17.6 -> now | first question (s), 17.6 -> now |
|---|---|---|---|---|---|---|---|
| (a) no profile, new containers and new Qdrant volume | 1 | 20.1 | **13.64** | 5.7 -> 1.66 | - | 13.4 -> 11.14 | - |
| (a) no profile, from a stopped stack | 3 | 17.4-17.9 | **13.38-13.42** | 5.7 -> 1.65-1.68 | - | 11.2 -> 11.12-11.15 | - |
| (a) no profile, recreated containers (`down` without `-v`) | 2 | 17.6, 17.6 | **13.62, 13.63** | 5.7 -> 1.68 | - | 11.2 -> 11.13-11.14 | - |
| (a) no profile, fresh stack of each throughput run | 6 | 17.6-18.3 | 13.63-16.16 (median 13.65) | - | - | - | - |
| (b) `gpu`, fresh containers, model and CUDA kernels in the volume | 3 | 19.7, 20.5, 23.4 | **19.41, 15.81, 17.18** | 9.3-12.7 -> 6.75-8.66 | 5.9-6.1 -> 2.8-3.5 | 9.0-9.1 -> 6.8-9.0 | 0.92-1.07 -> 0.95-0.98 |
| (b) the same at `cpus: 8` | 1 | - | 21.00 | 10.87 | 3.94 | 8.96 | 0.93 |
| (c) `gpu`, first-ever start (no CUDA kernel cache) | 1 | 85.0 | 85.49 | 74.3 -> 73.9 | 6.1 -> 7.1 | 9.0 -> 9.05 | 1.42 -> 1.20 |

The `start_interval: 1s` of 6.2 removed what 17.6 predicted: the app now starts 1.7 s after Qdrant instead of 5.7 s, and a start without a model takes 13.4-13.6 s instead of 17.4-20.1 s (4.0-4.5 s less, and 6.5 s less with a new Qdrant volume). With the `gpu` profile Ollama is healthy 2.8-3.5 s after its start instead of 5.9-6.1 s and the stack is up in 15.8-19.4 s instead of 19.7-23.4 s; the first question is answered in under 1 s as before. The first-ever start is unchanged (85.5 s), because it is the warm-up prompt (model load and CUDA kernel compilation, `ollama-pull` 67.6 s) that takes the time. What remains of a start without a model is the app itself: 10.2 s from container start to `Application startup complete` (imports and a 4.2 s lifespan), and up to one interval of the image's own healthcheck (`start_interval` 2 s): the app was marked healthy 0.9 s after its startup completed in the run checked. A 1 s interval there would save about half a second on average; every measurement in this table used the 2 s probe, and the 1 s interval was adopted afterwards (next paragraph, task 7.1).

**The image's healthcheck at `start_interval` 1 s (task 7.1).** Adopted after this measurement, on the isolated stack of 19.7 without the profile (`benchmarks/t71/cold.py`, local): fresh containers and Qdrant volume each run (`down -v`, then `up -d --wait`), one discarded warm-up, then 1 s (the new image) and 2 s (the same image, a compose override changing only `start_interval`) alternated three times; container times from `docker inspect`, `service started` from the app's log line:

| image `start_interval` | runs | `up -d --wait` (s) | Qdrant start to app start (s) | app start to `service started` (s) | lifespan (ms) | `service started` to healthy (s) | app start to healthy (s) |
|---|---|---|---|---|---|---|---|
| 1 s | 3 | 13.48-13.52 | 1.65-1.68 | 10.10-10.21 | 4,143-4,196 | **0.70-0.78** | **10.88-10.91** |
| 2 s | 3 | 13.51-13.53 | 1.66-1.68 | 10.14-10.18 | 4,169-4,192 | 0.95-1.02 | 11.13-11.15 |

The app is marked healthy 0.22-0.27 s earlier in every pair; `up -d --wait` itself returned after the same 13.5 s in both cases (compose polls the containers' state for `--wait`; its granularity was not measured), so the gain shows for whatever waits on the health status (`depends_on: service_healthy`, an orchestrator), not in this command. Load: one probe costs 157 ms of CPU (user 143, system 14 ms; 160 ms wall at p50, 176 ms at most; 20 probes in a running container, `getrusage` of the child processes), and Docker counts the interval from the end of the previous probe, so while starting the probes use about 0.13 CPU at 1 s against 0.07 at 2 s, of 16; the start itself did not change (app start to `service started` and the lifespan within 0.1 s and 50 ms of each other). No measurable load, so the image now probes every second while starting (`Dockerfile`, design D12).

### 19.5 Parity and smoke suite

**Parity** (`parity.py`, as 17.7: the 60 sample documents through the stack, runtime image against the test image as the app service, 4 clients): extraction JSON bodies **60 / 60 identical** (805 names, each with its box, 86 matches), ingestion status, result and `chunks_stored` **60 / 60 identical** (327 chunks, all `indexed`); the runtime run is also identical, body for body, to 17.7's runtime run. Wall times 66.2 s extraction for both images; ingestion 68.9 s for both (78.5 s in 17.7, the embedder pool).

**Smoke suite** (`docker compose -f docker-compose.yml -f docker-compose.smoke.yml [--profile gpu] run --rm smoke`, from the copy): without the profile **12 passed, 2 skipped** (the answerable-question variants: no model configured), exit 0; with the `gpu` profile **12 passed, 2 skipped** (the "without a model" variants), exit 0. `run --rm hygiene`: all 13 checks `ok` both times, exit 0.

### 19.6 Findings

- **Ingestion is no longer capped by the embedder** on the deployed stack: 90.7 docs/min at 16 CPUs (50.9 in 17.3), 65.9 at 8, 38.1 at 4 (17.9 finding 1, fixed by task 6.2).
- **The derived defaults reproduce the best measured settings** at 16, 8 and 4 CPUs without configuration and without the oversubscription warning: extraction 84.0 / 62.0 / 37.3 docs/min against 85.3 / 61.4 / 37.5 for the best settings of 17.4 (17.9 finding 3).
- **Start-up is about 4 s shorter** with `start_interval` on the Qdrant and Ollama healthchecks (17.9 finding 2).
- **Questions during ingestion** stay under 2.5 s at p95 on 16 CPUs (embedding 0.2-0.5 s under CPU contention) and are unaffected at 8 CPUs; the reserved slot and the FIFO hand-off keep the query's wait for a slot out of it.
- Nothing measured here called for a code or configuration change; no new limitation.

**Caveats.** One host, one session, two rounds for throughput, one run per question configuration. The question runs use the first answerable questions of section 16's order, a single client for the questions, and an ingestion load without warm-up (19.3). The first-ever start ran with the model blobs probably in the page cache (as in 17.6). CPU quotas on a 16-CPU host, not smaller machines (17.9).

### 19.7 Task 7.1: the language model key as a compose secret, probe access lines, smoke suite

**Isolated stack.** A copy of the tree of this task in `benchmarks/t71/stack/` (local; made with `tar` excluding `.env`, `.venv`, `.git` and `benchmarks/`, so the repository-root `.env` took no part), images `docker build --target runtime -t docintel:t71` (`sha256:de93b466...`, 5.84 GB) and `--target test -t docintel-test:t71` (`sha256:c8fbb8dc...`, 5.75 GB), Docker 29.9.2 and Compose 5.6.1, compose projects `t71key`, `t71nokey`, `t71guard` and `t71cold`, each torn down with `down -v`. The copy's own `.env` held a fake key (`sk-test-...-0000`, 27 characters, written out in full in that `.env`), `LLM_BASE_URL=http://fakellm:8080/v1` and `LLM_MODEL=fake-model`; `fakellm` is a 40-line Python server on the internal network (compose override of the copy) that logs, per request, whether the `Authorization` header equals `Bearer <fake key>` by comparing sha256 digests. No hosted model was called.

**Compose behaviour checked first** (throw-away probe projects): a secret with `environment: LLM_API_KEY` is resolved from `.env` (and the shell) and mounted read-only (`-r--r--r--`, root) at `/run/secrets/<name>`, readable by uid 10001; with the variable unset compose refuses to create the container (`environment variable "LLM_API_KEY" required by secret ... is not set`); with it set to the empty string no file is mounted; `file` and `environment` are mutually exclusive even when one is empty; secrets have no `content` source (`configs` have, and `config` prints it). Hence the design of D11: source `llm_api_key${LLM_API_KEY:+_from_env}`, an empty committed file in the no-key case.

| check (key in `.env`, project `t71key`) | result |
|---|---|
| occurrences of the key in `docker inspect t71key-app-1` | **0** |
| the app container's environment (`Config.Env`, `LLM_*`) | `LLM_BASE_URL=http://fakellm:8080/v1`, `LLM_MODEL=fake-model`, `LLM_API_KEY=`, `LLM_API_KEY_FILE=/run/secrets/llm_api_key` |
| environment of tini and the Uvicorn process (`/proc/<pid>/environ`) | `LLM_API_KEY=` and `LLM_API_KEY_FILE=/run/secrets/llm_api_key` only |
| occurrences of the key in `docker compose config` | **0**; the app's environment shows `LLM_API_KEY: ""` and `LLM_API_KEY_FILE: /run/secrets/llm_api_key`, its `secrets` `source: llm_api_key_from_env`, `target: llm_api_key`, and the top-level secret `name: t71key_llm_api_key_from_env`, `environment: LLM_API_KEY` |
| `/run/secrets/llm_api_key` | 27 bytes (the key's length), `-r--r--r--` root, read by uid 10001 |
| readiness | `{"status":"ready","checks":{"vector_store":"ok","llm":"configured"}}` |
| `/api/ingest` of `company_memo.pdf`, then "Who was promoted to Vice President of Engineering?" | `indexed`, 2 chunks; 200 with the fake model's answer and the memo's chunks as sources |
| fake server's record of that call | `{"path": "/v1/chat/completions", "auth_present": true, "auth_matches": true, "model": "fake-model"}` |
| occurrences of the key in the logs of every service | **0** |

**Without `.env`** (project `t71nokey`, `APP_IMAGE` and `APP_PORT` from the shell): `docker compose config` resolves the app's secret to `source: llm_api_key`, `file: <project>/deploy/no-llm-api-key`; the stack starts (`up -d --wait`), `/run/secrets/llm_api_key` is 0 bytes and readiness reports `"llm": "not_configured"`. With `LLM_API_KEY` set to the empty string the source is also `llm_api_key`.

**`ollama-pull` guard** (`docker compose --profile gpu run --rm --no-deps ollama-pull`, no Ollama running): with `LLM_BASE_URL=https://api.example.invalid/v1` it exits 1 with `LLM_BASE_URL points at another server; start the stack without the gpu profile`; with `http://ollama:11434/v1` or unset it passes the check and goes on to `ollama pull` (which then fails only because no Ollama server runs here).

**Probe access lines.** In `t71key`, after start (five healthcheck runs recorded by `docker inspect`, the first one before the server listened), my `/health/ready` request, one ingestion and one question, the app's log held exactly two access lines (`/api/ingest`, `/api/ask`): successful probes are logged at DEBUG (design D12; unit tests in `tests/api/test_middleware.py` cover the DEBUG level, a failing probe at INFO and a client request in between).

**Smoke and hygiene** against `t71nokey` (no profile, no model; `docker compose -f docker-compose.yml -f docker-compose.smoke.yml run --rm smoke` and `... hygiene` from the copy): **12 passed, 2 skipped** (the answerable-question variants with a model), exit 0; hygiene **13 / 13 ok**, exit 0. The `gpu` profile was not run in this task (task 8.2).


### 19.8 Task 8.1: review fixes

**Isolated stack.** A copy of the tree in `benchmarks/fx8/stack/` (local; `tar` excluding `.env`, `.venv`, `.git` and `benchmarks/`), images `docintel:fx8` (runtime, 5.76 GB: 5.84 GB before, uv no longer copied in) and `docintel-test:fx8`, compose projects `fx8` (fake Ollama), `fx8d` (small Qdrant volume), `fx8n` (subnet clash) and `fx8g` (real Ollama), each torn down with `down -v`. The copy's `.env` held only fake values (an unrelated `OTHER_TOKEN`, fake keys); no hosted model was called. The fake Ollama is a 90-line Python server in the app image at Ollama's address (`/api/version`, `/api/show`, `/api/generate`, `/v1/chat/completions`; it logs the request path, the model, `reasoning_effort` and a sha256 prefix of the `Authorization` header), with a fake `ollama` command for `ollama-pull` (`show` and `pull` with a chosen exit code). The real run used `gemma4:e4b-it-qat` copied from the evaluation volume into a new volume (6.1 GB, no download) on the RTX 4070 Ti.

| finding | check | result |
|---|---|---|
| 1 failed download | `PULL_EXIT=1` (fake), model absent from the server (fake), `LLM_MODEL=gemma4:no-such-tag` (real Ollama, registry reachable) | `up -d --wait` exit 1 after 3-5 s in each case; `service "ollama-pull"` or `"llm-warmup" didn't complete successfully: exit 1`; the app container `created`, never started; `llm-warmup` names the model and `ollama-pull`'s log; the real one logs `Error: pull model manifest: file does not exist` |
| 2a switch on | plain `up` (readiness `not_configured`), then `COMPOSE_PROFILES=gpu up` | app and `llm-warmup` recreated, readiness `configured`, a question answered by the fake |
| 2b Ollama down | `docker stop` Ollama, `docker restart` the app (no compose) | app healthy after 7 s, still pointed at Ollama; a question 503 `llm_unavailable` (10 s, 3 attempts); after `docker start` Ollama the next question 200 |
| 2c outside answer | a decoy server on the egress network with the alias `ollama`, Ollama stopped | `getent hosts ollama` in the app resolves to the decoy; the decoy received **0** requests; the app's `LLM_BASE_URL` is `http://10.213.47.250:11434/v1` |
| 2 `--profile gpu` alone | no `COMPOSE_PROFILES` | `ollama-pull` exits 1 with `the app is not pointed at Ollama: start with COMPOSE_PROFILES=gpu ...`; `up` exit 1; app not started |
| 3 environment | `OTHER_TOKEN`, fake `LLM_API_KEY` and `QDRANT_API_KEY` in `.env` | each occurs **0** times in `docker inspect` of every container, `docker compose config` and the logs; the app's configured environment holds only the settings it reads (empty unless set), `QDRANT_URL`, the two key files and `OLLAMA_URL`, besides the image's own variables |
| 3 Qdrant key | requests from the app container to `qdrant:6333` | `/collections` 401 without the key, 200 with it; `/readyz` 200 without; readiness `ok`, ingestion and a question work; Qdrant's configured environment holds only `QDRANT__TELEMETRY_DISABLED` |
| 4 `$` in a key | `LLM_API_KEY='sk-...$fake$KEY-...'` (23 characters) single-quoted, then unquoted | quoted: the secret file is 23 bytes and the fake server's header digest equals that of `Bearer <key>`; unquoted: 14 bytes, with compose warnings `The "fake" variable is not set` (KNOWN_LIMITATIONS P14) |
| 5 offline start | `ollama-pull` on the internal network only (no route out), real Ollama | model in the volume: exit 0, no download output; `LLM_MODEL=gemma4:no-such-tag`: exit 1, `lookup registry.ollama.ai ... server misbehaving` |
| 6 thinking | `--profile gpu` with `LLM_BASE_URL=http://10.213.47.250:11434/v1` set explicitly | Uvicorn's environment `LLM_REASONING_EFFORT=none`; the fake received `reasoning_effort=none` |
| 7, 9 image | hygiene service | 14 / 14 `ok` (new: no `.env` file under `/app`, no `uv` or `uvx`) |
| 9 networks | real Ollama | serving Ollama only on `backend` at `10.213.47.250`, no DNS for `registry.ollama.ai`, `ollama pull` fails there; `ollama-pull` only on `egress`; `ollama ps` 100 % GPU |
| 12 disk full | Qdrant on a 40 MB tmpfs volume, one ingestion | 503 `vector_store_unavailable`, detail `the vector store has no disk space left`, one ERROR line with that message; readiness 200 |
| 16 subnet clash | a network `10.213.47.0/24` created first | `up` exit 1: `networks have overlapping IPv4`; a second stack with `BACKEND_NETWORK_PREFIX=10.213.48` ran beside the first |

**Start times with the real Ollama** (`up -d --wait`, project `fx8g`): first start on the new volume (model present, no compiled kernels) 79.5 s, of which the warm-up prompt 63.6 s (85.5 s in 19.4); after `down` without `-v`, twice: 17.9 and 17.7 s, warm-up 3.1 and 2.7 s (15.8-19.4 s in 19.4). Without the profile: 9.9 s for one run on the fake stack's copy (13.4-13.6 s in 19.4; one run, not a comparison).

**Smoke suite** from the copy: without the profile **12 passed, 2 skipped** (answerable questions: no model); with `COMPOSE_PROFILES=gpu` and the real model **12 passed, 2 skipped** (the "without a model" variants), and a direct question answered `The research department produced 23 peer-reviewed publications in fiscal year 2024.`

---

## 20. Audit viewer in headless Chromium (`add-audit-viewer` tasks 5.1-6.2, design D3, D4, D7, D8)

Tests: `tests/e2e/` (browser suite, marker `e2e`, run by `docker-compose.e2e.yml`; the helpers `frames.py`, `ink.py` and `viewer.py`, and `test_fixtures.py`, which checks the helpers without a browser on the host and in the test image). Scripts: `benchmarks/viewer_e2e/` (local, not committed: `calib.py` and `perf.py` run in the `e2e` image on the stack's network, `analyze.py`, `render_diff.py` and `perf_table.py` on the host, `confcheck/conf.py` in the test image, `run.sh` the driver). Raw outputs (every page canvas pdf.js rendered, per-box records, crops of the failures, per-run timings, the suite's screenshots and responses) are in `benchmarks/viewer_e2e/out/` (local).

### 20.1 Method and environment

**Stack.** A copy of the repository without `.env` (`benchmarks/viewer_e2e/stack/`), project `v7e`, `BACKEND_NETWORK_PREFIX=10.213.71`; images built from that tree on 2026-10-29: `docintel:v7e` (runtime), `docintel-test:v7e` (test) and `docintel-test:v7e-e2e` (the `e2e` stage). No language model, no `.env`, every setting at its default. Host as in sections 17-19 (Windows 11, Docker Desktop 29.9.2 with Compose 5.6.1, Ryzen 7 7700X, WSL2 VM with 16 CPUs); nothing else ran during the timing runs.

**Browser.** Playwright 1.64.0 (sync API) with Chrome Headless Shell 156.0.8078.4 (`chromium-headless-shell` build 1248), `--disable-dev-shm-usage`, `shm_size: 1gb`, viewport 1280 x 900, `deviceScaleFactor` 1 (2 for one test), pdf.js 6.4.299 from the image. The browser container is only on the internal `backend` network, so it can reach nothing but the stack. Chromium cannot open `http://app:8000`: `app` is a top-level domain on its HSTS preload list, so the request goes out as HTTPS and fails with `ERR_SSL_PROTOCOL_ERROR`; `docker-compose.e2e.yml` gives the app the network alias `docintel-app` and the page is opened at `http://docintel-app:8000/audit` (KNOWN_LIMITATIONS V2).

**What a test reads.** The response JSON is captured from the network (`page.expect_response`), and every expected position is computed from it, not from the box elements' data attributes: a box element's rectangle relative to its page element must equal `(x, y, width, height) x data-scale` within 1 CSS pixel, its page index must be the response's `page_number`, and the overlay must sit at the page's 0,0; at zoom 2 the scale must be exactly 2, at `fit` it must make the widest page fill the pages column. The ink check (20.3) reads the page's `canvas.page-canvas` through `toDataURL`, converted to grey, and maps points to canvas pixels by the canvas size over the page's CSS size times the scale, so the device pixel ratio is accounted for. Every session records CSP violations (a `securitypolicyviolation` listener added by `add_init_script` before any page script), console errors, page errors, dialogs and every request (method and URL) of its browser context; `Session.check()` fails on any CSP violation, dialog or page error, any console error (only Chromium's `Failed to load resource` line is allowed, and only in tests that expect an HTTP error or a network failure), and any request other than `GET /audit`, `GET /audit/static/<file>`, `GET /audit/vendor/pdfjs/<file>` and `POST /api/extract` on the app's origin without a query (in-memory `blob:` and `data:` URLs aside).

### 20.2 Browser suite (task 5.1) and frame tests (task 5.2)

`docker compose -p v7e -f docker-compose.yml -f docker-compose.e2e.yml run --rm e2e` from the copy, image rebuilt from the final tree (tests not mounted): **35 passed** (the 8 helper tests are deselected by `-m e2e`; they run in the test image's suite and on the host), 25-27 s per run, two runs with the same result. `test_fixtures.py`: 8 passed on the host.

| module | tests | what they check |
|---|---|---|
| `test_frames.py` | 11 | frame PDF through the real service: every expected name found once with its box on its glyphs (IoU >= 0.5 with the box the fixture drew) and no name from outside a crop box; pdf.js page sizes equal the displayed sizes; matched / unmatched marks; position at `fit` and `2`; ink check at zoom 2; no out-of-page warning; at `deviceScaleFactor` 2 position at `fit` and `1.5` and ink at `1.5` (canvas twice the CSS size); the three sample PDFs: position at `fit` and `2` and ink |
| `test_behaviour.py` | 12 | response drawn with its request id and status; a name on page 3 selected in the list: box and item highlighted (`selected`, `aria-current`), details shown, box inside the viewport; filter `matched` / `unmatched` / `all`: visible boxes, list items and page headings; zoom `1`, `1.5`, `2`, `fit` keeps every box aligned; Tab from the last list button visits every box in page order, each focus shows its details, every `aria-label` names the name, page, match and score (or "no match") and the box in points; hover shows the details; the download equals the response body byte for byte and is named after the request id |
| `test_errors_and_safety.py` | 12 | a text file: 415 `not_a_pdf` shown with `data-code`, message and request id, no page drawn; 20.5 MiB: 413 `too_large`; a blank name the API rejects: `invalid_names`; three invalid names texts (broken JSON, an object, a single word) block the button and even a forced form submit sends nothing; 503 with a JSON body (route interception): code, message, request id and `HTTP 503`; 502 with an HTML body: `HTTP 502`, no code, its markup not rendered, download disabled; a network failure: "Network error"; a canned response whose names and match contain `<img onerror>`, `<script>` and `<svg onload>`: shown as literal text in the list, details and `aria-label`, no `img`, `svg`, `b` or `script` element created in the body, no dialog, no script run; a full audit with `scrollAllPages()` makes only the allowed requests and the page carries its CSP; control: an inline script injected into the page is refused and the listener records the `script-src` violation |

**Frame PDF** (`tests/e2e/frames.py`, built in the test with Pillow and pypdfium2, 390 KB, seven image-only pages at 150 DPI, the Aileron font at 20 pt). Each page is drawn upright, its pixels are turned into the page's user space and `/Rotate` turns them back, as a scanner that flags a sideways page produces it; the crop-box pages carry an image of the whole media box with a further name in the cropped-off margin. (The page agent's `benchmarks/viewer_dev/make_frames.py` had set `/Rotate` on upright content, so the displayed text was sideways or upside down and the service found no names on 180 / 270: that is KNOWN_LIMITATIONS O3, OCR without orientation detection, not a frame error.) Names list: the first name of every page plus a decoy, so each page has one matched and one unmatched box.

| page | frame | displayed size (pt) | names found | IoU with the drawn glyph box | ink: inside / ratio (zoom 2) |
|---|---|---|---|---|---|
| 0 | A4, `/Rotate 0` | 595.2 x 841.9 | Maria Gonzalez, John Smith | 0.96, 0.93 | 0.37 / 97, 0.35 / 68 |
| 1 | A4 scanned sideways, `/Rotate 90` (media box 841.9 x 595.2) | 595.2 x 841.9 | Hans Weber, Laura Chen | 0.96, 0.92 | 0.36 / 64, 0.33 / 39 |
| 2 | Letter upside down, `/Rotate 180` | 612 x 792 | Pedro Alvarez, Emma Brown | 0.96, 0.95 | 0.35 / 86, 0.38 / 223 |
| 3 | A4 scanned sideways, `/Rotate 270` | 595.2 x 841.9 | Lucas Martin, Sofia Rossi | 0.94, 0.95 | 0.38 / 77, 0.37 / 59 |
| 4 | Letter, crop box `[44.2 51.8 656.2 843.8]` in a 684 x 915.8 media box | 612 x 792 | David Miller, Anna Kowalski (not "Oliver Grant", outside the crop box) | 0.96, 0.94 | 0.38 / 61, 0.36 / 79 |
| 5 | A5 landscape | 595.2 x 420 | Peter Novak, Julia Fischer | 0.97, 0.99 | 0.37 / 77, 0.32 / 77 |
| 6 | A4 sideways with a crop box, `/Rotate 90`, crop `[72 44.2 913.9 639.4]` | 595.2 x 841.9 | Carlos Diaz, Nora Jensen (not "Victor Hale") | 0.94, 0.94 | 0.35 / 18, 0.38 / 25 |

All 14 expected names found, nothing else; every box within 1 px of points x scale at `fit`, `1`, `1.5` and `2` and at `deviceScaleFactor` 2; every box passes the ink check (lowest inside share 0.32, lowest ratio 18; at `deviceScaleFactor` 2 and zoom 1.5: 0.33 and 50); pdf.js reports the displayed sizes the fixture recorded, which are also PDFium's `get_size()` (`test_fixtures.py`). Sample PDFs: memo 8, minutes 23, report 24 boxes, all at position and passing (inside >= 0.39, ratio >= 11).

### 20.3 Ink check: calibration on the 60-document sample (task 6.1)

**Run.** The 60 documents of the seed-0 calibration sample (102 pages; 25 clean, 20 medium, 15 hard) through the viewer, one fresh browser context each, no names and no language hint: `POST /api/extract` 2.4-5.3 s per document (median 4.0 s). The position check at `fit` and `2` found **0** errors in 805 boxes, every session passed `check()`. At zoom 2 every page's canvas was saved (2 canvas pixels per point) and measured offline with `tests/e2e/ink.py`, so every variant below is measured on the same pixels. Each response box is labelled by the corpus truth: **truth-matched** when its IoU with a truth mention box (or the mention's box with its title) is at least 0.5, the corpus's criterion (770 boxes), else unmatched (35). Wrong-frame controls, on each truth-matched box: `pixels` (the box read as 200-DPI pixels taken for points, x 2.78), `yflip` (bottom-left origin taken for top-left), `down` (1.5 box heights lower, about one line), `right` (one box width plus 3 pt to the right, the next word).

**The metric** (design D7 item 2): share of ink pixels inside the box and in a ring of the same area around it (margin `m` with `(w + 2m)(h + 2m) = 2wh`, clamped to the page); a box passes when `inside >= T_in` and `inside / ring >= R`. Three definitions of "ink" were measured:

| ink definition | T_in, R by the rule below | truth-matched passing | `pixels` | `yflip` | `down` | `right` |
|---|---|---|---|---|---|---|
| darker than the midpoint between the page's median and its 0.1th percentile | 0.00, 1.5 | 99.4 % | 86.8 % | 56.2 % | 53.6 % | 76.2 % |
| 40 grey levels below the median of the box's neighbourhood (box, ring and one box height) | 0.16, 1.4 | 99.5 % | 1.6 % | 24.2 % | 36.5 % | 57.8 % |
| the same, without table rules (**adopted**) | **0.16, 2.1** | **99.1 %** | **0.9 %** | **20.3 %** | **31.4 %** | 56.5 % |

The page-wide level failed on degraded scans: black table rules, dark scan edges and salt-and-pepper noise set the page's darkest levels, so faint grey text (Courier lists on `hard` pages, 8-pt grey footers) did not count as ink and the inside share of correct boxes went down to 0.000; with that level no inside threshold separates anything. The local level fixed the inside share (truth-matched minimum 0.14), but on forms and tables the cell rules fell into the ring. Rules are now dark runs longer than the box plus one margin along a row, or longer than twice its height along a column, found after spreading the mask over 2 rows (the corpus pages are skewed by up to 2 degrees, so a rule's run in a single row is short) and bridging one-pixel gaps; with rules excluded the truth-matched ratio's 1st percentile rose from 1.54 to 2.16.

**Threshold rule.** The largest `T_in` (to 0.01) that keeps 99.5 % of the truth-matched boxes on the inside share alone (0.5th percentile 0.170 -> **0.16**), then the largest `R` (to 0.1) that keeps at least 99 % of them passing both (**2.1**). A grid of `T_in` 0.00-0.20 gives the same `R` at every value; 0.16 rejects most of the `pixels` control (2.6 % pass at `T_in` 0.02, 0.9 % at 0.16).

| population | n | inside p0.5 / p50 | ring p50 / p95 | ratio p1 / p50 | passing at 0.16 / 2.1 |
|---|---|---|---|---|---|
| response boxes, truth-matched | 770 | 0.170 / 0.407 | 0.006 / 0.046 | 2.16 / 66 | 763 (99.1 %) |
| response boxes, unmatched | 35 | 0.090 / 0.377 | 0.015 / 0.104 | 1.01 / 25 | 30 (85.7 %) |
| all response boxes | 805 | | | | **793 (98.5 %)**; clean 326 / 327, medium 335 / 338, hard 132 / 140 |
| corpus truth boxes (font-metric, 1.37 x the ink height) | 844 | 0.130 / 0.307 | 0.013 / 0.127 | 1.20 / 26 | 796 (94.3 %) |
| controls `pixels` / `yflip` / `down` / `right` | 770 each | | | | 7 (0.9 %) / 156 (20.3 %) / 242 (31.4 %) / 435 (56.5 %) |

The check catches a wrong unit or origin on almost every box of a page, and therefore every page of a wrong frame; it cannot tell a box from one moved onto a neighbouring word or line of the same text, which the position check and the frame tests with known glyph boxes cover (KNOWN_LIMITATIONS V1).

**Every failing box** (12 of 805), each inspected on its crop (`out/calib/fails_00.png`):

| class | n | boxes (document, page, name, IoU with truth, inside / ratio) |
|---|---|---|
| false-positive name, no name under the box | 3 | `en_040_kyc` p1 "VACANA WAV AN A" (0.00; 0.11 / 1.6) and `pt_089_minutes` p1 "LAY NAL" (0.00; 0.10 / 1.5): two signature scribbles read as letters and tagged as a person, the box spanning both signatures; `pt_070_minutes` p0 "Co Jrrao" (0.00; 0.09 / 10.0): noise in an empty shaded table header read as words |
| OCR box too wide or too tall | 3 | `es_021_memo` p1 "karina Ávila" (0.22; 0.15 / 1.0): a noise word "a" (Tesseract confidence 13) in the empty cell to the left joined the name, so the box spans the blank cell; `pt_098_kyc` p0 "Pereirada Costa" (0.40; 0.16 / 1.0): box reaching the row below, "Luiza" left out; `pt_098_kyc` p0 "Ale andra Garcia" (0.55; 0.17 / 1.0): box reaching the line above |
| correct box, neighbouring lines or accents in the ring of a dense `hard`/`medium` page | 6 | `en_045_kyc` p0 "Fernandes, Pedro Lucas" (0.73; 0.16 / 1.6), p1 "William McAllister" (0.68; 0.17 / 1.7) and "Ramos, Cindy" (0.85; 0.14 / 1.1): faint thin Courier list lines 1.3 box heights apart; `en_089_kyc` p0 "Sean Cook" (0.62; 0.19 / 1.9); `es_021_memo` p1 "Alfonso de la Cruz Soto" (0.73; 0.22 / 1.6): table rows above and below; `pt_089_minutes` p0 "INÊS CONCEIÇÃO" (0.95; 0.39 / 1.8): capital accents and cedilla of the line and the lines around it |
| name split across lines | 0 | |
| rendering difference between pdf.js and PDFium | 0 | see below |

**pdf.js against PDFium.** The 100 pages with boxes rendered by PDFium (the service's renderer) at the canvas's size, against pdf.js's canvas: mean absolute difference per page median 0.66, maximum 1.65 grey levels, no pixel differs by more than 32 levels on any page; the ink decision is the same for 804 of 805 boxes, the exception (`de_058_memo` "ANDRES FERNÁNDEZ GOMEZ", ratio 2.11 against 2.09) sits on the threshold (`render_diff.py`).

**Fixable causes (measured and partly fixed in section 21).** The three false positives and the too-wide box come from low-confidence OCR words: Tesseract's confidence under the boxes (the service's rendering, flattening and command, `confcheck/conf.py`) is "VACANA" 0, "LAY" 0, "Co" 18, "Jrrao" 9 and the stray "a" 13, while under 5 % of each page's words are below 50 (24 % on the `hard` `es_021_memo` page). The service keeps every word with a confidence of 0 or more (`app/adapters/ocr/tesseract.py`, `parse_tsv`); dropping words below a minimum confidence before NER, or not letting them start or extend a name span, would remove these four, but its cost in recall on real low-confidence names is unmeasured: it needs the paired name-recall / precision comparison on the 600-document corpus (sections 4.5 and 5.3) before adoption. Section 21 made it: dropping words costs recall, and the adopted rule (a name whose every word is below 20 is not reported) removes "Co Jrrao" but not the two signatures or the stray "a" (KNOWN_LIMITATIONS N9). The six correct boxes in class 3 are a limit of the ink measure, not of the service (KNOWN_LIMITATIONS V1).

### 20.4 Viewer performance in headless Chromium (task 6.2)

`perf.py`, three runs per size, each in a fresh browser context of one browser process; PDFs made of the sample's pages concatenated with pypdfium2 (1 page: 232 KB, 18 boxes; 10 pages: 2.5 MB, 85 boxes; 50 pages: 13.4 MB, 444 boxes); zoom `fit` (scale about 1.5) unless stated. Times from the page's own clock: navigation timing for DOMContentLoaded, the `ready` promise, the `/api/extract` resource timing (start to response end: the extraction, reported separately), then a `MutationObserver` for `body[data-state=done]` (every box element and list item is in the DOM) and the first `data-rendered="true"`. Memory: CDP `Performance.getMetrics` (JS heap of the page) and the PSS of every Chromium process from `/proc/<pid>/smaps_rollup` (after `scrollAllPages()`, and its peak sampled every 50 ms during it; the idle browser is 103 MB). Render time: for each of the first 10 pages, scrolled into view, the time from a zoom change to that page's `data-rendered="true"` at the new scale, to 100 % and to 200 %. Median (range) of the three runs.

| measure | 1 page | 10 pages | 50 pages |
|---|---|---|---|
| DOMContentLoaded (ms) | 23.1 (22.7-27.3) | 16.8 (15.6-24.3) | 24.1 (23.8-24.3) |
| viewer ready, pdf.js loaded (ms) | 46.9 (42.9-48.2) | 36.8 (34.3-43.6) | 44.8 (43.0-44.8) |
| **extraction**, `/api/extract` (ms, server and transfer) | 3,530 (3,519-3,549) | 8,438 (8,415-8,532) | 42,506 (42,170-42,799) |
| all boxes in the DOM, after the response (ms) | 3.8 (3.6-4.4) | 6.3 (6.2-6.4) | 21.1 (21.0-21.5) |
| first page rendered, after the response (ms) | 62.4 (51.7-66.0) | 62.7 (55.9-63.0) | 79.8 (77.5-80.3) |
| `scrollAllPages()` total (ms) | 0.1 (already rendered) | 360 (343-360) | 2,195 (2,178-2,209) |
| ... per page (ms) | | 36.0 (34.3-36.0) | 43.9 (43.6-44.2) |
| page rendered after a zoom change to 100 % (ms, median of 10 pages) | 25.2 (24.4-25.9) | 30.0 (29.3-33.1) | 32.0 (30.2-32.5) |
| ... to 200 % (ms) | 31.3 (31.2-31.3) | 43.9 (42.9-47.4) | 54.0 (50.4-58.1) |
| JS heap used / total after the scroll (MB) | 7.5 / 11.2 | 6.3 / 31.5 | 21.1 / 23.7 |
| Chromium PSS after the scroll (MB) | 280 (279-281) | 419 (419-427) | 541 (458-554) |
| Chromium PSS peak during the scroll (MB) | 272 (270-272) | 473 (443-481) | 554 (548-564) |
| pages holding a rendered canvas after the scroll | 1 | 4 | 4 |

Over all 63 page renders: 100 % median 30.2 ms (18.8-313.4), 200 % median 48.6 ms (29.1-65.1). The slow 100 % renders are always the same two pages (pages 6 and 9 of the concatenation: 106-144 and 180-313 ms) while at 200 % they take 55-65 ms; at 100 % more pages are near the viewport and render at the same time, which is the likely cause (not isolated). The viewer adds well under 0.1 s to a response that takes 3.5-43 s to compute; the lazy rendering keeps four canvases alive whatever the length, so the 50-page document costs about 120 MB more than the 10-page one (box elements, pdf.js's parsed document and the worker), not 40 more canvases. A zoom change re-renders the page in view and its near neighbours at once, so the per-page times include that concurrency.

### 20.5 Image sizes

`docker image inspect` sizes of the images built from this tree; the `e2e` stage before the change is `docintel-e2e:v7a`, built earlier the same day from the tree before it (full `playwright install --with-deps chromium`).

| image | size | difference |
|---|---|---|
| `docintel` (runtime) | 5,764,346,535 B (5.76 GB) | not changed by these tasks (the `e2e` stage is not in it); `docintel:v7a` from the earlier tree 5,764,324,123 B |
| `docintel-test` (test) | 5,753,166,624 B (5.75 GB) | |
| `e2e`, full Chromium and headless shell (before) | 7,412,150,086 B (7.41 GB) | +1.66 GB over the test image; layer 1.18 GB; `/ms-playwright` 681 MB (`chromium-1248` 403 MB, `chromium_headless_shell-1248` 273 MB, `ffmpeg` 5 MB) |
| `e2e`, `--only-shell` (after) | 6,801,559,338 B (6.80 GB) | +1.05 GB over the test image; layer 758 MB: `/ms-playwright` 278 MB, Playwright's Python package 137 MB, the system libraries and fonts of `--with-deps` the rest |
| `e2e`, headless shell from the `chromium` stage, sha256-pinned (review fixes, 20.7; adopted) | 6,800,474,090 B (6.80 GB) | +1.05 GB over the test image (`docintel-test:fx7` 5,753,256,021 B); `/ms-playwright` 285,084,594 B in its own 286 MB layer (the headless shell only: no ffmpeg, which only video recording uses), Playwright's Python package and the 73 apt packages of `playwright install-deps` 471 MB |

The headless shell is what Playwright's `chromium.launch(headless=True)` starts, so the full browser was never used by the tests; leaving it out saves 403 MB of browser and 0.61 GB of image. The runtime and test images contain neither Playwright nor a browser (hygiene check). The runtime image of the review fixes (`docintel:fx7`) is 5,764,357,319 B.

### 20.6 Findings and caveats

- **Frames:** with upright displayed text the service finds every name on `/Rotate` 90, 180 and 270 pages, on an offset crop box (alone and with `/Rotate 90`) and on mixed page sizes, with boxes at IoU 0.92-0.99 of the drawn glyphs, and pdf.js draws them on the glyphs. No frame error was found.
- **No page bug was found**; `app/web/` is unchanged by these tasks.
- **Ink-check limits:** it proves that a box sits on ink in the page's displayed frame, not that the ink is the name: 56.5 % of boxes moved onto the next word still pass (KNOWN_LIMITATIONS V1), and 0.9 % of the truth-matched boxes fail (7 of 770: 6 correct boxes on dense degraded pages, where neighbouring lines fall in the ring, and one OCR box reaching the line above).
- **False positives** from low-confidence OCR words (signatures, noise in shaded cells): measured in section 21; the adopted rule removes names read with a confidence below 20 throughout (14 spurious names on the corpus, "Co Jrrao" among the calibration cases); signatures with one word read above 20 remain (KNOWN_LIMITATIONS N9).
- One host, one browser build, synthetic corpus pages only; the timings are for a background-free machine and headless rendering (no GPU compositing), so a desktop browser will differ.
- **No hold-out set for the ink thresholds:** `T_in` 0.16 and `R` 2.1 were chosen on the same 770 truth-matched boxes of the 60-document sample whose pass share they produce (99.1 %), so that share is an in-sample figure. The pages not used for the choice keep a wide margin: the frame pages' lowest inside share and ratio were 0.32 and 18 (20.2; with the review's three new pages 0.29 and 18, 20.7) and the sample PDFs' 0.39 and 11, against thresholds of 0.16 and 2.1.

### 20.7 Review fixes (`add-audit-viewer` task 8.1)

The independent review's findings fixed in the viewer, its routes, its tests and the e2e image, and verified here. A copy of the tree without `.env` (`benchmarks/fx7/stack/`, local), projects `fx7t` (test suite) and `fx7s` (stack), `BACKEND_NETWORK_PREFIX=10.213.77`, `APP_PORT=18087`, no language model; images `docintel:fx7`, `docintel-test:fx7` and `docintel-test:fx7-e2e` built from it on 2026-10-29. The reviewer's probes (`benchmarks/review7/scripts`, local) were rerun against this stack (`benchmarks/fx7/review/out/probe`).

| finding | fix | verified by |
|---|---|---|
| `/UserUnit` (pdf.js applies it, PDFium ignores it) | pypdfium2 cannot read it (no page-dictionary call; every size and box call reports user units), so the service side is left as KNOWN_LIMITATIONS V4. The viewer renders each page at the zoom scale divided by `page.userUnit` (the service's frame) and shows a page audit warning | the reviewer's `userunit2.pdf`: ink 0.000 on both boxes before, 0.470 / 0.419 (ratios 118 / 12) after, with the warning; a `/UserUnit 2` page in the frame PDF (A5 landscape in user units): both names found (IoU 0.95, 0.97), at position, ink 0.378 / 0.351, exactly one page warning with the expected text, `data-user-unit` 2 |
| `/Rotate` not a number multiple of 90 | pdf.js passes a non-number through as `page.rotate`: the viewer flags it; `135` and `100` (pdf.js 0, PDFium 90) cannot be seen (KNOWN_LIMITATIONS V5) | e2e: a `/Rotate (90)` page is flagged (PDFium reads 0, pdf.js 90); the reviewer's `rotate135.pdf` still misses (ink 0.000, no warning), as recorded |
| no test of the 50-page scenario | e2e: the frame PDF six times over (60 pages, 120 names), canned response | every list item and box present at once (at most 8 pages rendered), `scrollAllPages()` rendered all 60 pages in 2.12-2.16 s (two runs), 4 canvases hold pixels afterwards; four extractions in one page: DOM nodes 1,895 and JS listeners 44 after every run (forced GC), JS heap 3.23, 3.30, 3.32, 3.34 MB |
| canvas size not checked | every rendered canvas's rectangle equals its page element's within 1 px, at zoom 2, at `deviceScaleFactor` 2 (zoom 1.5) and on the sample PDFs | e2e ink tests |
| JPX and CCITT never exercised | a JPEG 2000 page and a CCITT G4 page in the frame PDF | names found (IoU 0.94-0.97), ink 0.352 / 0.359 (JPX) and 0.289 / 0.353 (CCITT; bitonal, the lowest of the suite), `openjpeg_nowasm_fallback.js` and `jbig2_nowasm_fallback.js` requested, no CSP violation |
| the 250 ms timer announced | the live status is set at start, end, error and cancel only; the time ticks in `#elapsed`, outside any live region; the names' problems are shown at once and announced in a visually hidden polite region 1 s after typing stops, when they change | e2e: during a 1.3 s held request the status region changed twice (start, end) while `#elapsed` ticked; typing "Madonna" announced nothing until the pause, then once; completing the name announced "Names: no problems" |
| inputs live while loading, no cancel | file, names and language disabled while loading, the result line names the file, Cancel (`AbortController`) returns to idle with "Cancelled after ..." | e2e: all four controls disabled during a held request, the result names `first.pdf` after a second file was set; cancel: idle, no page or box, controls enabled, a late answer ignored, a new extraction works |
| CSP `img-src 'self' blob: data:` | `img-src 'self'` | the reviewer's strict-CSP probe (frames, memo, JPX, CCITT) and text probe (standard and embedded fonts): no violation; the whole e2e suite: no violation |
| no HEAD, `/audit/` redirect built from `Host` | GET and HEAD on every viewer route; the page also at `/audit/`; any other `/audit/...` path a 404 with the viewer's headers | `tests/viewer/test_routes.py`: HEAD has GET's headers and no body; with `Host: attacker.example`, `/audit/`, `/audit/static`, `/audit/vendor/pdfjs` and others get no `Location`. `redirect_slashes` stays on: it also serves the API (`POST /api/extract/` is redirected), outside the viewer |
| no conditional requests | `If-None-Match` (strong, weak, lists, `*`) answered 304 with the same headers on every viewer file | route tests; e2e: a reload gets 304 for every `/audit/static/` file and 200 for the `no-store` page |
| 422 `detail` and `request_id` of unexpected types | `Array.isArray` for `detail` and `loc`, text only from strings (other values as JSON), `request_id` only a string or finite number | e2e, canned 422 with mixed entries and a numeric id: "invalid_names: 5; plain text; body.3.names: bad", id 12345, download `extract-12345.json`; canned 500 with an object `detail` and id: "HTTP 500", no id shown, download `extract-response.json`; no page error |
| `deploy/pdfjs.lock` parsing, download deadline | values limited to letters, digits and `. _ : / -`; unknown or repeated names refused; a 300 s deadline over the whole download (read in pieces with `read1`); `http.client` errors reported cleanly | `tests/viewer/test_fetch_pdfjs.py` (40 tests) |
| e2e browser not hash-checked | the `chromium` stage downloads Chrome Headless Shell 156.0.8078.4 (`chrome-headless-shell-linux64.zip`, 125,797,539 B, sha256 `c3e50596...ba20`) from Playwright's own CDN path, checks it and unpacks it into Playwright's layout; the `e2e` stage runs only `playwright install-deps` and one launch | the archive's executable has the sha256 of the one `playwright install` had written (`4fa61f0b...f9d6`), 285,084,594 B in both; the build's launch check passed; e2e `test_the_installed_browser_is_the_one_this_playwright_pins` |

**Suites on this stack.** Browser suite 48 passed (35 before; added: 3 frame tests, 2 response-shape tests, 1 reload test, 2 long-document tests, 4 loading tests, 1 browser-pin test), 48.5 s on the final images (the same on the first build); smoke 12 passed, 2 skipped (no language model); hygiene all `ok`. The frame PDF is now ten pages (465 KB) with 20 names, all found, lowest ink inside share 0.289 and ratio 18.3 at zoom 2 (0.291 / 50.1 at `deviceScaleFactor` 2); samples 0.386 / 11.0.

---

## 21. Low-confidence OCR names (`add-audit-viewer` task 9, design D9)

**Question.** The viewer calibration (20.3) traced three false-positive names and one too-wide box to words Tesseract read with a low confidence (signature scribbles and noise in a shaded cell read as letters). Can the service use Tesseract's word confidence to drop them without losing real names?

**Method.** The 600-document corpus (1,045 pages, 7,489 truth mentions). Each document read once by the production OCR engine (`TesseractOcrEngine`, default configuration, true-language hint as in 4.5; 6 documents at once, 2 Tesseract processes each), keeping the TSV of every page; the production `ExtractNames` with the real XLM-R recogniser then ran on each variant of the same pages. Scoring with the benchmark's own `score_doc` (IoU >= 0.5, greedy one-to-one, trap rule; 4.5), paired by document against the unchanged chain (T = 0, which drops nothing: `parse_tsv` already drops confidences below 0), bootstrap over documents, 1,000 resamples, seed 0. `*` marks a CI that excludes 0. Persons (box) = distinct persons with at least one correctly boxed mention. The unchanged chain is the benchmark default within noise: dF1 -0.2 [-0.4, +0.1] against `stageB` OCR words on the same 600 documents.

**A. Dropping low-confidence words before recognition** (the words below T removed from the page, so from the recogniser's text and from every later step):

| T | words kept | P | R | F1 | dR | dF1 | FP trap / boundary / spurious | d persons (box) |
|---|---|---|---|---|---|---|---|---|
| 0 (unchanged) | 100.0 % | 93.9 [93.0-94.7] | 89.9 [88.8-90.9] | 91.8 [91.0-92.6] | | | 194 / 99 / 147 | |
| 10 | 99.0 % | 94.1 | 89.1 | 91.5 | -0.8 [-1.1, -0.4] * | -0.3 [-0.6, -0.0] * | 189 / 104 / 127 | -0.3 [-0.6, +0.1] |
| 20 | 98.6 % | 94.0 | 88.8 | 91.3 | -1.1 [-1.5, -0.7] * | -0.5 [-0.8, -0.2] * | 189 / 110 / 125 | -0.6 [-0.9, -0.1] * |
| 30 | 98.0 % | 94.0 | 88.1 | 90.9 | -1.8 [-2.3, -1.3] * | -0.9 [-1.3, -0.5] * | 185 / 116 / 119 | -0.9 [-1.4, -0.5] * |
| 40 | 97.3 % | 94.1 | 87.4 | 90.6 | -2.5 [-3.1, -1.9] * | -1.2 [-1.7, -0.8] * | 187 / 124 / 101 | -1.7 [-2.3, -1.2] * |
| 60 | 95.5 % | 93.6 | 84.9 | 89.0 | -5.0 [-5.8, -4.1] * | -2.8 [-3.4, -2.2] * | 183 / 152 / 101 | -3.2 [-4.0, -2.5] * |

It removes the four calibration cases at every T, but costs recall from T = 10: removing a word also changes the recogniser's context and the line structure around it (93 true positives lost at T = 10, although only 0.7 % of the true-positive mentions contain a word below 10). **Not adopted.**

**B. Rules on candidate names, text unchanged** (simulated on the T = 0 output: a mention's words are those whose centre lies in its box):

| rule | dP | dR | dF1 | spurious FP | d persons (box) | worst group dF1 |
|---|---|---|---|---|---|---|
| drop when every word < 10 | +0.1 [+0.1, +0.2] * | -0.0 [-0.1, +0.0] | +0.0 [+0.0, +0.1] * | 137 | +0.0 | hard -0.0 [-0.2, +0.1] |
| **drop when every word < 20** | **+0.2 [+0.1, +0.2] \*** | **-0.0 [-0.1, +0.0]** | **+0.1 [+0.0, +0.1] \*** | **135** | **+0.0** | hard -0.0 [-0.2, +0.1] |
| drop when every word < 30 | +0.2 [+0.1, +0.3] * | -0.1 [-0.1, -0.0] * | +0.1 [+0.0, +0.1] * | 130 | -0.0 [-0.1, +0.0] | hard -0.1 [-0.3, +0.1] |
| drop when every word < 40 | +0.3 [+0.2, +0.4] * | -0.1 [-0.2, -0.1] * | +0.1 [+0.0, +0.1] * | 127 | -0.0 [-0.1, +0.0] | es -0.1 [-0.3, +0.0] |
| drop when the mean < 20 | +0.3 [+0.2, +0.4] * | -0.1 [-0.1, -0.0] * | +0.1 [+0.0, +0.2] * | 128 | +0.0 | hard -0.0 [-0.2, +0.2] |
| drop when the mean < 30 | +0.4 [+0.2, +0.5] * | -0.2 [-0.3, -0.1] * | +0.1 [-0.0, +0.1] | 122 | -0.2 [-0.3, -0.1] * | hard -0.4 [-0.7, -0.0] * |
| trim leading / trailing words < 10 | -0.4 [-0.6, -0.2] * | -0.5 [-0.7, -0.3] * | -0.5 [-0.7, -0.3] * | 137 | -0.2 [-0.4, -0.1] * | hard -2.3 [-3.3, -1.4] * |
| trim leading / trailing one-letter words < 20 | -0.0 [-0.1, +0.0] | -0.0 [-0.1, +0.0] | -0.0 [-0.1, +0.0] | 147 | | |

Trimming edge words cuts real names, whose accented or degraded words are often read with a low confidence (the hard level loses most); trimming only one-letter words changes nothing measurable. The largest threshold whose recall CI includes 0 is 20 for the "every word" rule; at 20 it removes 12 spurious names and 3 true positives ("Grof", "Figueiredo", "Vasconcelos", each a person boxed elsewhere in its document) in the simulation. **Adopted:** Dconf, `MIN_NAME_CONFIDENCE = 20` (`app/domain/names.py`), the last span rule; words now carry their confidence (`PageWord.confidence`, from `parse_tsv`).

**C. End to end with the rule** (`docintel-test:conf`, built from a copy of the tree without `.env`; the production chain on the 600 documents, paired against the unchanged chain of A):

| run | P | R | F1 | trapFP | FP trap / boundary / spurious | persons (box) | persons (full name) |
|---|---|---|---|---|---|---|---|
| unchanged chain | 93.9 [93.0-94.7] | 89.9 [88.8-90.9] | 91.8 [91.0-92.6] | 7.3 [5.9-8.8] | 194 / 99 / 147 | 93.7 [92.4-94.8] | 88.1 [86.6-89.5] |
| **with Dconf** | **94.0 [93.1-94.9]** | **89.8 [88.7-90.9]** | **91.9 [91.1-92.6]** | 7.3 [5.9-8.8] | 194 / 99 / **133** | 93.7 [92.4-94.8] | 88.1 [86.6-89.5] |

Paired: dP **+0.2 [+0.1, +0.3] \***, dR -0.0 [-0.1, +0.0], dF1 +0.1 [+0.0, +0.1] *, trapFP +0.0 [+0.0, +0.0], persons (box, full name and text) +0.0 [+0.0, +0.0]. By language dF1 en +0.1 [+0.0, +0.2], es +0.0, pt +0.1 [-0.1, +0.3], fr +0.1 [+0.0, +0.2], de +0.0 [+0.0, +0.1]; by noise clean +0.1 [+0.0, +0.1] *, medium +0.1 [+0.0, +0.2] *, hard +0.0 [-0.2, +0.2]; every recall CI includes 0. 7,153 mentions are identical (page, name and box), 17 are removed and none is added: 14 spurious names (signature scribbles such as "PIV WVIY", "LAVAVES TIVAVAN", "VAVAVA YNAV", noise such as "Co Jrrao" and "BERNER EIHENHEIEBEEEEREENEN Bun", and "Some") and the 3 true positives of B. The rule is a pass over the words of each candidate: no measurable cost.

**Calibration cases of 20.3.** "Co Jrrao" (confidences 18 and 9) is removed. The two signatures keep one readable-looking word ("VACANA WAV AN A": 0, 63, 60, 69; "LAY NAL": 0, 60), and "karina Ávila" with the stray "a" (23, 48, 13) has a word above 20; a rule that removes them costs recall (A, and the mean and trim rows of B): KNOWN_LIMITATIONS N9.

**Reproduce:** `benchmarks/ocr_conf/` (local): `run_conf.py` (A, and the T = 0 output for B), `score_conf.py`, `sim_guard.py` (B), `run_verify.py` and `score_verify.py` (C); results in `out/report.md`, `out/sim_guard.md` and `out/verify_report.md`.

---

## 22. Defensive hardening (`add-defensive-hardening` task 7.2, design D10)

Scripts: `benchmarks/adversarial/` (local, like every benchmark script; `benchmarks/README.md` step 14). Raw outputs in `benchmarks/adversarial/out/` (local, not committed):
- `llm/`: the quality run, the offline re-score, `analysis_*.{md,json}`.
- `detector/`: the scores of every text and `detector_tables.md`.
- `cost/`: the ingestion cells; `cost/image_without_patterns/` and `cost/invalid_429/` hold runs set aside (22.5).

### 22.1 Method and environment

**Question.** The change adds four things to the answer path:
- **D8:** delimited passages (`<<<passage n: document <id>, page <p>>>>` ... `<<<end passage n>>>`) and the data-not-instructions rules in the system prompt.
- **D7:** question screening by the injection detector.
- **D9:** the support check (`RAG_SUPPORT_MIN_SHARE` 0.6), the output filter and the per-document chunk cap (`RAG_MAX_CHUNKS_PER_DOCUMENT` 2).
- **D5, D5b:** the detector, Prompt Guard 2 86M and the instruction patterns, also screening pages at ingestion.

The acceptance rules come from the design:
- **D8:** the paired accuracy difference against the current prompt must include 0 in its 95 % interval.
- **D9 and D10:** `RAG_SUPPORT_MIN_SHARE` must keep at least 99 % of correct answers, and the support check and output filter together must refuse fewer than 1 % of them.
- **D10:** the detector's precision and recall, and its cost per page and per question.

**Host.** RTX 4070 Ti 12 GB (driver 617.14), 16 logical CPUs, Docker Desktop 29.9.2, Compose 5.6.1. No other compose project ran during these measurements. Host CPU load was 4-34 % from desktop applications before the runs.

**Images.** Two test images were built from the working tree with `docker compose -f docker-compose.test.yml build tests`, both carrying the detector model at `/models/prompt-guard`:
- `docintel-test:hard` (2026-10-29 15:59 UTC, `sha256:2a8b2da9...`): commit `f37f774` plus the change's edits at that time. It holds the delimited prompt, the detector model, the support check, the output filter and the chunk cap. **It does not hold the instruction patterns (D5b)**, which reached the tree at 16:07-16:14 UTC. **The quality run of 22.2 used this image.**
- `docintel-test:hard2` (17:27 UTC, tree sha256 `b0defcb4...`): adds the patterns (`CompositeDetector([PatternDetector(), PromptGuardDetector(...)])`, `INJECTION_PATTERNS` default true) and the corrected support rule of 22.3. **The cost cells of 22.5 used this image.**

**Quality run.** As 16.1 and 16.4, with only the tree and the detector changed against the recorded `e4b_nt` run:
- The `llm_eval` harness with its pass-through recording proxy, compose project `llmeval_hard`.
- The 600-document index of section 16 (external volume `llmeval_qdrant`, 3,299 chunks, read only; no chunk is flagged). The first question of the fixed order got the same top-1 score, 0.8494.
- The same 750-question sample (600 answerable, 150 unanswerable), in the same fixed order, after 3 uncounted warm-up questions.
- `gemma4:e4b-it-qat` on Ollama 0.40.12 with thinking off (`--rewrite reasoning_none`), `OLLAMA_CONTEXT_LENGTH=8192`. No hosted model was called.
- Default settings, detector threshold 0.5.
- The same metrics and bootstrap (2,000 resamples, seed 0, paired per question).

**Offline re-score.** The support rule was then corrected in the tree: the share is now computed against `render_context(pages)`, the context exactly as the model saw it, labels included. `rescore.py` re-ran the tree's `is_refusal`, `support_share` and `output_filter` on every recorded reply and its recorded context. No model was called: the model calls are identical, and only the post-processing differs. Questions refused before the model call (threshold, flagged) are copied unchanged. The figures below are the re-scored ones unless marked "as run". The share the service logged in the run equalled the host recomputation of the old rule in 576 of 576 replies.

**Paired views.** Two views are paired against section 16's `e4b_nt`:
- **The service's answer:** what the client gets, after the guardrails.
- **The model's raw reply:** taken from the proxy recording, before the support check and the filter. It isolates the prompt wording.

### 22.2 Answer quality

| run | accuracy (containment) | token-F1 | accuracy, gold in context (n) | wrong refusals: all / threshold / model / unsupported / filtered | correct refusals: all / threshold / model / unsupported | language mismatch (undetermined of answered) |
|---|---|---|---|---|---|---|
| `e4b_nt`, section 16 (labels, no guardrails) | **72.3 (68.7-76.0)** | 27.5 (25.4-29.7) | 95.5 (93.5-97.3) (448) | 10.5 / 3.5 / 7.0 / 0.0 / 0.0 | 72.0 / 46.7 / 25.3 / 0.0 | 0.3 (0.0-1.0) (226 of 537) |
| `hard_e4b_nt` as run (old support rule: page texts only) | 69.3 (65.7-73.0) | 26.4 (24.4-28.5) | 92.1 (89.4-94.4) (445) | 14.8 / 3.5 / 7.5 / 3.8 / 0.0 | 74.0 / 46.7 / 25.3 / 2.0 | 0.0 (216 of 511) |
| **`hard_e4b_nt` re-scored (tree's rule: rendered context)** | **71.0 (67.3-74.7)** | 26.7 (24.6-28.8) | 94.4 (92.1-96.4) (445) | 11.8 (9.3-14.5) / 3.5 / 7.5 / 0.8 / 0.0 | 73.3 (66.0-80.0) / 46.7 / 25.3 / 1.3 | 0.0 (217 of 529) |

Every request returned 200 (0 errors in 750).
- **Accuracy by language** (en / es / pt / fr / de): 72 / 70 / 77 / 76 / 66 in section 16, 69 / 68 / 77 / 76 / 64 re-scored.
- **Refused by category:** absent entity 100 %, off-topic 100 %, cross-document 16 % before and 20 % after.

**Refusal reasons, re-scored**, over the 750 questions (answerable + unanswerable):

| reason | answerable | unanswerable |
|---|---|---|
| `below_threshold` | 21 | 70 |
| `model` | 45 | 38 |
| `unsupported` | 5 | 2 |
| `output_filtered` | 0 | 0 |
| `question_flagged` | 0 | 0 |
| `no_context` | 0 | 0 |

The threshold refusals are identical to section 16, since retrieval is unchanged. In the run as made, `unsupported` was 23 + 3; the re-score answers 19 of those 26 replies and keeps 7 refused.

**Paired differences against `e4b_nt`** (points):

| view | measure | n | difference, 95 % interval |
|---|---|---|---|
| **service, re-scored** | **accuracy** | 600 | **-1.3 [-2.8, +0.0]** |
| service, re-scored | wrong refusals | 600 | +1.3 [+0.3, +2.5] |
| service, re-scored | correct refusals | 150 | +1.3 [+0.0, +3.3] |
| service, re-scored | token-F1 | 600 | -0.8 [-1.8, +0.2] |
| service, as run | accuracy | 600 | -3.0 [-4.7, -1.3] |
| raw reply | accuracy, every question that called the model | 579 | -1.0 [-2.4, +0.3] |
| **raw reply** | **accuracy, same context pages in both runs** | 498 | **-0.8 [-2.0, +0.4]** |
| raw reply | accuracy, context changed by the chunk cap | 81 | -2.5 [-8.6, +3.7] |
| raw reply | model refusals, same context | 498 | -0.4 [-1.0, +0.0] |

Re-scored transitions (old, new): 421 right in both, 161 wrong in both, 13 lost, 5 gained.
- **Lost (13):** 7 had the same context and a different reply (temperature 0 is not deterministic: the side probe reproduced the recorded reply in 86 of 100 replays), 4 had a context changed by the chunk cap, and 2 were refused by the support check.
- **Gained (5):** 3 same context, 2 changed context.
- By language the re-scored difference lies between -2.5 and +0.0 points, with every interval including 0 or touching it.

**Chunk cap.** With `RAG_MAX_CHUNKS_PER_DOCUMENT` 2 the context changed for 83 of the 659 questions that called the model (81 of them answerable):
- 64 got more pages: a skipped third chunk of the top document leaves room for another document's page.
- 3 got fewer; 16 got a different page set of the same size.

Gold-in-context fell from 448 to 445. On the 81 answerable questions, model refusals rose (+6.2 [+1.2, +11.1]): with a second document's page the model hedges more often on this near-duplicate corpus (section 1). The accuracy effect is within the noise.

### 22.3 Support share and `RAG_SUPPORT_MIN_SHARE` (D9)

The set is the model's raw replies that contain the gold answer: 428 replies, every one with content tokens. Each line is one choice of the context the answer's tokens are checked against:

| known set (what the answer's tokens are checked against) | share p1 / p5 / p10 / p25 / p50 | minimum | below 0.5 / 0.6 / 0.7 / 0.8 | kept at 0.6 | largest threshold (step 0.05) keeping >= 99 % |
|---|---|---|---|---|---|
| page texts only (the rule of the run) | 0.452 / 0.705 / 0.750 / 0.889 / 1.000 | 0.349 | 7 / 12 / 21 / 49 | 97.2 % | 0.40 |
| **rendered context, labels included (the tree's rule)** | **0.649 / 0.750 / 0.820 / 0.900 / 1.000** | 0.500 | 0 / 2 / 7 / 35 | **99.5 %** | **0.60** (0.65: 98.8 %) |
| rendered context + the question | 0.698 / 0.868 / 0.899 / 1.000 / 1.000 | 0.643 | 0 / 0 / 5 / 12 | 100.0 % | 0.65 |

**False-refusal rate**, the correct raw replies that the support check or the output filter refused:
- **Tree's rule: 2 of 428 = 0.47 % (0.00-1.17)**, both `unsupported`. Both are short answers that echo the question's words: `Der Mehrwertsteuersatz beträgt 21,0%` (share 0.50) and a two-date reply (0.54).
- **Rule of the run: 12 of 428 = 2.80 % (1.40-4.44).** In 10 of these 12 the model had copied the passage label into a correct answer, and the UUID's hex segments counted as unsupported tokens. 20 of the 576 answered replies copy a label.
- The output filter refused none of the 659 legitimate model calls under either rule.

**Result: `RAG_SUPPORT_MIN_SHARE=0.6` is kept.** With the share taken against the rendered context, 0.6 keeps 99.5 % of correct answers and refuses 0.47 % of them, which meets D9's 99 % and D10's 1 %. The next step, 0.65, would keep only 98.8 %. Counting the question's tokens as known would give 100 %, but it would let an answer that only restates the question through; that was not adopted.

### 22.4 Injection detector: precision and recall (D5, D5b, D7)

**Method** (`detector_eval.py`, on the host CPU with the project's venv: torch 2.14.1+cpu, transformers 5.21.0, 8 threads). The model is the vendor copy of Prompt Guard 2 86M used for the image. Three detectors, as the service builds them:
- **model:** `PromptGuardDetector`, 512-token windows with a stride of 128, keeping the highest window score.
- **patterns:** `PatternDetector`, scoring 1.0 or 0.
- **combined:** `CompositeDetector([PatternDetector(), PromptGuardDetector(...)])`, keeping the higher score per text.

**Sets.**
- **Attack sets**, from `tests/adversarial/cases.py`, 40 cases each (8 attack classes x 5 languages):
  - `DIRECT`: questions sent to the service.
  - `DIRECT` with the suite's OCR-noise model applied (`noisy()`).
  - `INDIRECT`: sentences an attacker would put in a document.
  - `INDIRECT_NOISY`: the OCR-noisy variants of the indirect cases.
- **Benign controls** of the suite: 30 sentences and 15 questions.
- **Benign corpus sets.** The synthetic corpus holds no injections:
  - every page text of the 600-document index (1,030 pages, mean 1,771 characters, the OCR text the service stored; `dump_pages.py`), which is what ingestion screens;
  - every chunk (3,299, mean 616 characters);
  - the 750 sample questions of section 16, which is what question screening sees.

**The pattern rates on the suite are a floor check only.** The patterns were written against these cases, so their 100 % shows that nothing regressed, not that they recall attacks in general.

**Flag rate by threshold** (flagged / n, %; recall on the attack sets, false-positive rate on the benign sets):

| detector | set | n | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|---|---|
| model | direct | 40 | 75.0 | 75.0 | **75.0** | 72.5 | 72.5 |
| model | direct, OCR-noisy | 40 | 62.5 | 62.5 | **57.5** | 55.0 | 45.0 |
| model | indirect | 40 | 40.0 | 37.5 | **35.0** | 35.0 | 35.0 |
| model | indirect, OCR-noisy | 40 | 40.0 | 32.5 | **32.5** | 32.5 | 32.5 |
| model | corpus pages (benign) | 1,030 | 1.36 | 0.39 | **0.29** | 0.19 | 0.10 |
| model | corpus chunks (benign) | 3,299 | 0.39 | 0.21 | **0.15** | 0.06 | 0.03 |
| patterns | the four attack sets | 4 x 40 | 100 | 100 | 100 | 100 | 100 |
| patterns | corpus pages / chunks (benign) | 1,030 / 3,299 | 0.10 / 0.03 | 0.10 / 0.03 | 0.10 / 0.03 | 0.10 / 0.03 | 0.10 / 0.03 |
| **combined** | **the four attack sets** | 4 x 40 | 100 | 100 | **100** | 100 | 100 |
| **combined** | **corpus pages (benign)** | 1,030 | 1.46 | 0.49 | **0.39** | 0.29 | 0.19 |
| **combined** | **corpus chunks (benign)** | 3,299 | 0.42 | 0.24 | **0.18** | 0.09 | 0.06 |
| all three | suite benign sentences and questions; the 750 sample questions | 30 + 15 + 750 | 0 | 0 | 0 | 0 | 0 |

**The model alone.** At 0.5 it recalls the same share in every language: 6 of 8 direct cases per language, 2-3 of 8 indirect.

By attack class, at 0.5, over 5 cases each (direct / indirect):

| attack class | recalled (direct / indirect) |
|---|---|
| ignore-previous | 5 / 5 |
| forged delimiter | 5 / 4 |
| look-alike delimiter | 5 / 5 |
| role override | 5 / 0 |
| polite phrasing | 5 / 0 |
| spacing or case obfuscation | 5 / 0 |
| prompt exfiltration | 0 / 0 |
| answer change ("say the contract is void") | 0 / 0 |

The OCR noise costs it a further 17.5 points on direct cases. Its scores on the sample questions reach at most 0.0026, and on the suite's benign texts at most 0.0007.

**Benign pages flagged at 0.5** (combined: 4 of 1,030 pages; none in the 60-document sample of 22.5):
- One invoice page that the patterns match on an OCR artefact (a run of `>»` among prices read as a delimiter).
- Three pages the model scores 0.54-1.00: a page of 29 garbled characters of a German invoice, a French KYC signature block, and an English minutes header full of OCR noise.

These are OCR noise rather than instructions. A flagged page's chunks are kept but excluded from search (D6).

**Threshold.** 0.5 is kept. Lowering it to 0.3 adds one indirect case to the model's recall (35.0 to 37.5 %; the combined detector is at 100 % either way) and adds benign pages (0.39 to 0.49 %). Raising it to 0.7 or 0.9 loses model recall on the noisy direct cases (57.5 to 55.0 and 45.0 %).

### 22.5 Cost per page and per question

**Method** (`cost_run.sh`, `ingest_cost.py`):
- Image `docintel-test:hard2` with its default settings, on CPU. The detector runs with its default pool of 2 and `TORCH_THREADS` unset.
- The 60-document sample: 58 documents and 98 pages counted, since the 2 warm-up documents come back already indexed and are excluded.
- Ingested through `POST /api/ingest` of a real Uvicorn against the throwaway Qdrant of `docker-compose.test.yml`, a fresh index per cell. Compose project `advcost`, torn down with `down -v --remove-orphans` after every cell.
- The detector is configured through the container environment, since the image's `INJECTION_MODEL` build argument cannot change at run time:
  - model + patterns: the image's default;
  - model only: `INJECTION_PATTERNS=false`;
  - patterns only: `INJECTION_MODEL_PATH` unset;
  - none: both off.
- Each server's start-up lines confirm its configuration. `screen_ms` per document comes from the service's `ingestion completed` lines.

**Benchmark-only overrides.** The 12-client cells set `MAX_JOBS_PER_CLIENT=0` and `RATE_LIMIT_UPLOADS_PER_MINUTE=0`, because the D1 limits key on the peer address and one benchmark client is one address. In a first attempt, 56 of 60 requests were answered 429 by the job cap of 4 (`cost/invalid_429/`). The sequential cells never reached either limit; they returned 60 x 200 with the defaults.

| detector | clients | docs/min | pages/min | `screen_ms` per page p50 / p95 / mean | `screen_ms` per document p50 / p95 | ingestion `duration_ms` per document p50 / p95 | throughput against "none" |
|---|---|---|---|---|---|---|---|
| model + patterns | 1 | **9.85** | 16.6 | 1,472 / 1,929 / 1,358 | 2,882 / 3,859 | 6,653 / 8,466 | **-38 %** |
| none | 1 | 15.78 | 26.7 | 0 | 0 | 3,777 / 5,247 | - |
| model + patterns | 12 | **36.4** | 61.5 | 7,990 / 15,660 / 7,838 | 14,724 / 19,314 | 19,867 / 24,947 | **-59 %** |
| model only | 12 | 37.1 | 62.7 | 7,860 / 15,291 / 7,605 | 14,363 / 17,642 | 19,641 / 23,743 | -58 % |
| patterns only | 12 | 87.9 | 148.6 | 1.8 / 5.1 / 3.4 | 2.7 / 9.7 | 7,756 / 9,640 | +0 % |
| none | 12 | 87.8 | 148.3 | 0 | 0 | 7,580 / 9,951 | - |

- **The model is the whole cost.** Sequentially it adds about 1.36 s of CPU per page to an ingestion whose OCR takes about 2.7 s per document (`ocr_ms` p50 2,735-2,759 in both sequential cells).
- **Under load it becomes the bottleneck.** With 12 documents in flight, the per-page `screen_ms` (7.8 s mean) is mostly waiting for the detector's 2 slots, and throughput falls from 87.8 to 36.4 docs/min.
- **The patterns cost 1.8 ms per page** (p50) and nothing measurable in throughput.
- The host measurement of 22.4 agrees: 613 ms per page for the model with 8 threads.

**Question side.** Screening one question with the model takes about **130 ms** in the service: the time outside embedding, search and the model call rose from 0.1 ms to 129.9 ms p50, 156.6 ms p95, in the quality run on `docintel-test:hard`. On the host with 8 threads the model takes 18.7 ms per question and the patterns 0.07 ms. The service's median request (0.70 s against 0.52 s in section 16) includes this screening and the 9 % longer prompt (22.6).

**Runs set aside.** The first cost cells, in `cost/image_without_patterns/`, ran on `docintel-test:hard`, which lacks the patterns. There, "model + patterns" was the model alone and "patterns only" had no detector at all. Their model figures agree with `hard2`: 9.95 docs/min sequential, 37.4-38.3 with 12 clients.

### 22.6 Latency, tokens and memory of the quality run

| run | requests p50 / p95 (s) | with a model call p50 / p95 / p99 / max (s) | `llm_ms` p50 / p95 (s) | generation / prefill tokens/s (p50) | prompt tokens: total / mean / max | completion tokens: total / p50 | VRAM, device max (GB) |
|---|---|---|---|---|---|---|---|
| `e4b_nt` | 0.52 / 1.74 | 0.56 / 1.79 / 2.26 / 6.57 | 0.50 / 1.74 | 113 / 7,183 | 983,283 / 1,493 / 2,661 | 26,408 / 27 | 5.7 |
| `hard_e4b_nt` | 0.70 / 1.43 | 0.74 / 1.51 / 2.50 / 6.77 | 0.54 / 1.34 | 104 / 6,855 | 1,074,154 / 1,631 / 2,763 | 22,221 / 27 | 6.1 |

- **Prompt length.** The system prompt grew from 116 to 188 tokens, and every passage adds two delimiter lines, so the mean prompt is 9 % longer.
- **Context length.** The constructed worst case (the probe's densest pages packed by `select_context`, the longest question) is 2,647 prompt tokens, against 1,973 in 16.3. The upper bound of 16.3 becomes about 3,400 prompt tokens, about 3,900 with generation. `OLLAMA_CONTEXT_LENGTH=8192` keeps a margin of more than 100 %.
- **Truncation and tokens.** No `truncating input prompt` line appears in the run's Ollama log. `prompt_eval_count` equals `usage.prompt_tokens` in 100 of 100 replays. No thinking tokens were produced.
- **Memory.** The desktop used 1.4 GB of VRAM at idle, against 0.9 GB on the day of section 16.

### 22.7 Findings

1. **D8 holds.** On identical contexts, the delimited passages and the data-not-instructions rules change raw-reply accuracy by -0.8 [-2.0, +0.4] points. Model refusals and answer language are unchanged. The prompt is adopted as written.
2. **The guardrails cost 1.3 points.** With the corrected support rule, service accuracy is 71.0 against 72.3 (paired -1.3 [-2.8, +0.0]). Of the 13 answers lost, 2 are support-check refusals, 4 come from the chunk cap and 7 from non-determinism.
3. **`RAG_SUPPORT_MIN_SHARE=0.6` is kept.** The share must be taken against the rendered context, labels included, as the tree now does: that rule refuses 0.47 % of correct answers. The rule of the run refused 2.8 %, above the 1 % limit, because the model copies passage labels into its answers. The output filter refused no legitimate answer.
4. **The model and the patterns complement each other.**
   - The model alone recalls 75 % of direct and 35 % of indirect suite cases at 0.5, and misses every prompt-exfiltration and answer-change case in all five languages.
   - The patterns cover the suite (a floor check).
   - Combined, the false-positive rate is 0.39 % of corpus pages and 0.18 % of chunks, all on OCR noise, and 0 of the 750 sample questions are flagged.
   - `INJECTION_THRESHOLD=0.5` is kept.
5. **The model costs a lot of ingestion throughput on CPU.** Sequential ingestion falls by 38 % (+1.36 s per page) and ingestion with 12 clients by 59 %, where the detector's pool of 2 is the bottleneck. A question costs about 130 ms. The patterns cost about 2 ms per page.
6. **The detector moves to the GPU in a follow-up change** (done in `add-gpu-inference`, section 23). The user decided this because of the ingestion cost on CPU. This section (22.4 and 22.5, on `docintel-test:hard2`) is the CPU baseline that change will be compared against: the same 60-document sample, the same six cells and the same suite and benign sets.
7. **The chunk cap changed 83 of 659 contexts** and raised model refusals on them. Its accuracy effect is within the noise. It is kept for D9's security argument, not for quality.

**Caveats.**
- **Two images.** The quality run used `docintel-test:hard`, which has no instruction patterns. The patterns flag none of the 750 sample questions (22.4), so the quality figures stand for `hard2`. The cost cells used `hard2`. The first cost cells, on `hard`, were set aside (22.5).
- **Re-score, not re-run.** The final quality figures re-apply the tree's guardrails offline to the recorded replies and contexts. The model calls are identical, so only the post-processing differs; no new model run was made with the corrected rule.
- **Benchmark-only limit overrides.** The 12-client cost cells ran with `MAX_JOBS_PER_CLIENT=0` and `RATE_LIMIT_UPLOADS_PER_MINUTE=0`. The service's defaults would answer 429 to a single client address sending 12 documents at once.
- **Pattern rates are a floor check.** The patterns were written against the suite. Their recall on unseen phrasings is not measured here; the model's figures are the unbiased ones.
- **Benign set.** The benign set is the synthetic corpus of section 1. Real scans with denser OCR noise may raise the false-positive rate, and the corpus has no forms with imperative instructions beyond the suite's benign sentences.
- **One model, one quality run.** Only `gemma4:e4b-it-qat` was re-measured. Temperature 0 is not deterministic: 86 of 100 replays reproduce the recorded reply. The reference is the section 16 recording of 2026-10-28, so the latency columns compare across days.
- **CPU timing.** The cost cells ran on a desktop with other applications open (4-34 % host CPU before the runs). The detector's ingestion figures depend on its pool size of 2 and the shared torch threads.

### 22.8 What the security changes cost: time and resources before and after

One table for the whole change, each row against the service before `add-defensive-hardening` (sections 16-19) or against the same image with the detector model switched off (`INJECTION_MODEL_PATH` unset). "Guard" below means the Prompt Guard 2 model; "patterns" the rule detector.

**Start-up and memory** (`resources.sh`; runtime image `docintel:latest` built from the final tree, `QDRANT_LOCATION=:memory:`, no language model, the three sample PDFs ingested one after the other; two runs each, cgroup v2 `memory.current` and `memory.peak` of the container):

| | guard off | guard on | change |
|---|---|---|---|
| start to `/health/ready` | 7.1 / 7.7 s | 8.7 / 8.7 s | +1.0-1.6 s |
| lifespan (models loaded, `service started` line) | 4.5 s | 5.8 s | +1.3 s |
| memory after start-up | 1,003 / 1,004 MiB | 1,234 / 1,233 MiB | +230 MiB |
| memory peak, 3 sample PDFs sequential | 1,111 / 1,113 MiB | 1,409 / 1,351 MiB | +240-300 MiB |
| hardened stack under the smoke and hygiene suites (cgroup peak, task 6.1) | 3.0 GB (section 19, 16 CPUs, ingest) | 3.45 GB | +0.45 GB, different loads: indicative |
| runtime image (`docker images`) | 5.76 GB (`INJECTION_MODEL=0`) | 7.88 GB | +2.1 GB listed, +1.13 GB of layer (the model) |

**Ingestion time** (CPU):

| measurement | guard off | guard on | change |
|---|---|---|---|
| sample PDFs, sequential, idle host: `screen_ms` per page | about 1 ms (patterns) | 311-340 ms | +0.3 s per page |
| sample PDFs, sequential: `duration_ms` per document (1-2 pages) | 1,612 / 2,075 / 2,112 ms | 1,906 / 2,767 / 2,719 ms | +18-34 % |
| 60-document sample, sequential (22.5): docs/min | 15.78 | 9.85 | -38 %, +1.36 s per page |
| 60-document sample, 12 clients (22.5): docs/min | 87.8 | 36.4 | -59 %, the detector's 2 slots are the bottleneck |
| patterns alone, 12 clients | 87.8 | 87.9 | none measurable, about 1.8 ms per page |
| OCR, embedding, store per document | unchanged (`ocr_ms`, `embed_ms`, `store_ms` within run-to-run spread) | | |

The screening time per page differs between the samples (about 0.3 s) and the corpus cells (about 1.4 s): the corpus pages are denser (two 512-token windows instead of one) and the cost cells ran on a desktop with other applications open (22.5 caveats).

**Questions** (22.2, 22.6; the GPU with Gemma 4 e4b):

| | before | after | change |
|---|---|---|---|
| screening per question (model) | 0 | 130 ms p50, 157 ms p95 | +130 ms |
| request p50 / p95 | 0.52 / 1.74 s | 0.70 / 1.43 s | +0.18 s p50; p95 lower (another day, see 22.6) |
| system prompt | 116 tokens | 188 tokens | +72 |
| prompt tokens, mean / max | 1,493 / 2,661 | 1,631 / 2,763 | +9 %; worst case about 3,400, within the 8,192 context |
| VRAM, device max | 5.7 GB | 6.1 GB | the idle desktop rose by 0.5 GB that day: the model's share is unchanged (the guard runs on the CPU) |
| service accuracy | 72.3 | 71.0 | -1.3 [-2.8, +0.0] |

**The other defences** cost nothing measurable in these runs: the rate limiter and the headers run per request in microseconds (the 12-client cells match the throughput of section 19 with the detector off); the render budget and the image pre-check add a metadata walk per page that does not show in `ocr_ms`; the support check and the output filter take well under 1 ms per answer (a 100,000-token answer takes 0.06 s, review 9.1); the pattern set takes 20-35 ms on an adversarial 30,000-character text and about 1.8 ms on a corpus page.

**Summary.** Everything but the guard model is free at this scale. The guard model costs about 230-300 MiB of RAM, 1-1.6 s of start-up, 1.1 GB of image, 130 ms per question, and on CPU 0.3-1.4 s per page, which is 38 % of sequential and 59 % of concurrent ingestion throughput. That cost is what `add-gpu-inference` addresses.

---

## 23. In-process models on the GPU (`add-gpu-inference` task 4.1, design D5)

Scripts: `benchmarks/gpu/scripts/` (local, like every benchmark script; `benchmarks/README.md` step 15). Raw outputs in `benchmarks/gpu/out/` (local, not committed):
- `cost/`: the ingestion and extraction cells, `tables.md`, `vram.csv` and `marks.txt`.
- `vram/`: the models alone and together.
- `ask/`: the question runs.
- `parity/`: the model parity and the `cuda`-marked suite.
- `cold/`: start-up and memory.

**Two rounds.** 23.1-23.10 hold the first round (tree of 18:46 UTC). After it, three changes were made in the code:
- NER stays in float32 on the GPU (`place(..., half=False)`), per D2, because fp16 moved name boundaries (23.7);
- the detector gained a reserved question slot (`screen_query`, `FifoSlots` at pool + 1), because questions waited for it (23.6);
- the GPU pool defaults became 1 / 1 / 1 (23.5).

The affected numbers were re-measured on images rebuilt from that tree (23.11). Below, they are given as **now**, with the first round's figures kept as "before".

### 23.1 Method and environment

**Question.** What each component costs, and what moving the three in-process models (NER, e5 embedder, Prompt Guard 2 detector) to the GPU gains, measured against the CPU baseline of section 22 on the same sample and settings.

**Host.** As in section 22:
- RTX 4070 Ti 12 GB (driver 617.14, WDDM; CUDA 12.6 wheels, torch 2.14.1+cu126) and 16 logical CPUs.
- Docker Desktop 29.9.2 on WSL2.
- No other compose project ran. The desktop held 1,493-1,521 MiB of VRAM at idle (1.4 GB in 22.6).

**Images.** All built from the current tree on 2026-10-29, between 18:46 and 18:55 UTC; no file of `app/`, `tests/`, the Dockerfile or the lock changed after the builds.
- **`docintel-test:gpu`**: `TORCH_VARIANT=cuda TEST_IMAGE=docintel-test:gpu docker compose -f docker-compose.test.yml build tests`, `sha256:f8acb329...`. The `tests` service extends `torch-cuda`, which also reserves the GPU for the container.
- **`docintel-test:cpu`**: the same tree with `TORCH_VARIANT=cpu`, `sha256:f90ec5a2...`. It is the CPU reference of this section, so CPU and GPU rows run the same code. Section 22's `docintel-test:hard2` predates the device change.
- **Runtime images for 23.8**: `docintel:gpu41` (`--build-arg TORCH_VARIANT=cuda`) and `docintel:cpu41`.

**Cells** (`gpu_cost.py`, a generalisation of `ingest_cost.py`):
- The 60-document sample, with the same 2 warm-up documents, re-sent and excluded: 58 documents and 98 pages per cell.
- A real Uvicorn in the test container against the throwaway Qdrant of `docker-compose.test.yml`, fresh per cell. Compose project `gpucost`, `down -v --remove-orphans` after every cell.
- Default settings: on the GPU, `DEVICE=auto` resolves to `cuda:0`, and the derived pools are jobs 12, NER 2, embedder 2, detector 1, OCR workers 4. On the CPU they are 12 / 8 / 4 / 2 / 4. Every server's `concurrency in use` line confirms these.
- **Benchmark-only overrides, as 22.5:** `MAX_JOBS_PER_CLIENT=0` and `RATE_LIMIT_UPLOADS_PER_MINUTE=0` in every cell. Their 12-client cells would otherwise be refused by the per-address limits.
- Detector switches as 22.5: `INJECTION_MODEL_PATH` unset (model off) and `INJECTION_PATTERNS=false` (patterns off).
- `/api/extract` sends the corpus name lists, as section 17's client does.
- One pass per cell. The 12-client default and pool cells ran twice (23.5).

**Placement of one model.** `DEVICE` is global (D1), so the settings cannot put one model on the CPU and the others on the GPU. `bench_app.py` does it for the benchmark only:
- It wraps `app.bootstrap.adapter_configs` and sets `device="cpu"` on the adapters named in `BENCH_CPU_ADAPTERS`.
- Each adapter moved to the CPU gets the pool the CPU rule derives on this host: embedder 4, detector 2.
- The server logs a `bench device override` line naming the change. Nothing in `app/` changed.
- NER on the CPU with the rest on the GPU is covered by extraction with `DEVICE=cpu`, since NER is the only model `/api/extract` runs.

**VRAM.**
- The host's `nvidia-smi` is sampled every second through every chain, and every 250 ms in `gpu_vram.sh`.
- Inside a process, `torch.cuda.mem_get_info` and torch's allocator counters. Under WSL2 this view starts from a different base than the host's, so only its deltas are used.

### 23.2 VRAM

**Each model alone and the three together** (`vram_models.py`; the adapters as the service builds them, fp16 on `cuda:0`):
- One fresh container per row, after the load.
- 40 s of load from `pool` threads on the 200 densest corpus pages (NER `find_persons` and detector `screen`) or on batches of 16 chunks (embedder), plus one `embed_query` thread.
- "Host" is the device's peak above the desktop's 1,493 MiB. It includes the process's CUDA context and libraries.

| model(s) | pools (NER / emb / det) | torch allocated after load | in-process device use: after load / peak (above the context) | torch max reserved | **host peak above idle** |
|---|---|---|---|---|---|
| **NER, now fp32** | **1** | 1,066 MiB | 1,158 / 1,210 MiB | 1,160 MiB | **1,403 MiB** |
| NER, before (fp16) | 2 | 544 MiB | 598 / 686 MiB | 622 MiB | 879 MiB |
| NER | 4 | 544 | 598 / 774 | 694 | 967 |
| e5 embedder | 2 (+1 query slot) | 546 | 598 / 1,100 | 1,028 | **1,293** |
| e5 embedder | 4 (+1) | 546 | 598 / 1,326 | 1,238 | 1,519 |
| Prompt Guard 2 | 1 | 547 | 602 / 794 | 738 | **987** |
| Prompt Guard 2 | 2 | 547 | 602 / 960 | 896 | 1,153 |
| **the three, now (NER fp32)** | **1 / 1 / 1 (defaults)** | 2,158 | 2,226 / 2,660 | 2,578 | **2,853** |
| the three, before (fp16) | 2 / 2 / 1 | 1,622 | 1,680 / 2,252 | 2,154 | 2,445 |
| the three | 4 / 4 / 2 | 1,622 | 1,680 / 2,612 | 2,474 | 2,805 |

Each model's weights take about 545 MiB in fp16. The context and libraries of one process add about 280 MiB on the host; with the three models the peak adds 0.5-1.0 GB of working memory.

**In service** (host peak of the device over each cell, desktop included):

| situation | device peak | above the desktop |
|---|---|---|
| runtime image after start-up (23.8) | 3,394 MiB | 1,873 MiB |
| 12-client ingestion, defaults | 3,818-3,820 MiB | 2.3 GB |
| 12-client ingestion, detector pool 2 | 3,882-3,958 MiB | 2.4-2.5 GB |
| 12-client extraction | 3,610-3,646 MiB | 2.1 GB |
| Gemma 4 e4b alone (the CPU-image question run, `OLLAMA_NUM_PARALLEL=2`, context 8,192) | 6,264-6,305 MiB | 4.8 GB |
| Gemma alone plus the three models, questions only | 8,179 MiB | 6.7 GB |
| 12-client ingestion, now (NER fp32, pools 1 / 1 / 1) | 4,384-4,386 MiB | 2.9 GB |
| 12-client extraction, now | 4,178 MiB | 2.7 GB |
| Gemma plus the three models, questions only, now | 8,733 MiB | 7.2 GB |
| **the three models and Gemma, 12-client ingestion during the questions, now (23.11)** | **9,369 MiB of 12,282** | **7.8 GB** |
| the same, before (NER fp16, pools 2 / 2 / 1) | 8,803 MiB | 7.3 GB |

**Margin.** The worst case measured is 12-client ingestion with questions on the shared GPU. With NER in fp32 and pools of 1 it leaves **2.91 GB (24 %)** of the 12 GB device free with the desktop's 1.5 GB in use, or about 4.4 GB on a headless host. Before, with NER in fp16 and pools 2 / 2 / 1, it left 3.48 GB (28 %). fp32 NER adds about 0.52 GB of weights. Raising the pools to 4 / 4 / 2 adds about 0.36 GB at peak (table above).

### 23.3 Per-stage times and throughput: GPU image against the CPU image of the same tree

Everything on, i.e. the image's defaults: detector model and patterns. Stage times are per page of each document (p50, p95 in the second line of a cell), from the outcome lines.

| route | clients | image / device | docs/min | OCR ms/page | NER ms/page | screen ms/page | embed ms/page | store ms/doc | `duration_ms`/doc p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|
| ingest | 1 | CPU image | 9.7 | 1,632 | - | 1,541 / 2,028 | 551 / 876 | 4.9 | 6,776 / 8,611 |
| ingest | 1 | **GPU image** | **19.8 (+104 %)** | 1,619 | - | **94 / 181** | **16 / 39** | 4.0 | 3,032 / 4,166 |
| ingest | 12 | CPU image | 35.3 | 2,126 | - | 8,340 / 16,815 | 832 / 1,340 | 5.8 | 20,446 / 24,798 |
| ingest | 12 | **GPU image** | **135.4, 131.0 = 133.2 (x3.8)** | 2,624-2,658 | - | **78-79 / 214-243** | **17-20 / 104-109** | 6.2-6.8 | 4,899 / 6,167 |
| extract | 1 | CPU image | 15.4 | 1,624 | 563 / 908 | - | - | - | 3,866 / 5,073 |
| extract | 1 | **GPU image** | **19.7 (+28 %)** | 1,631 | **54 / 75** | - | - | - | 2,888 / 4,037 |
| extract | 12 | CPU image | 86.0 | 2,818 | 1,528 / 2,624 | - | - | - | 7,707 / 10,118 |
| extract | 12 | **GPU image** | **125.6, 129.9 = 127.8 (+49 %)** | 2,550-2,631 | **126-173 / 327-437** | - | - | - | 5,138 / 6,379 |

- Every request of every cell returned 200.
- The CPU image reproduces section 22.5: 9.7 against 9.85 docs/min sequential, 35.3 against 36.4 with 12 clients, and 15.6 / 85.8 against 15.78 / 87.8 with the detector off.
- **Per page, the GPU is 10x faster for NER** (563 -> 54 ms in fp16; 102 ms, 5.5x, in the fp32 now used, 23.11), **16x for screening** (1,541 -> 94 ms) and **34x for embedding** (551 -> 16 ms) sequentially. With 12 jobs the gaps widen to 9x, 107x and 49x: on the CPU these stages also wait for pool slots.
- **OCR is now the whole cost.** It is unchanged at about 1.62 s per page sequentially and 2.5-2.7 s with 12 jobs (CPU contention between 12 Tesseract jobs). On the GPU image OCR is 94 % of a document's ingestion time sequentially (mean 1,795 of 1,914 ms per page), against 48 % on the CPU image.
- **The GPU is mostly idle.** Its mean utilisation was 7-16 % in every GPU cell (maximum 44-100 % in bursts).
- Store stays at 4-7 ms per document.
- **Container memory peak** (cgroup, client included): 3.6 GB on the GPU against 3.2 GB on the CPU for 12-client ingestion, and 2.5 GB against 1.8 GB sequentially, from the CUDA libraries.

**NER in fp32, now (23.11).** Extraction runs at 19.3 docs/min sequentially, against 19.7 in fp16, and 125.3, 130.4 = 127.9 with 12 clients, against 127.8. NER takes 102 / 163 ms per page (p50 / p95) sequentially, against 54 / 75 in fp16 and 556 / 897 on the CPU, and 126-155 / 282-356 ms with 12 jobs. Ingestion with the 1 / 1 / 1 pools runs at 20.0 docs/min sequentially and 131.2, 134.6 = 132.9 with 12 clients, against 19.8 and 133.2. fp32 doubles NER's time per page on the GPU, but extraction stays OCR-bound and its throughput does not move.

### 23.4 Per-component impact on the GPU, one switch at a time

Ingestion; docs/min, with the change against "everything on" of the same image and clients. Per-page `screen_ms` and `embed_ms` are p50.

| cell | device of NER / embedder / detector | 1 client | 12 clients | screen ms/page (1 / 12) | embed ms/page (1 / 12) |
|---|---|---|---|---|---|
| everything on | GPU / GPU / GPU | 19.8 | 135.4 (131.0) | 94 / 78 | 16 / 17 |
| detector model off (`INJECTION_MODEL_PATH` unset) | GPU / GPU / - | 20.6 (+4 %) | 135.6 (+0 %) | 1.2 / 1.6 | 44 / 54 |
| patterns off (`INJECTION_PATTERNS=false`) | GPU / GPU / GPU | 19.9 (+0 %) | 134.1 (-1 %) | 93 / 45 | 18 / 17 |
| both off | GPU / GPU / - | 20.7 (+5 %) | 138.1 (+2 %) | 0 / 0 | 44 / 62 |
| **all on the CPU** (`DEVICE=cpu`, CUDA image) | CPU / CPU / CPU | **9.7 (-51 %)** | **36.1 (-73 %)** | 1,538 / 8,220 | 553 / 842 |
| embedder on the CPU (benchmark override, pool 4) | GPU / CPU / GPU | 14.9 (-25 %) | 85.1 (-37 %) | 108 / 94 | 557 / 1,637 |
| detector on the CPU (benchmark override, pool 2) | GPU / GPU / CPU | 11.2 (-43 %) | 40.1 (-70 %) | 1,567 / 8,132 | 45 / 46 |

Extraction, where NER is the only model:

| cell | 1 client | 12 clients | NER ms/page (1 / 12) |
|---|---|---|---|
| GPU (default) | 19.7 | 125.6 (129.9) | 54 / 173 |
| `DEVICE=cpu`, CUDA image | 15.4 (-22 %) | 85.5 (-33 %) | 567 / 1,636 |
| CPU image | 15.4 | 86.0 | 563 / 1,528 |

- **On the GPU no switch matters.** With all three models on the GPU, the detector model costs 4 % of sequential ingestion throughput and nothing measurable with 12 clients (135.4 against 135.6 docs/min), against -38 % and -59 % on the CPU (22.5). The patterns cost 1-2 ms per page here too.
- In the cells without the detector the embedding takes 44-62 ms per page instead of 16-17 ms. Without screening first, each document's embedding call pays the GPU's first-touch costs (allocator growth, cuBLAS selection) on its own; this is noise of a few tens of ms against 2.6 s of OCR.
- **Attribution to each model, by moving one back to the CPU:**
  - **detector:** on the GPU it gains +77 % sequentially (11.2 -> 19.8) and **x3.4 with 12 clients** (40.1 -> 135.4). On the CPU it is still the bottleneck of 22.5: 8.1 s per page of screening, mostly waiting for its 2 slots.
  - **embedder:** +33 % (14.9 -> 19.8) and +59 % (85.1 -> 135.4).
  - **NER** (extraction): +28 % (15.4 -> 19.7) and +49 % (85.5 -> 127.8).
- The CUDA build running on the CPU (`DEVICE=cpu`) matches the CPU image within 2.3 %: 9.7 / 36.1 against 9.7 / 35.3 docs/min ingesting, 15.4 / 85.5 against 15.4 / 86.0 extracting. The CUDA wheels cost nothing on the CPU path.

### 23.5 Pool calibration on the GPU (D3)

12 clients, two passes per configuration (the second in reverse order), docs/min. Section 18's rule: the smallest setting within 5 % of the best mean.

| pool varied (others at the defaults) | 1 | 2 (default) | 4 | 8 | best mean, band from | smallest in band |
|---|---|---|---|---|---|---|
| NER, extraction | 130.8, 131.4 = **131.1** | 125.6, 129.9 = 127.8 | 130.4, 130.7 = 130.6 | 130.6, 128.8 = 129.7 | 131.1, 124.5 | **1** |
| embedder, ingestion (detector 1) | 133.4, 131.2 = **132.3** | 135.4, 131.0 = 133.2 | 133.1, 132.0 = 132.6 | - | 133.2, 126.5 | **1** |
| detector, ingestion (embedder 2) | 133.2 (default) | 133.5, 135.1 = **134.3** | - | - | 134.3, 127.6 | **1** |
| embedder 1 + detector 2 | | 131.8, 133.2 = 132.5 | | | | |

- Every configuration lies within 3 % of every other: 125.6-135.4 docs/min, a spread no larger than the two passes of one configuration (up to 4.3 docs/min). OCR bounds throughput at 12 jobs, and no pool size moves it.
- The stage times confirm that the pools are not waited on:
  - NER: 126-188 ms per page at p50 for pools 1-8;
  - embedding: 17-21 ms;
  - screening: 54-87 ms at p50, and 201-543 ms at p95 with the higher p95 at embedder 1 and 4 (passes vary).
- The device peak rises by 60-140 MiB for detector pool 2 (23.2).

**Recommended defaults: `GPU_NER_POOL = 1`, `GPU_EMBEDDING_POOL = 1`, `GPU_INJECTION_POOL = 1`, now applied in `app/bootstrap.py`; their throughput is confirmed in 23.11.** This is the rule's choice; it also gives the least VRAM (23.2: the pools add about 90-230 MiB each from 1 to 4). The embedder keeps its reserved query slot on top of the pool. Not measured here: a GPU-bound workload where the pools would matter, such as many more jobs than the 12 of `MAX_CONCURRENT_JOBS` or a host with more OCR capacity. The question runs of 23.6 used the pools in use at the time, 2 / 2 / 1; 23.11 repeats them with 1 / 1 / 1.

### 23.6 Questions alone and during 12-client ingestion, the GPU shared with Gemma

**Method** (`ask_gpu.py`, `ask_gpu.sh`):
- The `llm_eval` compose override, project `llmeval_gpu`, with Ollama 0.40.12 and `gemma4:e4b-it-qat`: `OLLAMA_NUM_PARALLEL=2`, `OLLAMA_CONTEXT_LENGTH=8192`, thinking off (`reasoning_none`), no hosted model.
- The index is a copy of `llmeval_qdrant` in the volume `gpu_qdrant`, made fresh before every run and removed at the end; `llmeval_qdrant` was only read.
- **A subsample:** the first 200 questions of section 16's fixed order (177 call the model), one client, sequential, after 3 warm-up questions.
- "Under load": 12 more clients ingest the 60-document sample over and over during the questions. Each file gets a comment line after `%%EOF`, so every ingestion OCRs, screens, embeds and stores, as in 19.3. The duplicates this adds to the index are in both load runs alike.
- The same harness ran with the CPU image (Gemma still on the GPU) for the comparison.
- The screening time is the time outside embedding, search and the model call, as 22.5 measured it. It also holds the prompt build and the answer guardrails, about 1.5 ms (below).

| image | load | request p50 / p95 / max (s) | `embed_ms` p50 / p95 | screening p50 / p95 (ms) | `search_ms` p50 / p95 | `llm_ms` p50 / p95 | embed + screening share of service time | ingestion during the run |
|---|---|---|---|---|---|---|---|---|
| CPU | none | 0.7 / 1.2 / 6.6 | 49.9 / 68.6 | 128 / 152 | 3.5 / 4.9 | 520 / 1,056 | 24.7 % | - |
| GPU, before (pools 2 / 2 / 1) | none | 0.6 / 1.1 / 6.7 | 5.6 / 7.7 | 18.7 / 25.9 | 3.3 / 4.2 | 565 / 1,131 | 4.1 % | - |
| CPU, before (no question slot in the detector) | 12 clients | 12.8 / 16.8 / 22.7 | 88 / 200 | 11,721 / 15,451 | 6.8 / 15.7 | 999 / 1,801 | 91.9 % | 37.1 docs/min (screen 8.2 s/page) |
| **CPU, now (reserved question slot)** | 12 clients | **1.0 / 2.0 / 10.9** | 81 / 164 | **236 / 368** | 5.5 / 14.2 | 690 / 1,699 | 30.1 % | 33.5 docs/min (130 documents in 233 s; screen 8.6 s/page) |
| **GPU, now (pools 1 / 1 / 1, NER fp32)** | none | **0.6 / 1.1 / 6.5** | 5.5 / 7.3 | 16.7 / 22.6 | 3.4 / 12.7 | 558 / 1,131 | 3.8 % | - |
| **GPU, now (pools 1 / 1 / 1, NER fp32)** | 12 clients | **0.9 / 1.6 / 10.2** | 26 / 166 | 96 / 321 | 8.1 / 108 | 749 / 1,380 | 19.4 % | **123.4 docs/min** (408 documents, 693 pages; screen 108 ms/page) |
| GPU, before (pools 2 / 2 / 1) | 12 clients | 0.9 / 1.7 / 10.4 | 20 / 143 | 101 / 390 | 8.0 / 24.0 | 750 / 1,359 | 20.2 % | 124.4 docs/min (418 documents, 709 pages; screen 113 ms/page) |

- All 200 questions returned 200 in every run, with the same refusals in both images: 23 threshold, 23-25 model, 1 unsupported, 0 flagged.
- **On the GPU, screening takes 19 ms per question** (130 ms in 22.5, 128 ms on the CPU image here) and embedding the question 5.6 ms (50 ms).
- **Under 12-client ingestion on the CPU, a question waits for the detector.** Question screening shares the detector's 2 FIFO slots with the pages of 12 ingestion jobs, and nothing reserves a slot for questions as the embedder's reserved query slot does (18.3). A question then waits behind the queued pages, 11.7 s at p50, and the request takes 12.8 s at p50. Before the detector, 19.3 measured a p95 under 2.5 s in this situation.
  - **This was a defect of the CPU path. It is now fixed:** the detector reserves a question lane in its `FifoSlots` (`screen_query`), as the embedder does.
  - **Re-measured (23.11):** on the CPU image under the same 12-client ingestion, questions take **1.0 / 2.0 s** (p50 / p95, against 12.8 / 16.8 s) and screening 236 / 368 ms (against 11.7 / 15.5 s). Ingestion during the run held 33.5 docs/min, against 37.1 before (-10 %). Questions now take their slot ahead of pages, and this run lasted 233 s against 2,529 s, so the two figures are not strictly comparable.
  - On the GPU the same queue drains fast enough: screening 0.10 s p50, 0.39 s p95.
- **On the GPU, Gemma and the models share the device.** Under load the model call rises from 0.57 to 0.75 s at p50 and from 1.13 to 1.36 s at p95. Questions run at p50 / p95 0.9 / 1.7 s, against 0.70 / 1.43 s idle in 22.6, while ingestion keeps 124 docs/min (93 % of the 133 without questions).
- **Answer guardrails** (`guard_time.py`, on the host CPU): the tree's `is_refusal`, `support_share` against the rendered context and `output_filter` take **1.5 ms p50, 2.4 ms p95, 5.9 ms max** per answer, re-run 5 times on each recorded reply of the three runs (2,655 timings). They run on the CPU whatever the device.

### 23.7 Parity, CPU against GPU

| check | result |
|---|---|
| `/api/extract` on the 60-document sample, sequential: whole JSON bodies (names, boxes, matches, order), CPU image against GPU image | **60 / 60 identical**: 803 names, all 803 with boxes, 86 matches |
| the same, CUDA image with `DEVICE=cpu` against the GPU and against the CPU image | 60 / 60 identical with each |
| `/api/ingest`, sequential: status, result, `chunks_stored`, `chunks_flagged` | **60 / 60 identical**: 327 chunks, 0 flagged |
| e5, 750 sample questions (`embed_query`): cosine GPU against CPU | min **0.999988**, mean 0.999999; 0 below 0.999 |
| e5, the 3,299 chunks of the 600-document index (`embed_passages`, batches of 16) | min **0.999971**, mean 0.999996; 0 below 0.999 |
| e5, nearest chunk of each of the 750 questions among the 3,299 | **top-1 746 / 750 (99.5 %)**; same top-5 set 718 / 750; top-1 score difference at most 0.0006; 0 questions cross `RAG_THRESHOLD` 0.808 |
| detector, the adversarial suite (`tests/adversarial/cases.py`): direct, direct OCR-noisy, indirect, indirect noisy (4 x 40), benign sentences (30) and questions (15) | **identical flags at 0.5 in every set**: 30 / 23 / 14 / 13 flagged on both, 0 benign; score difference at most 0.057 (noisy direct) |
| detector, 1,030 corpus pages and the 750 sample questions | identical flags: 3 pages flagged on both, 0 questions; score difference at most 0.008 |
| detector, the 3,299 chunks: GPU against the host CPU scores of 22.4 | 3,299 / 3,299 identical flags (5 flagged); score difference at most 0.043 |
| **NER now (fp32 on the GPU), the 1,030 corpus page texts** | **1,030 / 1,030 pages identical**, 8,414 spans on both; extraction 60 / 60 bodies identical (803 names); `pytest -m cuda` 9 passed, 5 skipped |
| NER before (fp16), the 1,030 corpus page texts (`find_persons` after `title_case_caps`, without extraction's word-level rules) | 1,024 / 1,030 pages identical; 8,414 spans on the CPU against 8,412 on the GPU |
| `pytest -m cuda` in `docintel-test:gpu` with the GPU (real models through the image's `NER_MODEL_PATH` and `EMBEDDING_MODEL_PATH`, and the detector) | **9 passed**, 5 skipped (the Playwright tests, e2e image only) |

**The six NER differences** are all boundary cases on degraded text:
- CPU only: `Arenas`, `Harmández` and `M.`;
- GPU only: `Leonor`;
- a span cut differently: `Gebrüder` against `Gebrüder Weiss`, and `Lázaro Quiñones` against `Lázaro`.

None falls in the 60-document sample, whose extraction responses are identical. D2 keeps NER in fp32 on the GPU if spans differ; at 0.6 % of pages in both directions these look like fp16 rounding at decision boundaries, and fp16 was kept in the first round. Following D2, the code now keeps NER in fp32 on the GPU, and the re-measurement finds every span identical to the CPU's (23.11).

In-image CPU against GPU time for these passes:
- NER, 1,030 pages: 225 s on the CPU (16 threads) against 21 s on the GPU;
- e5, 3,299 chunks: 257 s against 4.3 s;
- detector, 1,030 pages: 583 s against 16 s.

### 23.8 Images, start-up and memory

**Images** (`docker images`: disk use / content):

| image | CPU variant | CUDA variant | difference |
|---|---|---|---|
| runtime (`docintel:cpu41` / `docintel:gpu41`) | 7.89 GB / 2.95 GB | 17.2 GB / 6.42 GB | +9.3 GB disk, +3.5 GB content |
| test (`docintel-test:cpu` / `docintel-test:gpu`) | 7.91 GB / 2.96 GB | 17.2 GB / 6.44 GB | the same |

**Start-up and memory** (`cold_gpu.sh`, as 22.8's `resources.sh`):
- `QDRANT_LOCATION=:memory:`, no language model, the three sample PDFs ingested one after the other.
- Two rounds of three cells. Round 1 started with a cold file cache for the CUDA image, which inflates its memory figures.

| cell | start to `/health/ready` (round 1 / 2) | lifespan (`service started`) | cgroup memory after start / peak (round 2) | device memory after start / after the PDFs |
|---|---|---|---|---|
| CPU image | 9.8 / 8.0 s | 6.6 / 5.6 s | 1,234 / 1,399 MiB | 0 |
| **CUDA image on the GPU** | **12.0 / 9.7 s** | **8.0 / 6.9 s** | **2,227 / 2,473 MiB** | **+1,873 / +1,939 MiB** |
| CUDA image, `DEVICE=cpu` | 8.4 / 8.4 s | - / 5.7 s | 1,363 / 1,523 MiB | 0 |

- On the GPU, start-up is 1.3-2.2 s longer: CUDA initialisation and moving the models to the device.
- The process uses about 1 GB more RAM: the CUDA libraries, and the weights are loaded on the CPU before they move.
- Round 1 of the CUDA image peaked at 3,987 MiB with its cold page cache.
- The start-up line names the device: `"device": "cuda:0", "gpu_name": "NVIDIA GeForce RTX 4070 Ti", "gpu_memory_mib": 12281`.

### 23.9 Impact of each component on the GPU

Against the CPU baseline: section 22.5 and this section's CPU image of the same tree. The 12-client figures carry the benchmark-only limit overrides of 23.1.

| component | time per page or per question: CPU -> GPU | throughput, sequential and 12 clients | VRAM (peak above idle) | notes |
|---|---|---|---|---|
| **OCR** (Tesseract, stays on the CPU) | 1.62 s/page sequential, 2.1-2.8 s with 12 jobs; unchanged | the limit: 94 % of a document's time on the GPU image | 0 | now the only lever for throughput (`OCR_WORKERS`, CPUs) |
| **NER** (xlm-roberta-base, **fp32** now) | 563 -> 102 ms/page (fp16 before: 54); 1,528 -> 126-155 with 12 jobs | extraction 15.4 -> 19.3 (+25 %); 86.0 -> 127.9 (+49 %) | 1,403 MiB (pool 1), 1.07 GB of weights (fp16 before: 879 MiB) | spans identical on all 1,030 corpus pages and the sample (fp16 before: 6 pages differed) |
| **embedder** (e5-base, fp16) | 551 -> 16 ms/page; query 50 -> 5.6 ms | ingestion +33 % and +59 % (embedder alone moved back to the CPU) | 1,293 MiB (pool 2) | cosine >= 0.99997; top-1 99.5 % |
| **detector model** (Prompt Guard 2) | 1,541 -> 94 ms/page (8,340 -> 78 with 12 jobs); question 128 -> 19 ms | its cost falls from -38 % / -59 % on the CPU to -4 % / 0 %; moving it to the GPU gives +77 % and x3.4 | 987 MiB (pool 1) | identical flags on every set |
| **patterns** | 1.2-1.6 ms/page; 0.07 ms/question (22.4) | 0 % / -1 % | 0 | CPU code |
| **answer guardrails** (support check, output filter) | 1.5 ms p50, 2.4 ms p95 per answer | none measurable (CPU code) | 0 | |
| **Gemma sharing the GPU** | model call 0.57 -> 0.75 s p50, 1.13 -> 1.36 s p95 under 12-client ingestion; questions 0.9 / 1.6 s with pools 1 / 1 / 1 (0.9 / 1.7 before) | ingestion 133 -> 123-124 docs/min with questions running (-7 %) | Gemma 4.8 GB; all together peak 9.37 of 12.28 GB, **2.91 GB free** (before: 8.80, 3.48 free) | on the CPU image the same load gives questions 1.0 / 2.0 s with the reserved detector slot (before: 12.8 / 16.8 s) |
| **all three models on the GPU** | ingestion 6.8 -> 3.0 s per document (p50) | ingestion **9.7 -> 19.8 (x2.0)** and **35.3 -> 133.2 (x3.8)**; extraction x1.3 and x1.5 | about 2.7-2.9 GB in service with NER fp32 (2.3-2.5 before) | + 1.0 GB RAM, + 1.3-2.2 s start-up, + 9.3 GB image |

### 23.10 Findings

1. **The GPU removes the detector's cost.** Ingestion with every defence on rises from 9.7 to 19.8 docs/min sequentially and from 35.3 to about 133 with 12 clients. That is above the 87.8 docs/min that 22.5 measured on the CPU with the detector off. On the GPU, switching the model or the patterns off changes throughput by 0-5 %.
2. **Each model's gain:**
   - the detector gives most of the 12-client gain: x3.4 with the detector alone moved;
   - the embedder +59 %;
   - NER +49 % on extraction.
3. **OCR is now the bound.** It takes 94 % of an ingestion's time. The GPU stays at 7-16 % mean utilisation, and no pool size changes throughput.
4. **GPU pools: 1 / 1 / 1 by section 18's rule**, every setting within 3 % (23.5). They are now the defaults in code (start values 2 / 2 / 1); 23.11 confirms ingestion 132.9 and extraction 127.9 docs/min with them.
5. **VRAM fits with room.** With NER in fp32 the three models take about 2.7-2.9 GB in service and Gemma 4.8 GB. The worst measured case leaves 2.91 GB of the 12 GB device free with a desktop on it (3.48 GB with fp16 NER).
6. **Results do not depend on the device.** Extraction and ingestion responses are identical on the 60-document sample; e5 vectors are within 0.99997 cosine with 99.5 % top-1 agreement; the detector flags are identical on every set. NER spans differed on 6 of 1,030 raw corpus pages in fp16; in fp32, now the code's choice, all 1,030 are identical.
7. **Defect found on the CPU path, now fixed:** question screening had no reserved detector slot, so under 12-client ingestion on the CPU a question waited about 12 s for the detector (23.6). With the reserved question lane (`screen_query`), questions on the CPU under that load take 1.0 / 2.0 s (p50 / p95), and ingestion during the run held 33.5 docs/min (23.11).
8. **The CUDA build costs nothing on the CPU** (`DEVICE=cpu` within 2.3 % of the CPU image). It costs 9.3 GB of image and about 1 GB more RAM on the GPU path.

**Caveats.**
- **Benchmark-only overrides.** The limit overrides of 22.5 apply to every cell. The one-model-on-the-CPU cells use `bench_app.py`; the service's settings cannot express them (`DEVICE` is global).
- **Short cells.** One pass per cell, two for the defaults and the pools. A 12-client GPU cell lasts about 26 s, so differences under 4 % between GPU cells are noise.
- **Questions.** The question runs are a subsample (200 of the 750 questions), one run per configuration. The first round used pools 2 / 2 / 1; 23.11 repeats the GPU runs with 1 / 1 / 1. The load runs add duplicates of the sample to a copy of the index, as 19.3 did.
- **VRAM and memory.** Host `nvidia-smi` under WDDM reports the whole device, desktop included, with 1 s or 250 ms sampling, so short peaks can be missed. The in-process figures are deltas from a WSL2 base. RAM figures are cgroup `memory.current`, page cache included.
- **One host, synthetic scans, short documents** (1-2 pages). Workloads where the GPU is the bound were not measured: long documents, more than 12 jobs, several replicas on one GPU. The NER parity of 23.7 uses page texts from the index, not extraction's word-level pipeline (that is the 60-document HTTP check).

### 23.11 Re-measurement after the code changes (NER fp32, detector question slot, pools 1 / 1 / 1)

**Images.** `docintel-test:gpu` (`sha256:e9380ff8...`), `docintel-test:cpu` (`c911c1e8...`), `docintel:gpu41` (`dbfa7656...`) and `docintel:cpu41` (`1e34e097...`), rebuilt from the changed tree on 2026-10-29 at 21:58 UTC, with the same commands as 23.1. Each server's start-up line confirms the GPU pools 1 / 1 / 1.

**Method.** The scripts of 23.2-23.7 unchanged, outputs in `benchmarks/gpu/out/v2/`:
- `gpu_vram.sh ner:1,1,1 all:1,1,1`;
- `parity_models.py --skip e5,detector` with `pytest -m cuda`;
- `cells_v2.txt` through `gpu_run.sh` (one pass, two for the 12-client cells);
- `ask_gpu.sh`: GPU alone, GPU under load, CPU under load.

| measurement | before (round 1) | now |
|---|---|---|
| NER raw-page spans, GPU against CPU (1,030 pages) | 1,024 identical (fp16) | **1,030 / 1,030 identical**, 8,414 spans each |
| extraction bodies, GPU against CPU image (60 documents) | 60 / 60 | **60 / 60** (803 names, all with boxes); also 60 / 60 against round 1's CPU bodies |
| `pytest -m cuda` in the CUDA image | 9 passed, 5 skipped | 9 passed, 5 skipped |
| NER ms/page on the GPU, sequential p50 / p95 | 54 / 75 (fp16, pool 2) | **102 / 163** (fp32, pool 1); CPU image 556 / 897 |
| NER ms/page on the GPU, 12 jobs p50 | 126-173 | 126-155 |
| extraction docs/min, sequential / 12 clients | 19.7 / 127.8 | **19.3 / 127.9** (125.3, 130.4); CPU image 15.6 sequential |
| ingestion docs/min, sequential / 12 clients | 19.8 / 133.2 | **20.0 / 132.9** (131.2, 134.6) |
| NER VRAM alone, host peak above idle | 879 MiB (fp16, pool 2) | **1,403 MiB** (fp32, pool 1); weights 1,066 MiB |
| the three models alone, host peak above idle | 2,445 MiB (2 / 2 / 1) | **2,853 MiB** (1 / 1 / 1) |
| device peak, 12-client ingestion (no Gemma) | 3,820 MiB | 4,386 MiB |
| **worst case: the three and Gemma, 12-client ingestion with questions** | 8,803 of 12,282 MiB, 3.48 GB free | **9,369 MiB, 2.91 GB free (24 %)** |
| questions on the GPU alone, p50 / p95 / max | 0.6 / 1.1 / 6.7 s | **0.6 / 1.1 / 6.5 s** (screening 17 / 23 ms, embed 5.5 / 7.3 ms) |
| questions on the GPU under 12-client ingestion | 0.9 / 1.7 / 10.4 s; ingestion 124.4 docs/min | **0.9 / 1.6 / 10.2 s**; ingestion 123.4 docs/min |
| **questions on the CPU image under 12-client ingestion** | 12.8 / 16.8 / 22.7 s, screening 11.7 / 15.5 s | **1.0 / 2.0 / 10.9 s**, screening 236 / 368 ms |
| ingestion during that CPU run | 37.1 docs/min | 33.5 docs/min |

All 200 questions of every run returned 200, with the refusals of round 1: 23 threshold, 23-25 model, 1 unsupported, 0 flagged. Every ingestion and extraction request returned 200.

**Result.**
- NER in fp32 restores span parity on every corpus page. It costs about 0.5 GB more VRAM and 48 ms more NER time per page sequentially, and no extraction throughput.
- The reserved detector slot removes the question wait on the CPU (12.8 -> 1.0 s p50). Ingestion during that run ran 10 % slower, in a run ten times shorter.
- Pools 1 / 1 / 1 keep throughput and question latency.
- The worst-case margin on the 12 GB device is now 2.9 GB.

