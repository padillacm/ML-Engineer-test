# Known limitations

Living list of limitations found while building and testing the service that **cannot be fixed with the current stack** (Tesseract 5.5 + pdfium + `Davlan/xlm-roberta-base-ner-hrl` + rule-based post-processing + rapidfuzz), plus the trade-offs accepted on purpose. Everything that could be fixed was fixed and has a test. Each entry says where the evidence is, what it costs, and what would remove it. Numbers come from `docs/BENCHMARKS.md` unless stated otherwise.

## Stack limitations

### OCR (Tesseract + pdfium)

| # | Limitation | Evidence | Impact | Would remove it |
|---|---|---|---|---|
| O1 | Hard scans (heavy noise, blur, low contrast) lose words: name recall ~87-90 %, box IoU >= 0.5 ~81-84 % on the hard tier, against ~99-100 % on clean pages. | BENCHMARKS 2.6, 2.8 | Names on poor scans are missed or boxed loosely. | Stronger OCR (tessdata "best", a neural OCR engine), or the complexity router in `DESIGN.md`. |
| O2 | The combined five-language model drops diacritics ("Ramirez", "Muller", once "Miller" for Müller); accent errors triple (6.4 % -> 19.5 %). It is only used when the document language cannot be detected (short or mixed-language text). | BENCHMARKS 2.6; `tests/integration/test_ocr_positions.py` | Accent-insensitive matching absorbs most of it, but a changed letter ("Miller") can fall below the 0.90 fuzzy threshold. | Callers pass a language hint; a better language identifier; a multilingual OCR model. |
| O3 | Pages scanned sideways or upside down without a `/Rotate` flag are not read; Tesseract's orientation detection (OSD) is not used. | Review of `add-ocr-engine` | Rotated scans yield no or garbage words. | Run OSD (`--psm 0`) and rotate before OCR (extra pass, ~0.3-0.5 s/page). |
| O4 | pdfium rejects truncated PDFs that MuPDF could partly repair (vector PDFs cut at 90-99 %). | `add-ocr-engine` fixer report | Some damaged-but-readable files fail with `pdf_unreadable` instead of returning their intact pages. MuPDF is excluded by its AGPL-3.0 licence. | A permissively licensed repair tool (e.g. qpdf) as a pre-step. |
| O5 | A page whose content stream points to a missing object renders blank without any error, so it cannot be distinguished from a genuinely blank page. A `/MediaBox [0 0 0 0]` falls back to 612x792 pt. | `add-ocr-engine` fixer report | Such pages silently contribute no names. | Structural PDF validation before rendering (qpdf `--check`). |
| O6 | pdfium and the benchmark's MuPDF rasterise slightly differently (27.6 % of pixels differ, by 0.34 grey levels on average). | BENCHMARKS 2.8 | Measured effect within noise (paired name-recall diff -0.1 [-1.1, +0.7]). | None needed; listed for traceability. |
| O7 | One Tesseract process per page reloads the model (~0.3 s of the ~0.9 s per page). | BENCHMARKS 2.7; design `add-ocr-engine` D1 | Throughput, not accuracy. | `tesserocr` (C API, model kept loaded); its compiled bindings are fragile on Windows and in slim images. |
| O8 | Only Latin-script languages (en, es, pt, fr, de) are installed and evaluated; no handwriting, stamps or signatures. | Corpus definition, BENCHMARKS 1 | Other scripts and handwritten names are not extracted. | More Tesseract language packs plus a matching NER model; a handwriting recogniser. |

### Name recognition (XLM-R NER + rules)

| # | Limitation | Evidence | Impact | Would remove it |
|---|---|---|---|---|
| N1 | Organisations and places that look like people are tagged as persons ("ETH Zurich", "Robert Bosch", "Carl Zeiss"); trap false-positive rate 7.1 % (Spanish person-like companies 37 %, German companies 44 %). | BENCHMARKS 4.2; strict xfail `test_an_organisation_in_a_sample_is_not_reported` | False names in the output that a reviewer has to dismiss. The span passes every rule, so only the model can fix it. | A larger or fine-tuned NER model, or a legal-form/street lexicon (measured +0.3 F1 but tuned on the synthetic corpus, so not adopted). |
| N2 | A bare surname without a title is not reported (recall ~20 % for that mention type), on purpose: dropping one-word spans is the main precision rule (+9.6 F1). Titled surnames ("Dr. Weber", "Pr Rogner") are kept. | BENCHMARKS 4.3 | "Weber said..." is not boxed. | Coreference to a full name earlier in the document (not in this stack). |
| N3 | "Last, First" lists with an unknown language and no title stay joined ("Christopher Wong, Anna Kowalski" -> one mention), as in the benchmark; with a known language other than es/pt they are split. | `tests/domain/test_names.py::TestListOfFullNames` | Two people share one box when the language is not detected. | Better language detection on short pages, or a model that tags each name separately. |
| N4 | "CAPITALISED NAME, Mixed Name" is read as "SURNAMES, Given names" and joined, so "Cc: JOÃO FERNANDES, Luisa Ribeiro" (two people) becomes one mention. From the words alone it cannot be told apart from "GONZÁLEZ RUIZ, María José", which is far more common. | 600-doc measurement: dF1 -0.0 [-0.1, +0.0] (only pt_111 changes) | One known corpus case merged. | Layout or punctuation cues the OCR does not expose, or a model trained on such lists. |
| N5 | A title at the end of the previous line does not mark the name on the next line ("Dr." / "Weber"), and a title inside a "Last, First" mention is kept in the text ("Smith, Dr. John"). Both follow the benchmarked rules. | Review of `add-name-extraction` | Rare; costs a dropped titled surname or a cosmetic title in the name. | Cross-line title rule, to be measured before adoption. |
| N6 | "M." is a title only when the page language is French; elsewhere it is an initial ("M. Dupont" in an English page is reported with the initial). | Design `add-name-extraction` D4 | Ambiguous by nature. | None with rules alone. |
| N7 | Inference cost grows with page length (~0.25 s per 1,000 characters at 2 threads); pages over 50,000 characters are refused with `ner_failed`. | `app/adapters/ner/config.py` (`max_chars`) | Very dense pages fail instead of blocking other requests for minutes. | ONNX/quantised model, GPU, or splitting the page into independent windows. |
| N8 | `torch.set_num_threads` is process-wide, so two recognisers with different thread settings in one process share the last value. | `app/adapters/ner/recognizer.py` docstring | Only matters if several recognisers are created; the service creates one. | Run models in separate worker processes. |

### Name matching (rapidfuzz)

| # | Limitation | Evidence | Impact | Would remove it |
|---|---|---|---|---|
| M1 | Near-identical given names pass the 0.90 threshold: Jon / John 0.947, Anna / Ana 0.947, Luis / Luisa 0.966, Mario / Maria 0.929. | `tests/domain/test_matching_decoys.py` ("known" rows) | Possible false matches between different people with the same surname. | Higher threshold (costs OCR-typo recall), or a second signal (date of birth, ID number) outside this service. |
| M2 | "rn" read as "m" by OCR misses the threshold (0.897 for "Rnaria Gonzalez"). | `tests/domain/test_matching_decoys.py` ("known-miss") | A real person is not matched. | OCR-confusion-aware similarity (not in rapidfuzz), or a lower threshold (costs precision). |

### Packaging and operations

| # | Limitation | Evidence | Impact | Would remove it |
|---|---|---|---|---|
| P1 | The image is 3.9 GB (CPU torch ~0.7 GB, NER model 1.1 GB). | `docker images` | Slow pulls and cold starts. | ONNX export of the NER model, a smaller model (mBERT-hrl, -2.5 F1), multi-stage runtime image without dev tools (block 6). |
| P2 | `huggingface_hub` in the model build stage is version-pinned but not hash-pinned (`--require-hashes` would need every transitive dependency hashed). The model files themselves are sha256-verified. | `Dockerfile` comment | Build-time supply-chain exposure limited to that one stage. | A hashed requirements file for the build stage. |
| P3 | One integration test (`test_the_box_covers_the_name_without_title_or_role`) failed once in a full container run and could not be reproduced in 13 later runs; OCR and NER were shown to be deterministic and far from any threshold. The most likely cause is the 60 s Tesseract timeout under host CPU contention. | `add-name-extraction` fixer report | Possible rare CI flake under load. | If it recurs, the failure message will show whether it was `ocr_failed` (timeout) or an assertion; a larger timeout or fewer workers under test. |

## Evidence caveats

- The benchmark corpus is synthetic: template text (~150 sentences per language), fixed name pools, fact-lookup questions, eight fonts, Latin names only. Absolute scores are optimistic for real documents; paired comparisons between versions are the reliable signal (BENCHMARKS 1).
- Truth boxes are font-metric boxes about 1.37x taller than the ink, so a perfect ink box scores IoU ~0.73; IoU >= 0.5 is the criterion used throughout.
