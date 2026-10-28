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

### Retrieval and answering (e5-base + Qdrant + refusal threshold)

| # | Limitation | Evidence | Impact | Would remove it |
|---|---|---|---|---|
| R1 | The refusal threshold 0.808 was calibrated on the synthetic corpus, where it keeps 94.8 % of answerable questions (re-measured on the implemented chain, BENCHMARKS 8.1). On the three real sample PDFs several answerable questions score just below it and would be refused: "Who wrote the memo?" 0.755, "What was the total research budget?" 0.795, "Who received the Young Investigator Award?" 0.803, while an off-topic question scores 0.747. e5 cosines are compressed into ~0.70-0.85, so the margin between relevant and irrelevant is a few hundredths. | `tests/integration/test_rag_e2e.py` (agent report, block 4) | On real documents the service may refuse questions it could answer. | Recalibrate with labelled questions on real documents of the target domain (procedure in `DESIGN.md`); a lower threshold that leaves fine-grained refusal to the prompt instruction; a reranker or hybrid BM25 retrieval (future work). |
| R4 | On heavily degraded scans retrieval is weaker than the benchmark: answer recall@5 -3.1 [-5.8, -0.5] and parent-page answer-in-context -2.6 [-5.0, +0.2] on the hard noise tier, while every overall and per-language metric matches the benchmark. The cause is OCR quality on those pages (O1), not chunking or embedding. | BENCHMARKS 8.1 (paragraph-aware re-run) | Questions about badly scanned pages are answered less often. | Better OCR on hard pages (O1), or the complexity router in `DESIGN.md`. |
| R2 | "Absent entity" questions (a plausible person or project that is not in the documents) are refused by the threshold only ~37 % of the time; cross-document questions ~11 %. The prompt instruction to reply with the fixed refusal is the second line of defence and depends on the LLM. | BENCHMARKS 8, 8.1 | Some unanswerable questions reach the LLM. | A stronger LLM-side check (verify the answer is supported by the cited pages), a reranker. |
| R3 | Document text goes into the prompt verbatim, so instructions written inside a document ("ignore previous instructions") reach the model; the system prompt and the labelled context blocks mitigate but cannot exclude prompt injection with a general-purpose LLM. | Design `add-rag-pipeline` D5 | A malicious document could steer an answer; sources still show the retrieved text. | Output verification against the sources, model-side injection defences. |

### Packaging and operations

| # | Limitation | Evidence | Impact | Would remove it |
|---|---|---|---|---|
| P1 | The test image is 5.75 GB as `docker images` reports it (3.9 GB before the embedding model); its uncompressed layers total 3.8 GB: NER model 1.1 GB, embedding model (multilingual-e5-base) 1.1 GB, Python dependencies 1.2 GB (CPU torch ~0.7 GB; qdrant-client's required grpcio ~18 MB). | `docker images`, `docker history` | Slow pulls and cold starts. | ONNX export of the models, a smaller NER model (mBERT-hrl, -2.5 F1) or embedder (e5-small, -5.1 answer recall@5, BENCHMARKS 6), multi-stage runtime image without dev tools (block 6). |
| P2 | `huggingface_hub` in the model build stage is version-pinned but not hash-pinned (`--require-hashes` would need every transitive dependency hashed). The model files themselves are sha256-verified. | `Dockerfile` comment | Build-time supply-chain exposure limited to that one stage. | A hashed requirements file for the build stage. |
| P3 | One integration test (`test_the_box_covers_the_name_without_title_or_role`) failed once in a full container run and could not be reproduced in 13 later runs; OCR and NER were shown to be deterministic and far from any threshold. The most likely cause is the 60 s Tesseract timeout under host CPU contention. | `add-name-extraction` fixer report | Possible rare CI flake under load. | If it recurs, the failure message will show whether it was `ocr_failed` (timeout) or an assertion; a larger timeout or fewer workers under test. |

## Evidence caveats

- The benchmark corpus is synthetic: template text (~150 sentences per language), fixed name pools, fact-lookup questions, eight fonts, Latin names only. Absolute scores are optimistic for real documents; paired comparisons between versions are the reliable signal (BENCHMARKS 1).
- Truth boxes are font-metric boxes about 1.37x taller than the ink, so a perfect ink box scores IoU ~0.73; IoU >= 0.5 is the criterion used throughout.
