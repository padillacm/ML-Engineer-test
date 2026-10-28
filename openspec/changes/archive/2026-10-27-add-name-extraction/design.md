# Design

## Context

Available from blocks 1-2: `OcrEngine` returning `OcrPage`s of `PageWord`s in PDF points; `PageText.from_words` / `locate(span)` / `clamp`; `match_names` (NFKD fold, `token_sort_ratio >= 0.90`); the `NameRecognizer` port (`find_persons(text) -> list[TextSpan]`); `FakeOcrEngine`, `FakeNameRecognizer`; the `docintel-test` image (Python 3.12, Tesseract 5.5, `tini`). The evidence for every default below is `docs/BENCHMARKS.md` sections 3-5 (public NER ranking, end to end on OCR text, span-rule ablation, fuzzy threshold).

## Goals / Non-Goals

**Goals:** reproduce the benchmark's recommended chain (model, input preparation, span rules, boxes, fuzzy) with every piece except model inference in pure, unit-tested domain code; offline, pinned model; a use case that the block-5 endpoint calls unchanged.

**Non-Goals:** HTTP, upload validation, environment settings (block 5); GPU inference; spaCy or GLiNER fallbacks (rejected in the benchmark); a legal-form/street lexicon to cut organisation false positives (exploratory +0.3 F1, tuned on the corpus, future work).

## Decisions

### D1. Model: `Davlan/xlm-roberta-base-ner-hrl`, pinned and baked in

Best end-to-end F1 on OCR text (92.0 vs 89.5 for mBERT-hrl, paired +2.5 [+1.9, +3.1]); AFL-3.0 licence (XLM-R base is MIT). The adapter loads from a local directory (`model_path`); the image downloads the model at build time at a pinned commit (`revision`) into `/models/xlm-roberta-base-ner-hrl` and sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so run time never touches the network. `Davlan/bert-base-multilingual-cased-ner-hrl` stays a configuration alternative (same API, smaller, -2.5 F1), not a second adapter. The model's `PER` label is the only one used; `ORG`, `LOC`, `DATE` are discarded.

### D2. Inference with `transformers` on CPU `torch`

`pipeline("token-classification", model, tokenizer, aggregation_strategy="first", stride=64, device="cpu")`; the fast tokenizer gives character offsets and `stride` covers pages longer than 512 tokens (the benchmark's setting). No score threshold (a 0.8 floor cost 1.0 F1). `torch` comes from the PyTorch CPU wheel index through `[tool.uv.sources]` so neither the host nor the image pulls CUDA. `torch.set_num_threads(ner_threads)` (default 2) keeps NER from competing with the four single-threaded Tesseract workers. Inference runs under one lock per recogniser: the pipeline object is not documented as thread-safe and one model instance per process is the memory budget (about 1 GB resident).
*Alternative: ONNX Runtime export.* Faster and smaller, but adds a conversion step whose accuracy must be re-measured; recorded as future work.

### D3. Input preparation in the domain (`app/domain/names.py`)

The page text is `PageText.from_words(words).text` (words joined by single spaces, so line breaks are already spaces, as in the benchmark's best variant). A same-length copy has every ALL-CAPS word of two or more letters title-cased character by character (first letter kept, the rest lowered only when lowering keeps one character, so "STRASSE" and "ß" never change length); offsets stay identical, so spans found on the copy index the original words. Title-casing was the largest single gain (+1.6 F1 for XLM-R; capital-letter names are 14.6 % of mentions).

### D4. Span rules in the domain, on word ranges

Model spans are snapped to whole words with `PageText` offsets. The rules work on word index ranges of `PageText.words` using each word's text and line index, in the benchmark's order (`ner_corpus/scripts/pipe.py`): split at line breaks (Xsplit); extend to the preceding initials on the same line, "J. R. R." (M3; any upper-case letter, e.g. "Ł."); merge adjacent model spans across hyphens and apostrophes (M1) and "Last, First" pairs with a surname of up to four words (M2 relaxed, the largest rule: +12.3 F1 alone; only when the first name is itself a model span, since rescuing unspanned words was the rejected Rlf rule; it does not join two full names in a list, i.e. when both sides have two or more words and the document language is known and not es/pt, or a title precedes the left side, or only the right side is in capitals (a capitalised left side is the usual "SURNAMES, Given names" form and joins in any language); a range that already contains a "Last, First" join is never joined again, so "Weber, Hans, Smith, John" gives two names: measured on all 600 corpus documents this is F1 +0.0 [-0.1, +0.1] while a plain "both sides two words" rule cost -0.3 [-0.5, -0.1] by splitting Iberian "Apellido Apellido, Nombre Nombre" names; with an unknown language an untitled list stays joined); strip leading titles (T: the union of the benchmark's list and the project's: Dr, Dra, Docteur, Doutor, Doutora, Prof, Profa, Professor, Pr (French), Sr, Sra, Srta, Sres, Señor, Señora, Senhor, Senhora, Doña, Dña, Dona, Don (Spanish only) and Dom (Portuguese only, both are also first names), Mr, Mrs, Ms, Mx, Miss, Madam, Sir, Dame, Mme, Mlle, Madame, Mademoiselle, Monsieur, MM, Herr, Frau, Fräulein, Hr, Fr, Ing, Lic, case-insensitive, with or without period; the missing "Pr" alone cost -3.3 F1 in French in the first calibration; "M." only when the document language is French, otherwise it is an initial; a title on the same line right before the range also marks it titled); drop one-word spans unless a title preceded them (Dsingle: +9.6 F1, the main precision rule); drop spans with no capital letter, spans with digits or "@", and spans with fewer than three letters (Dlower, Ddigit, Dshort). Rejected by the ablation and not implemented: particle bridging, proportional box trimming, regex rescue, score floor. Each rule is a small pure function with its own tests; the reported text is the original words of the final range joined by spaces with punctuation stripped only at the edges (so "Gonzalez Ruiz, Maria" keeps its comma and a final initial keeps its period). The entry point is `clean_mentions(page_text, spans, language=None)`, whose ranges index `page_text.words`.

### D5. Lines come from the OCR engine

Xsplit needs line boundaries, and the page text has none. `PageWord` gains a required `line: int` (0-based, per page, in reading order), filled by the Tesseract adapter from the TSV `block_num/par_num/line_num` triple. Geometry heuristics were rejected: two-column layouts and tables put different lines at the same height. This is the only change to the OCR contract and it is a `page-ocr` delta.

### D6. Use case `ExtractNames` (`app/application/extract_names.py`)

`ExtractNames(ocr: OcrEngine, recognizer: NameRecognizer, threshold: float = DEFAULT_THRESHOLD)`, `__call__(pdf, requested: Sequence[NamePair], language: str | None = None) -> ExtractionResult`. Per page: build the text, run the recogniser on the title-cased copy, apply the rules, locate each final word range as the union of its word boxes, clamp to the page. `ExtractionResult(names: list[NameMention], matches: list[FuzzyMatch])` with `NameMention(name: str, box: Box)` (new domain types). Matches use `match_names` over distinct extracted names (first-occurrence order, distinct by normalised text). The use case depends only on ports and domain code; the router in block 5 maps the result to `ExtractionResponse`.

### D7. Configuration

`TransformersNerConfig` (frozen): `model_path`, `labels=("PER",)`, `aggregation="first"`, `stride=64`, `threads=2`, `batch_size=8`. Validation at construction (`ConfigurationError` for a missing directory, missing tokenizer or model files, invalid numbers). Block 5 builds it from environment settings.

### D8. Container

A `model` build stage installs only `huggingface_hub`, downloads the pinned revision, verifies the expected files and is copied into `base` at `/models`; the `test` stage inherits it. Model tests are marked `ner_model` and skip when the directory is missing (the Windows host); they run in the container.

### D9. Tests

- Domain: one test module per rule plus the spec scenarios (titles, "Last, First", initials, single words, line split, lowercase/digit/short drops, title-casing length invariant with ß and accented capitals).
- Use case: fakes for OCR and NER (scripted spans), covering order, repeated names, boxes covering only name words, empty request, matches deduplicated.
- Adapter unit: the `transformers` pipeline replaced by a stub returning scripted entity groups (label filtering, offsets, lock, config validation).
- Container, `ner_model`: the spec scenarios with the real model (mixed entities, capitals, long page, offline with sockets blocked), the `NameRecognizer` contract, and the three sample PDFs end to end through real OCR.
- Calibration: the use case on the same 60-document sample of the synthetic corpus as block 2, scored with the benchmark's mention definition (IoU >= 0.5), paired against the benchmark's default; acceptance is the benchmark's F1 interval.

## Risks / Trade-offs

- [Image grows by ~1.3-1.5 GB (torch CPU + model)] -> one image layer cached between builds; ONNX export recorded as the size and speed follow-up.
- [Benchmark F1 comes from template documents with fixed name pools] -> the calibration reproduces the benchmark, not reality; `DESIGN.md` states this and the audit viewer exists so a reviewer can check boxes.
- [Organisation names that look like people (Robert Bosch, Carl Zeiss) are reported] -> known 7.1 % trap false-positive rate; lexicon rule left as future work.
- [Bare surnames are not reported (recall 20 % in the benchmark)] -> intended by Dsingle; titled surnames are kept.
- ["M." is ambiguous without the language] -> treated as an initial unless the language is French (hint or detection in a later change).
- [CPU contention between NER threads and Tesseract workers] -> both are configurable; block 6 sets them against the container CPU limit.

## Migration Plan

`PageWord` gains a required field, so the OCR adapter, fakes and tests that build `PageWord`s change in the same change. Archiving blocks 1 and 2 first gives this change a baseline to modify. Rollback: revert the change; no data is stored.

## Open Questions

- None that change the specs or tasks.
