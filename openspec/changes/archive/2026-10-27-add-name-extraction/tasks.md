# Tasks

## 1. Baseline

- [x] 1.1 Archive `add-domain-foundation` and `add-ocr-engine` with `openspec archive` (specs move to `openspec/specs/`); verify `openspec list --specs` shows `page-geometry`, `name-matching`, `text-chunking`, `architecture-boundaries` and `page-ocr`, and `openspec validate --all --strict` passes

## 2. Lines in OCR output

- [x] 2.1 Add the required `line: int` field to `PageWord` (validated >= 0), fill it in the Tesseract adapter from `block_num/par_num/line_num` (0-based per page, reading order) and update `FakeOcrEngine`, builders and tests; verify unit tests for the TSV mapping and an integration test in the container where two printed lines get different indices and each page starts at 0

## 3. Domain: input preparation and span rules

- [x] 3.1 Add `NameMention` and `ExtractionResult` to the domain types and implement `app/domain/names.py` title-casing (same length, ALL-CAPS words of 2+ letters, ß and accented capitals safe); verify length and offset invariants with a seeded randomised test
- [x] 3.2 Implement span snapping to word ranges and the REC rules (Xsplit, T with the "M." language rule, M1, M2 relaxed, M3, Dsingle, Dlower, Ddigit, Dshort) as separate functions plus one `clean_mentions` pipeline; verify one test module per rule and every scenario of the `name-extraction` spec that does not need the model

## 4. Use case

- [x] 4.1 Implement `ExtractNames` in `app/application/extract_names.py` (per-page text, recogniser on the title-cased copy, rules, union box of the final words clamped to the page, distinct-name fuzzy matching); verify with fakes: page and reading order, repeated names on two pages, box covers only name words, empty request, matches deduplicated with score 1.0 for the accent case, the architecture guard still passes

## 5. NER adapter

- [x] 5.1 Add `transformers` and CPU `torch` (PyTorch CPU index via `[tool.uv.sources]`) to `pyproject.toml`, `uv lock`, `uv sync --extra dev`; verify `torch.cuda.is_available()` is False and no `nvidia-*` package is in `uv.lock`, and run `uv-audit`
- [x] 5.2 Implement `TransformersNerConfig` and `TransformersNameRecognizer` in `app/adapters/ner/` (local path only, offline, `PER` only, `aggregation_strategy="first"`, `stride=64`, threads, lock, vendor errors mapped to `ProcessingFailed`); verify unit tests with a stubbed pipeline and config validation (missing directory -> `ConfigurationError`)
- [x] 5.3 Add a `NameRecognizer` contract (spans inside the text, in order, non-overlapping) and run it against `FakeNameRecognizer` on the host and the real adapter under the `ner_model` marker

## 6. Container

- [x] 6.1 Add the `model` stage (pinned `huggingface_hub`, pinned model revision, file check) copied to `/models` in `base`, offline environment variables, and the `ner_model` marker with auto-skip when the model directory is absent; verify `docker build --target test` succeeds, the image runs the suite with the model, and report the image size before and after

## 7. Verification with the real model

- [x] 7.1 Add container tests for the spec scenarios with the real model (mixed entities, capitals, long page, offline with sockets blocked) and an end-to-end run of the three sample PDFs through real OCR and NER; verify they pass in the container and are skipped on the host
- [x] 7.2 Calibrate on the 60-document sample of the local synthetic corpus (data not committed): mention precision, recall and F1 at IoU >= 0.5 per language and noise level, paired against the benchmark default (`ner_corpus` e3); acceptance is F1 within the benchmark's interval; record the result and CPU seconds per page in `docs/BENCHMARKS.md` section 4

## 8. Wrap-up

- [x] 8.1 Run host checks (`pytest --cov`, ruff, mypy, all pre-commit hooks, `uv-audit`) and the full container suite; verify all pass and coverage stays >= 90 %
- [x] 8.2 Run `openspec validate add-name-extraction --strict`; verify it passes
