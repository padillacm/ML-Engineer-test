# Tasks

## 1. Project scaffolding

- [x] 1.1 Edit `pyproject.toml`: add `rapidfuzz`, remove `thefuzz` and `python-Levenshtein`, add `ruff` to the `dev` extra, add `[tool.pytest.ini_options]` (`testpaths = ["tests"]`, `pythonpath = ["."]`) and `[tool.ruff.lint]` selecting at least `E722` (bare except), `BLE001` (blind except), `S110`/`S112` (silent except-pass/continue) and `B`; run `uv lock` and verify `uv sync` succeeds, `uv.lock` exists and `uv run ruff check app tests` passes
- [x] 1.1b Add a lint-failure check: write a throw-away file containing `except: pass`, verify `uv run ruff check` fails on it, then delete the file
- [x] 1.2 Create `app/domain/`, `app/application/`, `app/adapters/` and `tests/{domain,application,architecture,fakes}/` with `__init__.py` files; verify `uv run pytest` runs without collection errors (exit code 5, no tests yet) and `uv run python -c "import app.domain, app.application, app.adapters"` succeeds
- [x] 1.3 Add `.gitattributes` with `* text=auto eol=lf` and verify `git ls-files --eol` shows `eol=lf` for the new files

## 2. Domain types and errors

- [x] 2.1 Add `app/domain/types.py` with frozen slotted dataclasses `Box`, `PixelBox`, `PageWord`, `OcrPage`, `SearchHit`, `TextSpan`, `Chunk`, `NamePair`, `FuzzyMatch`, `ChatMessage`, `EmbeddingIdentity`, and unit-test equality, immutability and `Box` right/bottom edges (spec: page-geometry "Page space is points")
- [x] 2.2 Add `app/domain/errors.py` with `ServiceError(code, message, category)` and its subclasses and a code registry; test category/code exposure, catch-by-base-class and that a message built from a configured secret never contains it (spec: architecture-boundaries "Failures use a closed set of categories")

## 3. Page geometry

- [x] 3.1 Implement `PageRaster.to_points` with per-axis scaling and `union`, `clamp` and `iou` in `app/domain/geometry.py`; test the two conversion scenarios, union (including empty and cross-page), clamp (including fully outside) and IoU scenarios from the spec
- [x] 3.2 Implement `PageText` and `locate(span)` in `app/domain/geometry.py`; test full-name span, mid-word start, blank words and out-of-range span scenarios, plus a seeded randomised loop over generated word lists asserting every located box contains every overlapped word box

## 4. Name matching

- [x] 4.1 Implement `normalize` in `app/domain/matching.py`; parametrised tests for accents, case, ß, whitespace, and idempotence (`normalize(normalize(x)) == normalize(x)`)
- [x] 4.2 Implement `score` and `match_names(extracted, pairs, threshold=0.90)` with best-match, tie-by-request-order, threshold validation and degenerate-input handling; test every scenario of the name-matching spec, including the literal `"María González"` ~ pair `("Maria", "Gonzalez")` case with score 1.0, "Karl-Heinz Müller" vs "Heinz Müller" below 0.90 and the OCR-typo case at or above 0.90
- [x] 4.3 Add a small decoy regression table (same surname different given name, near-miss, reversed order, accent-only difference) as a data-driven test, so later threshold changes are visible

## 5. Text chunking

- [x] 5.1 Implement `clean_page_text` in `app/domain/chunking.py`; test the three cleanup scenarios and that `"Karl-\nHeinz"` keeps its hyphen
- [x] 5.2 Implement the hierarchical splitter with `max_chars` and word safety (no overlap yet); test short page, long page, blank page, mid-word boundary, oversized token, paragraph-preferred and sentence-preferred scenarios
- [x] 5.3 Add word-aligned overlap, validation of `overlap_ratio` in [0, 0.5), provenance (`page`, `sequence`, `start`, `end`) and cross-page isolation; test the default 96-character bound, zero overlap reconstruction, invalid overlap, offsets equal `page_text[start:end]`, and sequence numbering across pages
- [x] 5.4 Add a determinism test and a corpus-shaped smoke test: chunk a checked-in multi-paragraph OCR-style text fixture (hyphenated line breaks, blank lines, 3 pages) and assert no chunk exceeds the limit, no word is split and offsets match the cleaned page text

## 6. Ports and fakes

- [x] 6.1 Add `app/application/ports.py` with the five `Protocol` classes using only core and standard types; verify with a test that each fake satisfies its port via `isinstance` on the runtime-checkable protocols and that a small use-case-shaped function typed against the ports runs with the fakes
- [x] 6.2 Implement `EmbeddingIdentity` equality and the fakes in `tests/fakes/` (`FakeOcrEngine`, `FakeNameRecognizer`, `FakeEmbedder`, `FakeVectorStore`, `FakeLLMClient`); test identity scenarios, store identity binding (same identity reopens, different identity raises `IncompatibleEmbedding` without changes), replace-on-reingest, and cosine scores within 0..1
- [x] 6.3 Write the `VectorStore` and `Embedder` contract tests as a reusable parametrised module that takes a factory fixture; verify it passes against the fakes (real adapters will be added to the same parametrisation in later blocks)

## 7. Architecture guard

- [x] 7.1 Add `tests/architecture/test_imports.py` (AST scan, allowlist of stdlib + `rapidfuzz` + own core packages, relative-import resolution); verify it passes on the real tree
- [x] 7.2 Add tests that prove the guard works: scan a temporary package containing a forbidden import (`fastapi`, `pydantic`, `app.adapters`, `app.models`) and assert the failure names file and import; assert `app.domain` does not import `app.application`
- [x] 7.3 Add a test asserting `app/models/schemas.py` is unchanged (compare its SHA-256 with the value recorded in the test) so the fixed HTTP contract cannot drift unnoticed

## 8. Benchmark evidence

The numbers behind the defaults and the capacity estimate live today only in a session scratchpad. They are preserved in the repository so `DESIGN.md` and `DECISIONS.md` can cite them.

- [x] 8.1 Write `docs/BENCHMARKS.md` with, for each benchmark (OCR/DPI/preprocessing, NER on OCR text, fuzzy threshold, embedders, chunking, refusal threshold): dataset and how it was generated, model/library versions and pins, hardware, the headline table per language and noise level with confidence intervals, caveats, and the exact commands/script names that reproduce it; verify every figure quoted in the proposal and design appears in it and matches the source result files
- [x] 8.2 Add a "Capacity estimate" section to `docs/BENCHMARKS.md` that derives documents/hour per core from the measured inputs (OCR seconds per page, NER milliseconds per page for the median page length, embedding milliseconds per chunk, pages per document = 1,045 / 600) with the formula written out, the hardware they were measured on, and a statement that it must be re-measured on target hardware; verify by recomputing the result with a short script committed next to the document
- [x] 8.3 Copy the small evaluation and generator scripts (corpus generator with its seed, OCR/NER/embedding/fuzzy evaluation scripts) into `benchmarks/` without any data, with a `README.md` stating what each does; verify the folder is under 5 MB and contains no PDFs or model weights

## 9. Wrap-up

- [ ] 9.1 Run `uv run pytest -q` and `uv run ruff check app tests` from a clean `uv sync` and verify all pass with no network access and without importing `spacy`, `chromadb`, `sentence_transformers` or `pytesseract` (assert via `sys.modules` in a test)
- [x] 9.2 Run `openspec validate add-domain-foundation --strict` and verify it passes; add a short "Layout" note to `README.md` describing `domain` / `application` / `adapters` and how to run the tests, and verify the documented command runs as written

## Workflow follow-up

- Review the change with the user, then archive it with `openspec archive` once merged on the feature branch.
- Update `DECISIONS.md` (rapidfuzz dependency rationale, measured defaults) at the end of the project rather than per block.
