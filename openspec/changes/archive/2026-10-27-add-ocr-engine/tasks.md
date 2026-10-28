# Tasks

## 1. Dependencies and tooling

- [x] 1.1 Add `pypdfium2`, `pillow` and `numpy` to `pyproject.toml` (and `pymupdf` to the `dev` extra only, for test fixtures), run `uv lock` and `uv sync --extra dev`; verify `uv run python -c "import pypdfium2, PIL, numpy"` works and `uv run pre-commit run uv-audit --hook-stage manual --all-files` reports no new advisories
- [x] 1.2 Register the `tesseract` pytest marker, make tests with it skip automatically when the `tesseract` executable is not on `PATH`, extend coverage `source` to `app/adapters`, and add mypy settings for `pymupdf` if its stubs are incomplete; verify `uv run pytest -q` and `uv run mypy` still pass

## 2. Domain additions

- [x] 2.1 Add `pdf_unreadable` and `unsupported_language` to the error code registry; verify the registry test and a round-trip pickle test cover them
- [x] 2.2 Implement `app/domain/language.py` (supported codes, stop-word vote with minimum margin, `None` when inconclusive); verify unit tests for each language, mixed text, short text and an unsupported code, and that the architecture guard still passes

## 3. Rendering and preprocessing

- [x] 3.1 Implement `app/adapters/ocr/rendering.py`: open from bytes in a context manager, map open failures and encryption to `pdf_unreadable`, check the pixel budget before rendering, render grayscale at the configured DPI and return PNG bytes plus a `PageRaster`; verify with PDFs generated in the test (A4, Letter, rotated page, encrypted, corrupt bytes, oversized page)
- [x] 3.2 Implement `app/adapters/ocr/preprocess.py` (illumination flattening with numpy and Pillow, reference `preprocess()` from the benchmark); verify that a synthetic gradient-lit image becomes evenly lit (low background variance), that output shape and dtype are unchanged, and that `flatten=False` passes the image through

## 4. Tesseract runner and TSV parsing

- [x] 4.1 Implement the `TesseractRunner` (stdin/stdout, `OMP_THREAD_LIMIT=1`, timeout, configurable executable) and map `FileNotFoundError`, non-zero exit and `TimeoutExpired` to `ProcessingFailed("ocr_failed")`; verify with a fake executable path and with a monkeypatched `subprocess.run`
- [x] 4.2 Implement TSV parsing to `PageWord`s via `PageRaster.to_points` and `clamp` (word level only, blank and `conf < 0` rows dropped, empty boxes dropped); verify with checked-in TSV fixtures including headers only, blank words, accented words and boxes at the image edge

## 5. Engine assembly

- [x] 5.1 Implement `TesseractOcrConfig` with validation and `TesseractOcrEngine.recognize` (render sequentially, OCR in a thread pool, reassemble in page order, cancel on first failure, close everything); verify with the fake runner: every page returned in order including the first, blank page yields no words, results identical for 1 and 4 workers, failure on page 2 closes the document and raises `ocr_failed`, no files appear in the temp or working directory
- [x] 5.2 Implement the language flow (hint -> single model; no hint -> combined pass on the first page with enough words, detect, single model for all pages; inconclusive -> combined; unsupported hint -> `unsupported_language`); verify each branch with the fake runner by asserting the `-l` argument used per call
- [x] 5.3 Add an `OcrEngine` contract to `tests/application/contracts.py` (page count and order, positive page sizes, boxes inside the page) and run it against `FakeOcrEngine`; verify the real adapter is wired to the same contract under the `tesseract` marker

## 6. Container

- [x] 6.1 Add `Dockerfile` (`base` stage: `python:3.12-slim-trixie`, Tesseract with eng/spa/por/fra/deu, uv; `test` stage: project copy, `uv sync --frozen --extra dev`, `pytest` command) and `.dockerignore`; verify `docker build --target test -t docintel-test .` succeeds and `docker run --rm docintel-test tesseract --version` prints 5.x with the five languages listed by `--list-langs`
- [x] 6.2 Document in `README.md` how to run the container tests; verify the documented commands run as written

## 7. Real-OCR verification

- [x] 7.1 Add integration tests under `tests/integration/` (marker `tesseract`): the three sample PDFs (all pages, first page non-empty, boxes inside pages), a generated image-only PDF with words at known rectangles including accents, a rotated page and 150/300 DPI (IoU >= 0.5), Spanish detection, encrypted PDF, no temp files, and the Tesseract version printed; verify they pass with `docker run --rm docintel-test` and are skipped on the host
- [x] 7.2 Re-measure the Pillow-based flattening against the benchmark on a sample of the local synthetic corpus (at least 60 documents, all languages and noise levels, data not committed): name recall and IoU >= 0.5 must fall within the benchmark's confidence intervals; record the numbers in `docs/BENCHMARKS.md` section 2 and switch to OpenCV headless only if they do not

## 8. Wrap-up

- [x] 8.1 Run `uv run pre-commit run --all-files`, `uv run pytest -q --cov` on the host and the full suite in the container; verify all pass, coverage stays >= 90 % and the architecture guard confirms the domain and application layers import no vendor library
- [x] 8.2 Run `openspec validate add-ocr-engine --strict`; verify it passes

## Workflow follow-up

- Archive `add-domain-foundation` (and later this change) with `openspec archive` once the user agrees, so `openspec/specs/` holds the accepted capabilities.
