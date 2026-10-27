# Proposal

## Why

The core defines an `OcrEngine` port but nothing implements it, so no endpoint can read a scanned PDF yet. OCR quality decides everything downstream: a name the OCR misses can never be extracted, boxed or matched, and a box computed in the wrong space is useless to the compliance reviewer. The benchmark on 600 synthetic scanned PDFs (en/es/pt/fr/de, see `docs/BENCHMARKS.md` section 2) fixed the configuration: render at 200 DPI in grayscale, flatten illumination, Tesseract `--oem 1 --psm 3` with a single language, one thread per engine process. With it, names were found in 97.6 % of mentions and boxes reached IoU >= 0.5 in 96.2 %; without flattening, hard scans collapsed to 43 % name recall.

## What Changes

- Add a Tesseract-based `OcrEngine` adapter under `app/adapters/ocr/` that renders every page with pdfium (`pypdfium2`) from in-memory bytes, flattens illumination, runs the Tesseract CLI through stdin/stdout (no temporary files), parses its word-level TSV and returns `OcrPage`s with word boxes in PDF points.
- Add language handling: an explicit language hint selects one model; without a hint, a first pass with all five models over the first pages with enough text (up to a small bound) detects the document language, and the whole document is then read with that single model. An inconclusive detection falls back to the combined model.
- Add a stop-word language detector to the domain (pure, no dependencies).
- Process pages concurrently with a bounded number of single-threaded engine processes (`OMP_THREAD_LIMIT=1`), keeping page order.
- Map failures to the core error hierarchy: unreadable or encrypted PDF, unsupported language, oversized page render, engine missing, crashing or timing out. Add the error codes `pdf_unreadable` and `unsupported_language` to the closed code registry.
- Add a configuration object for the adapter (DPI, flattening, page segmentation mode, languages, detection threshold, timeout, workers, pixel budget), validated at construction. Binding it to environment variables is block 5.
- Add a `Dockerfile` with a `test` target (Python 3.12, Tesseract 5 with eng/spa/por/fra/deu) that runs the full test suite including real-OCR integration tests; block 6 adds the runtime target to the same file.
- Add an `OcrEngine` contract test run against the fake now and against the real adapter in the container.
- Dependencies: `pypdfium2` (Apache-2.0/BSD), `pillow`, `numpy`; `pymupdf` only in the `dev` extra to build test PDFs and ground truth, because its AGPL-3.0 licence would apply to a network service. No `pytesseract` (it writes temporary files) and no OpenCV (too heavy for one blur and one median filter).

## Capabilities

### New Capabilities
- `page-ocr`: turning every page of a scanned PDF into recognised words with boxes in PDF space, including rendering, preprocessing, language selection, concurrency, resource handling and failure modes.

### Modified Capabilities
<!-- None. The OcrEngine port signature from add-domain-foundation is kept unchanged. -->

## Impact

- New code: `app/adapters/ocr/`, `app/domain/language.py`; two new codes in `app/domain/errors.py`; new tests under `tests/adapters/ocr/` and `tests/integration/`.
- New files: `Dockerfile`, `.dockerignore`.
- `pyproject.toml`/`uv.lock`: three runtime dependencies; a pytest marker for tests that need the Tesseract binary; coverage extended to `app/adapters`.
- Tests that need Tesseract are skipped on a host without it and run in the container. Docker Desktop must be running for them.
- Out of scope: NER and name boxes (block 3), the HTTP endpoints, upload validation and settings from environment (block 5), the runtime image and compose (block 6).
