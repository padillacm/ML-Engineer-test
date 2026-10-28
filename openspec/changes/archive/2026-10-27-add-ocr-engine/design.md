# Design

## Context

Block 1 left the `OcrEngine` port (`recognize(pdf: bytes, language: str | None) -> list[OcrPage]`), `PageRaster.to_points` (per-axis pixel-to-point conversion from the real raster size), `clamp`, `OcrPage`/`PageWord`, the error hierarchy and a `FakeOcrEngine`. The defaults below come from `docs/BENCHMARKS.md` section 2 (Tesseract 5.5.0, tessdata "fast" models, 600 synthetic scanned PDFs). Tesseract is not installed on the Windows host; the user chose to run real-OCR tests in a container.

## Goals / Non-Goals

**Goals:** an adapter that reproduces the benchmarked configuration; deterministic, page-ordered output; no temporary files; every failure mapped to a core error; tests that run on the host without Tesseract (fake runner) and a container target that runs the same suite with the real engine.

**Non-Goals:** NER or name boxes (block 3); environment-based settings (block 5); runtime image, compose and Ollama (block 6); tessdata "best" models or other OCR engines (future work in `DESIGN.md`).

## Decisions

### D1. Call the Tesseract CLI directly, not `pytesseract`

`pytesseract` writes the image and the output to temporary files, which the requirements forbid. The adapter runs `tesseract stdin stdout --oem 1 --psm 3 --dpi <dpi> -l <langs> tsv` through `subprocess.Popen` (list arguments, no shell, `start_new_session=True` on POSIX) and `communicate(input=png_bytes, timeout=...)`, with the inherited environment plus `OMP_THREAD_LIMIT=1`. On timeout or any other interruption the whole process group is killed (`os.killpg(SIGKILL)`), so a wrapper script cannot leave orphaned children; the container runs `tini` as PID 1 to reap anything reparented. The executable path is configurable. The call sits behind a small `TesseractRunner` callable so unit tests inject a fake and never need the binary.
*Alternative: `tesserocr` (C API, model kept loaded).* Faster per page (no ~0.3 s model load per process) but needs compiled bindings that are fragile on Windows and in slim images; worth revisiting for throughput, noted in `DESIGN.md`.

### D2. Rendering with pdfium (`pypdfium2`) from bytes

PyMuPDF was the first choice and is what the benchmark used, but it is AGPL-3.0: running it inside a network service obliges offering the service's source to its users. The runtime therefore renders with `pypdfium2` (Apache-2.0/BSD bindings to Google's pdfium); PyMuPDF stays in the `dev` extra only, to build test PDFs and ground-truth rectangles, and an architecture test forbids importing it from `app/`. The document is opened from bytes; password-protected, corrupt, truncated (missing page objects) and zero-page files raise `InvalidInput("pdf_unreadable")` and never yield invented blank pages. Each page is rendered in grayscale at `dpi/72` scale with its rotation and crop box applied as displayed; the raster descriptor is `PageRaster(bitmap.width, bitmap.height, displayed_width_pt, displayed_height_pt)`, so rotation and rounding are measured rather than assumed. The expected pixel count is checked against `max_page_pixels` (default 40 M) before any page is rendered and rejected with `too_large`. pdfium is not thread-safe, so every pdfium call runs under one module-level lock; only grayscale arrays/PNG bytes leave that section, and pages, bitmaps and the document are closed deterministically. Because pdfium rasterises slightly differently from MuPDF, task 7.2's paired calibration is re-run with this renderer.

### D3. Illumination flattening in numpy, bit-exact with OpenCV, without OpenCV

The benchmark's reference `preprocess()` divides the gray image by a Gaussian-blurred copy (sigma 25 x dpi/150), clips, applies a 3x3 median and stretches contrast from the 1st percentile. A first version used Pillow's `GaussianBlur`, which is a box-blur approximation: 13.4 % of pixels differed from OpenCV and hard-page name recall dropped 4.3 points in a paired comparison (`docs/BENCHMARKS.md` 2.8). The adapter now reproduces OpenCV's 8-bit `GaussianBlur` exactly: kernel size `round(6*sigma+1)|1` with integer taps in 1/256 units built as OpenCV builds them, separable row-then-column filtering with `BORDER_REFLECT_101`, done with float64 FFT convolution in strips (all intermediates are integers below 2^24, so rounding is exact), the division and stretch as lookup tables, and Pillow's 3x3 median (already identical to `cv2.medianBlur`). Output is bit-identical to OpenCV on all 89 hard benchmark pages; cost is about 0.14-0.19 s per A4 page at 200 DPI and peak memory fell from ~13 to ~3.7 bytes per pixel. This avoids a ~60 MB OpenCV wheel and its native dependencies. A blank page is returned unchanged (the reference stretch would turn it black).

### D4. Language: detect once per document from its first pages

Hints are engine-neutral codes (`en es pt fr de`) mapped to Tesseract codes (`eng spa por fra deu`). Without a hint the adapter reads pages in order with the combined `eng+spa+por+fra+deu` model, accumulating their text and asking the domain detector after each page with alphabetic words, until it decides or `detect_max_pages` (default 3) such pages were read; the whole document is then read with the single detected model. If detection stays inconclusive, the combined results already computed are reused and only the remaining pages are read. Cost: one extra combined pass per document (~1.7 s) instead of the per-page two-pass strategy measured in the benchmark (~2.5 s per page). The combined model alone triples accent errors (6.4 % to 19.5 %), which is why it is only the fallback.

`app/domain/language.py` holds ~40 frequent function words per language and a vote with a minimum margin; it returns `None` when the margin is too small. The benchmark's version identified 600/600 documents, but the corpus is template-based, so the margin and fallback matter more than that number.

### D5. Concurrency: a thread pool of single-threaded processes

Rendering is sequential on the calling thread under the pdfium lock; each page's PNG is submitted to a `ThreadPoolExecutor(max_workers=workers)` whose threads only wait on the Tesseract process. At most `2 x workers` rendered pages are held at once, and rendering stops as soon as any page has failed. Default `workers = min(4, os.cpu_count())`; each Tesseract process gets `OMP_THREAD_LIMIT=1`. The benchmark measured 6.8 pages/s with 16 single-thread processes versus 0.07 pages/s with OpenMP left unset in a multi-process layout. On the first failure the remaining futures are cancelled and running processes finish or time out; the error is re-raised as the core error. Results are reassembled in page order.

### D6. TSV parsing

Rows with `level == 5` (word), non-blank `text` and a finite `conf >= 0` become words (non-finite values or integers that overflow are `ocr_failed`); pixel boxes go through `PageRaster.to_points` and `clamp`. Words whose clamped box is empty are dropped. Confidence is not exposed (no consumer yet, and the benchmark did not tune a confidence filter).

### D7. Configuration object

`TesseractOcrConfig` (frozen dataclass): `dpi=200`, `flatten=True`, `psm=3`, `oem=1`, `languages=("en","es","pt","fr","de")`, `detect_min_words=20`, `detect_max_pages=3`, `timeout_s=60.0`, `workers=None` (auto), `max_page_pixels=40_000_000`, `tesseract_cmd="tesseract"`. Invalid values raise `ConfigurationError`. Block 5 builds it from environment settings.

### D8. Errors

Two codes join the closed registry: `pdf_unreadable` (cannot open, corrupt, encrypted) and `unsupported_language`. `too_large` and `ocr_failed` already exist. `FileNotFoundError`, non-zero exit, `TimeoutExpired` and unparsable TSV all become `ProcessingFailed("ocr_failed")` raised `from` the original, with page number and return code in the message but never the image or text.

### D9. Container for the real engine

A `Dockerfile` with a `base` stage on `python:3.12-slim-trixie` (Debian 13 ships Tesseract 5.5, the version benchmarked) installing pinned versions of `tesseract-ocr`, the five language packs and `tini`, plus `uv` copied from its pinned image; a `test` stage that runs `uv sync --frozen --extra dev` with a build-cache mount (the uv cache stays out of the image), runs as a non-root user under `tini` and has `pytest` as its command. `docker build --target test -t docintel-test . && docker run --rm docintel-test` runs the full suite including the `tesseract` marker. Block 6 adds a `runtime` stage to the same file. `.dockerignore` keeps `.venv`, caches, `benchmarks/`, `docs/`, `openspec/`, `.claude/` and `.git` out of the build context.

### D10. Tests

- Unit (host, no binary): rendering, flattening, TSV parsing, coordinate conversion, language flow, concurrency order, error mapping, resource cleanup, using PDFs generated in the test with PyMuPDF (dev-only) and a fake runner that returns canned TSV.
- Contract: an `OcrEngine` contract in `tests/application/contracts.py` (page count and order, page sizes, boxes inside the page) run against `FakeOcrEngine` on the host and the real adapter in the container.
- Integration (`@pytest.mark.tesseract`, skipped when the binary is absent): the three sample PDFs (every page, first page non-empty); a generated image-only PDF with words drawn at known rectangles, including accents, a rotated page and 150/300 DPI renders (IoU >= 0.5); a Spanish document detected as `es`; an encrypted PDF; no files created in the temp directory.
- Coverage now includes `app/adapters` with the same 90 % gate; integration-only lines are covered by the container run, so the host gate applies to domain and application only and the container run checks the adapter.

## Risks / Trade-offs

- [A hand-written port of OpenCV's blur could drift from OpenCV] -> analytic kernel tests, a direct-convolution oracle and literal OpenCV outputs for small cases are in the test suite; the paired corpus calibration (task 7.2) is re-run with the final renderer.
- [Per-page process start costs ~0.3 s of model loading] -> acceptable at the measured ~1 s/page; the `tesserocr` alternative is recorded for throughput work.
- [Language detection on a template corpus looked perfect] -> margin threshold and combined-model fallback; the decision is logged in block 5 for observability.
- [Real scans may be rotated by 90 degrees without a `/Rotate` flag] -> out of scope; Tesseract's OSD (`--psm 0`) is noted as future work.
- [Debian's Tesseract version can change with the base image] -> the image tag is pinned and `tesseract --version` is printed by an integration test so drift is visible.
- [Integration tests need Docker Desktop running] -> they are skipped, not failed, on a host without Tesseract; the pre-push hook keeps running the host suite.

## Migration Plan

Additive: no endpoint uses the adapter until block 5. Rollback is deleting `app/adapters/ocr/`, the two error codes and the new files.

## Open Questions

- Default `workers` for the eventual production container (CPU limits there); answered in block 6 with the compose resource limits.
