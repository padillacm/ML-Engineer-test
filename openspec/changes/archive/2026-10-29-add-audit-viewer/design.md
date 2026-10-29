# Design

## Context

`POST /api/extract` returns `extracted_names[]` with a `BoundingBox` (0-based `page_number`, `x`, `y`, `width`, `height` in points, origin at the top-left of the page as displayed: `/Rotate` applied, relative to the crop box; `page-geometry` and `page-ocr` specs) and `fuzzy_matches[]` (`extracted_name`, `matched_name`, `score`). The service has no user interface. It is unauthenticated and published on `127.0.0.1` only (`deployment` spec). The broader security pass (ingestion, prompt, retrieval) is deferred until all blocks are done; this change covers only what the page itself needs to be safe.

## Goals / Non-Goals

**Goals**
- Let a person check every box of a real response by eye, on any PDF the service accepts.
- Make the PDF-space claim testable end to end with a renderer that is not the service's own.
- No new runtime dependency in Python, no JavaScript build toolchain, no network at run time.

**Non-Goals**
- Editing boxes or names, saving audits, users or authentication (the service has none; the deployment spec keeps it on loopback).
- Viewing ingested documents or RAG answers (the service keeps no PDF after a request).
- Mobile layout beyond "usable at 375 px wide".

## Decisions

### D1. Render in the browser with pdf.js, not on the server
The boxes are drawn over a rendering of the PDF made by pdf.js, an independent renderer, from the same file the user uploaded.
- **Why:** the audit must be able to catch a wrong frame. If the server rendered the page images (pypdfium2, the same code path that produced the boxes), a wrong rotation, crop box or pixel-to-point scale would be applied identically to the image and the boxes, and the overlay would look right while the response is wrong. pdf.js computes the displayed frame on its own.
- **Also:** no new endpoint and no change to the fixed contract; the viewer audits exactly what a client receives.
- **Rejected:** a server endpoint returning page images (circular check, a new API surface, larger responses); the browser's built-in PDF viewer (cannot be drawn on); returning an annotated PDF (useful later, but the same circularity, and not interactive).

### D2. pdf.js version and supply
- Pin the newest stable `pdfjs-dist` release that is at least two weeks old and past the CVE-2024-4367 fix (arbitrary script execution through a crafted font, fixed in 4.2.67), recorded with its version and tarball sha256.
- **Proposed:** a Docker build stage downloads the npm tarball, checks the sha256, keeps only `build/pdf.min.mjs`, `build/pdf.worker.min.mjs` and `LICENSE`, and the test and runtime images copy them read-only to `/app/vendor/pdfjs` (same pattern as the models). `scripts/fetch_pdfjs.py` does the same for a host run (`uv run python scripts/fetch_pdfjs.py`); without the files the page explains how to get them (503 on the library files).
- **Alternative (user's choice):** commit the three files (about 1.5 MB) under `app/web/vendor/pdfjs/`. Simpler for a reviewer running on the host, but binary-like files in the repo and an update is a manual copy. The sha256 test is the same either way.
- Options set: `isEvalSupported: false`, worker from `'self'`, no CMaps or standard fonts fetched from the network (the scanned PDFs are images; text rendering is not needed for the audit, and a missing font only affects vector text).
- **As built:** pdfjs-dist 6.4.299 (2026-10-03), tarball sha256 `86269b40…c415`, past CVE-2024-4367 (fixed 4.2.67) and CVE-2026-16633 (fixed 6.2.108). pdf.js 6 removed `isEvalSupported` and its builds contain no `eval(` or `new Function(`, so the compiled-glyph path CVE-2024-4367 abused no longer exists; the option is passed and ignored. Nine files are kept, each with its own sha256: the library, the worker, the JavaScript decoders pdf.js 6 imports for CCITT/JBIG2 and JPEG 2000 images when WebAssembly is off (`useWasm: false`, so the CSP needs no `wasm-unsafe-eval`), and five licence files. The pin lives in one file, `deploy/pdfjs.lock`, read by `scripts/fetch_pdfjs.py` (which the `pdfjs` build stage runs, so host and images get the same bytes), the hygiene check and the tests. pdf.js files are cached for one day, not longer, because their URL carries no version.

### D3. Coordinate rule
- pdf.js `page.getViewport({ scale })` uses the page's `/Rotate` and its crop box by default, with a top-left origin, so the displayed frame in pdf.js is the frame of the response. A box is drawn at `left = x * scale`, `top = y * scale`, `width * scale`, `height * scale`, in CSS pixels, over a canvas rendered at `scale * devicePixelRatio`.
- No derotation or y-flip in the browser. If the service's frame were wrong, boxes would miss on rotated or cropped pages; the end-to-end tests include those pages (D7).
- The response's page sizes are not part of the contract, so the viewer cannot cross-check them; it reports a box that lies outside the pdf.js page (by more than 1 pt) as an audit warning in the list.
- **/UserUnit and /Rotate (review fix).** pdf.js applies a page's `/UserUnit`; PDFium, the service's renderer, ignores it and pypdfium2 cannot read it, so the response for such a page is in user units. The viewer draws the page in the service's frame (the pdf.js viewport at `scale / page.userUnit`) and lists the page audit warning "page uses /UserUnit N: the service's boxes are in user units, not points" (KNOWN_LIMITATIONS V4). A `/Rotate` that pdf.js reports as something other than a number multiple of 90 is flagged too; a number that is not a multiple of 90 cannot be seen, because pdf.js reports 0 while PDFium uses 90 (V5).

### D4. Page structure and interaction
- One HTML page, ES modules, one stylesheet, no framework (the page is small; no build step and no npm dependency besides pdf.js).
- Boxes are absolutely positioned elements over each page canvas, not drawn on the canvas: they stay crisp at any zoom, can be focused and labelled for screen readers, and the tests read their positions from the DOM.
- Matched names: solid box plus a "matched" label; unmatched: dashed box plus "no match". A name is matched when it appears as `extracted_name` in `fuzzy_matches`; its best `matched_name` and `score` are shown.
- Names input: one "First Last" per line (the last word is the last name; a line with a comma is read as "Last, First"), or a JSON array as the API takes it; the page shows the JSON it will send. Validation errors from the API are shown with their `code`.
- Pages are rendered when they come near the viewport (IntersectionObserver) and their canvases are released when far away, so memory stays bounded on 50 pages; box elements exist for all pages so the list works before rendering.
- The raw response can be downloaded as JSON; the request id (`X-Request-ID`) and the duration are shown for support.
- **Loading (review fix).** While a request runs, the file, names and language inputs are disabled and Cancel aborts it (`AbortController`); the page returns to idle with a "Cancelled" status, and the result line names the file it belongs to. The status live region is set at start, end, error and cancel only; the elapsed time ticks in an element outside any live region. Name problems are shown at once and announced in a visually hidden polite region 1 s after typing stops, only when they change. Response bodies are read defensively: `detail` and `loc` only as arrays, `request_id` only as a string or finite number.

### D5. Serving and safety
- GET and HEAD `/audit` (also `/audit/`) return the page; `/audit/static/{name}` serves a file listed in `app/web` with an allowed extension (`.mjs`, `.js`, `.css`, `.svg`; the page only at `/audit`); `/audit/vendor/pdfjs/{name}` serves one of the nine pinned pdf.js files; any other path under `/audit/` answers 404 with the viewer's headers, never the router's trailing-slash redirect, whose `Location` comes from the `Host` header. A name never becomes a path: it is matched against the directory listing.
- Headers on all of them: `Content-Security-Policy: default-src 'none'; script-src 'self'; worker-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`. `Cache-Control` is `no-store` on the page and on errors, `no-cache` on the page's own files and `public, max-age=86400` (one day) on pdf.js. File responses carry an ETag, and a matching `If-None-Match` answers 304 with the same headers.
- Every value from the response is inserted with `textContent` or attributes, never `innerHTML`: names come from OCR of untrusted documents.
- `AUDIT_VIEWER_ENABLED=false` removes the routes (404), for deployments that expose only the API.
- The page is part of the HTTP layer (`app/api/viewer.py` plus `app/web/`); the architecture test keeps it from importing adapters or use cases.

### D6. No change to the HTTP contract
The viewer uses `POST /api/extract` as any client does. It does not need page sizes, word boxes or confidences; adding them would change the response for every client. If the audit shows that one is needed, it is a separate proposal.

### D7. Tests
- **Python (host and container):** route, files, content types, headers, 404 when disabled and for unknown files, pdf.js hash, the page has no inline script or style (CSP would block it), architecture.
- **Browser (container only):** an `e2e` image stage `FROM test` with the `e2e` dependency group (Playwright for Python, pinned) and one pinned headless Chromium; `docker-compose.e2e.yml` runs it against the runtime stack on the internal network. Checks:
  1. **Position:** every box element's rectangle equals the response's box x scale, offset by its page's position, within 1 px, at two zoom levels.
  2. **Ink:** for each box, the share of dark pixels of the rendered canvas inside the box against a ring of the same area around it; a box on the name has dark text inside and paper around. Threshold set from the 60-document sample (D8).
  3. **Frames:** synthetic PDFs with `/Rotate` 90/180/270, a crop box offset from the media box, and mixed page sizes in one file: the ink check passes on all of them.
  4. **Behaviour:** list and box selection, filter, zoom, keyboard focus and labels, error display (non-PDF upload, oversized upload, invalid names), CSP violations reported by the browser (none allowed), no network request other than `/api/extract` and the page's own files.
- Synthetic PDFs are made by the existing test fixtures (the corpus generator already writes rotated and cropped pages); the e2e stage adds no PDF library to the runtime image.
- **As built:** the frame PDFs are built by `tests/e2e/frames.py` (Pillow and pypdfium2, test dependencies) with upright displayed text, as a scanner writes it: content rotated in user space and `/Rotate` compensating, so the test checks the frame and not OCR orientation (KNOWN_LIMITATIONS O3). Ink is a pixel darker than the local paper level (median of the box's neighbourhood) minus 40 grey levels, with table rules excluded; a box passes with an inside share ≥ 0.16 and inside/ring ratio ≥ 2.1 (BENCHMARKS 20). Chromium forces HTTPS for the host name `app` (an HSTS-preloaded top-level domain), so the e2e compose file reaches the service through the alias `docintel-app` (KNOWN_LIMITATIONS V2). The e2e stage installs only Chromium's headless shell (758 MB layer instead of 1.18 GB).
- **As built (review fixes):** the frame PDF has ten pages (adding `/UserUnit 2`, JPEG 2000 and CCITT G4); every rendered canvas must match its page element within 1 px; a 60-page canned-response test checks lazy rendering and no DOM or listener growth over repeated extractions; loading tests check announcements, locking and cancel. The e2e browser is Chrome Headless Shell from a sha256-pinned `chromium` stage; only its apt libraries come from `playwright install-deps` (KNOWN_LIMITATIONS V6).

### D8. Evidence plan
1. pdf.js version, tarball sha256, files kept and their sizes.
2. Ink check on the 60-document sample: share of boxes that pass at the chosen threshold; every failure inspected and classified (OCR box too tight or too wide, name split across lines, rendering difference) in BENCHMARKS.
3. Rotated, cropped and mixed-size pages: all boxes pass.
4. Performance in headless Chromium on this host: page load, first page rendered, all boxes drawn for 1, 10 and 50 pages, browser memory for 50 pages, and render time per page at 100 % and 200 %.
5. Image sizes before and after (runtime and test), and the `e2e` stage size.

### D9. Low-confidence OCR names (follow-up of the calibration, task 9)
The ink calibration (task 6.1, BENCHMARKS 20.3) traced three false-positive names and one too-wide box to words Tesseract read with a low confidence. Two places can use the confidence: before recognition (drop the words below a minimum, so the recogniser never sees them) or after it (rules on the candidate names, the text unchanged). Both are compared on the 600-document corpus, paired by document against the unchanged chain; a variant is adopted only if its recall CI includes 0 (the names a reviewer must not lose matter more than the false ones the viewer shows).
- **As built:** dropping words costs recall from a minimum of 10 (-0.8 [-1.1, -0.4]): removing a word changes the recogniser's context. Trimming low-confidence edge words from names costs more. The adopted rule, Dconf, drops a candidate name whose every word has a confidence below 20: precision +0.2 [+0.1, +0.3], recall -0.0 [-0.1, +0.0], persons unchanged, end to end with the image under test (BENCHMARKS 21). Words carry their confidence (`PageWord.confidence`, `None` for an engine that reports none, which the rule treats as readable). Of the calibration cases it removes "Co Jrrao"; the two signatures, which keep one word read above 20, and the stray "a" remain (KNOWN_LIMITATIONS N9).

## Risks / Trade-offs

- **pdf.js and the service disagree on an unusual PDF** (for example a `/UserUnit` or a malformed crop box): the boxes would look wrong although the service followed its own spec. The ink check on the sample and the frame tests bound this; differences found go to KNOWN_LIMITATIONS with the file that shows them.
- **Untrusted PDFs in the browser**: pdf.js parses the user's own file in the user's browser; the risk is to that browser session. Mitigations: pinned patched version, no eval, strict CSP, no other origin reachable.
- **Headless Chromium in the test image grows the e2e stage**, not the runtime image. The runtime image grows only by the pdf.js files.
- **Lazy rendering** makes "every page is rendered" a matter of scrolling; the tests scroll through every page.

## Open Questions

- pdf.js supply: downloaded at build with a sha256 (proposed) or committed to the repository? (D2)
- Browser end-to-end tests with Playwright (proposed), or Python tests plus a documented manual check only? The automated ink and frame checks are what make the PDF-space claim testable; the cost is the `e2e` stage (about 150 MB of Chromium plus Playwright) and its run time.
