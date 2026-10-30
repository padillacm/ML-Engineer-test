# Audit viewer

`/audit` is a page for checking every box of a real `/api/extract` response by eye. It sends the
chosen PDF to `POST /api/extract` like any client, renders the same file in the browser with pdf.js
(an independent renderer, not the service's PDFium) and draws each returned box at its points x the
zoom scale over the page as displayed (`/Rotate` applied, relative to the crop box, top-left origin,
no correction in the browser). A box in image pixels, measured from the wrong origin or on a wrong
frame visibly misses its name.

With the stack running (`docker compose up -d --wait`; the images carry pdf.js), open
`http://127.0.0.1:8000/audit`. On the host, fetch the pinned pdf.js files once with
`uv run python scripts/fetch_pdfjs.py` (into `vendor/pdfjs`, git-ignored), start the service as in
[Configuration](configuration.md) (running the service outside Docker) and open the same address. Without the files the page says so and the pdf.js
URLs answer 503.

- **Inputs:** a PDF; optionally the names to match, one per line as "First Last" (the last word is
  the last name) or "Last, First", or the JSON array the API takes (the page shows the JSON it sends
  as `names`, and a line it cannot read blocks the request); the language (auto-detect, en, es, pt,
  fr, de). While a request runs, these inputs are locked and **Cancel** aborts it (the page returns
  to idle and says so); the result line names the file it belongs to, with the duration, the request
  id (`X-Request-ID`) and any error with its `code` and message. Only the start, the end, an error or
  a cancel are announced to screen readers (the running time is shown but not announced), and a
  problem in the names is announced once typing pauses.
- **Boxes:** a name that appears as `extracted_name` in `fuzzy_matches` has a **solid** box labelled
  "matched", every other name a **dashed** box labelled "no match" (line style and label, not colour
  alone). Hovering or focusing a box (Tab moves through the boxes in page order) shows the name, its
  0-based page, the box in points and the best matched name with its score; the box's accessible
  label says the same.
- **Side list and controls:** the names grouped by page; selecting one scrolls to its box,
  highlights it and shows its details. "Show" filters all, matched only or unmatched only; zoom is
  fit width, 100 %, 150 % or 200 %; "Box labels" hides the labels and keeps the boxes; "Download
  JSON" saves the raw response, byte for byte, as `extract-<request id>.json`.
- **Audit warnings:** the response has no page sizes, so each box is checked against the page size
  pdf.js reports; a box outside the page by more than 1 pt is marked with an audit warning in the
  list, its details and the status line (KNOWN_LIMITATIONS V3). A page whose frame the two renderers
  read differently gets a page audit warning (under the status, in its caption, its boxes' details
  and labels): a `/UserUnit` other than 1, which PDFium ignores, so the service's boxes are in user
  units and the page is drawn in that frame (V4), and a `/Rotate` that pdf.js reports as something
  other than a number multiple of 90 (V5).
- **Long documents:** pages are rendered only near the viewport and released when far from it, while
  the list and every box exist as soon as the response arrives; a 50-page PDF stays responsive
  ([benchmark record](BENCHMARKS.md) 20.4).
- **Privacy and safety:** the PDF is read in the browser and sent only to `/api/extract` of the same
  origin; the page stores nothing and loads nothing from another host (no CDN). It runs under a
  strict Content-Security-Policy (scripts, styles, images, workers and connections only from the
  service, no inline script or style, no framing), writes every value of the response as text, never as HTML
  (names come from OCR of untrusted documents), and runs pdf.js with WebAssembly off and no eval.
- **Access:** the page is unauthenticated like the API and published with it on `127.0.0.1` by
  default. `AUDIT_VIEWER_ENABLED=false` removes it: `/audit` and its files answer 404 and the API is
  unaffected.
- **Serving:** the page is at `/audit` (also `/audit/`), its files at `/audit/static/<name>` (only
  names listed in `app/web` with an allowed extension) and pdf.js at `/audit/vendor/pdfjs/<name>`;
  every route answers GET and HEAD, any other path under `/audit/` is a 404 (never a redirect). The
  page is `no-store`; its files are `no-cache` and pdf.js is cached for one day, each with an `ETag`
  answered by 304 when unchanged.

## pdf.js

The viewer uses pdf.js **6.4.299** (npm `pdfjs-dist`): the newest stable release at least two weeks
old when pinned, past CVE-2024-4367 and CVE-2026-16633. The pin lives in one file,
`deploy/pdfjs.lock`: the version, the tarball URL and its sha256, and the sha256 of each of the nine
files kept (the library, its worker, the two JavaScript image decoders pdf.js imports for JBIG2/CCITT
and JPEG 2000 with WebAssembly off, and five licences). `scripts/fetch_pdfjs.py` reads it, refuses any
mismatch and writes nothing unless every check passes; the Dockerfile's `pdfjs` stage runs the same
script, and the test and runtime images copy the files read-only to `/app/vendor/pdfjs`
(`PDFJS_DIR`), so nothing is fetched at run time. Browsers cache them for one day (their URL has no
version), then revalidate them by `ETag`.
To bump the version, follow the comment at the top of `deploy/pdfjs.lock` (tarball from
the npm registry, its sha512 checked, the new hashes, the advisory database), then rebuild and rerun
the suites. Licences: pdf.js Apache-2.0; the JBIG2 decoder comes from PDFium (BSD-3-Clause) and the
JPEG 2000 decoder from OpenJPEG (BSD-2-Clause). The five licence files are kept next to the code and
served at `/audit/vendor/pdfjs/LICENSE*`.
