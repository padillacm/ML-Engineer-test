# Proposal

## Why

`POST /api/extract` returns names with boxes in PDF points, but nothing lets a person check them: an auditor has to trust numbers like `x=72.4, y=118.9` without seeing the page. The take-home grades exactly this ("bounding boxes must be in PDF space, not image pixels"), and the service's own tests check boxes only against synthetic truth boxes. An audit viewer that draws the service's real response over an independent rendering of the PDF makes every box checkable by eye, and makes the PDF-space claim testable end to end: if the boxes were in image pixels, rotated or measured from the wrong origin or crop box, they would visibly miss the names.

## What Changes

- **Audit viewer page** served by the service at `GET /audit` (one HTML page plus its own static files, same origin, no CDN, no build step):
  - choose a PDF, optionally type the list of names to match (one "First Last" per line, or paste the JSON the API takes) and a language hint;
  - the page calls the existing `POST /api/extract` unchanged and shows the request id, duration and any error (`code` and message);
  - every page of the PDF is rendered in the browser with **pdf.js** (pinned version), with each extracted name drawn as a box over it; boxes of names that matched the list and names that did not are told apart by colour and by a label (not colour alone);
  - a side list of the names grouped by page; selecting a name scrolls to its box and highlights it; hovering or focusing a box shows the name, page, the box in points and its best match with the score;
  - zoom (fit width, 100 %, 150 %, 200 %), a filter "only matched / only unmatched", and a download of the raw JSON response;
  - pages are rendered lazily (only near the viewport), so a 50-page PDF stays responsive.
- **Coordinates**: a box in points is drawn at `point x scale` on the pdf.js viewport of the page as displayed (its `/Rotate` applied, relative to its crop box), the same frame `page-ocr` specifies. No derotation in the browser; rotated and cropped pages are part of the tests.
- **Security of the page itself** (the broader security pass stays deferred): a strict Content-Security-Policy on the page and its files (`script-src 'self'`, no inline script or style, `connect-src 'self'`, `frame-ancestors 'none'`), pdf.js with `isEvalSupported: false` and a version past the CVE-2024-4367 fix, every text from the response written as text (never as HTML), the PDF is read only in the browser and sent only to `/api/extract`.
- **Configuration**: `AUDIT_VIEWER_ENABLED` (default on); off, `/audit` and its files answer 404. Listed in `.env.example` and README.
- **pdf.js supply**: the pinned `pdfjs-dist` files (library and worker, plus its licence) are fetched by a Docker build stage with a sha256 check, like the models, and copied read-only into the test and runtime images; a small script does the same for a host run. (Alternative for the user to choose: vendor the files in the repository.)
- **Tests**:
  - Python: the route and its files, headers (CSP, `nosniff`, cache), 404 when disabled, the image contains the pinned pdf.js with the expected hash, architecture (the viewer lives in the HTTP layer and calls no use case directly).
  - Browser end to end in a pinned headless Chromium (Playwright, a separate `e2e` image stage and compose file, no new runtime dependency): upload the sample PDFs and synthetic PDFs with `/Rotate` 0/90/180/270, a crop box offset and mixed page sizes; assert that every drawn box sits at the response's points x scale within 1 px, and that the rendered ink inside each box is the name (dark-pixel coverage inside the box against a ring around it, threshold calibrated on the 60-document sample); screenshots kept as artefacts; keyboard navigation and labels checked.
- **Measurements** recorded in `docs/BENCHMARKS.md`: page load and first-page render time, time to draw all boxes for 1, 10 and 50 pages, browser memory for 50 pages, and the share of boxes on the 60-document sample that pass the ink check (with the failures inspected and classified).

## Capabilities

### New Capabilities

- `audit-viewer`: the browser page that draws the extraction response over the rendered PDF, its coordinate rule, interaction, safety rules and how it is enabled.

### Modified Capabilities

- `http-api`: a page route and its static files besides the four API endpoints; the HTTP contract (`app/models/schemas.py`) is unchanged.
- `deployment`: the images carry the pinned pdf.js files; image hygiene checks them.
- `service-configuration`: `AUDIT_VIEWER_ENABLED`.
- `page-ocr`: words carry the engine's confidence (follow-up of the ink calibration, design D9).
- `name-extraction`: a name whose every word was read with a confidence below 20 is not reported (D9).

## Impact

- New: `app/web/` (page, script modules, stylesheet; no framework), `app/api/viewer.py`, a `pdfjs` build stage, `scripts/fetch_pdfjs.py` (host), `tests/viewer/` (Python) and `tests/e2e/` (browser), `docker-compose.e2e.yml`.
- Changed: `Dockerfile`, `app/main.py` (router), `app/settings.py`, `.env.example`, `README.md`, `docs/BENCHMARKS.md`, `docs/KNOWN_LIMITATIONS.md`, `pyproject.toml` (an `e2e` dependency group with Playwright, used only by the `e2e` stage).
- Downloads (on approval): the `pdfjs-dist` package (about 5 MB unpacked, only the needed files kept) and one pinned Chromium build for the `e2e` stage (about 150 MB). Nothing is downloaded at run time.
- `app/models/schemas.py` and the four endpoints are unchanged.
