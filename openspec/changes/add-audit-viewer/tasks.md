# Tasks

## 1. Baseline

- [x] 1.1 Archive `add-deployment`; verify `openspec validate --all --strict` passes with 14 specs

## 2. pdf.js supply

- [x] 2.1 Choose the `pdfjs-dist` version (D2: newest stable at least two weeks old, past the CVE-2024-4367 fix), record version and tarball sha256; add the `pdfjs` build stage and `scripts/fetch_pdfjs.py`; verify the hash check fails on a tampered file and the files land read-only in the test and runtime images

## 3. Serving

- [x] 3.1 `app/api/viewer.py` and `AUDIT_VIEWER_ENABLED` (D5): page, static and vendor routes from fixed directories, headers, 404 when disabled or unknown; tests for routes, headers, traversal attempts, disabled flag, no inline script or style in the page, architecture boundaries

## 4. Page

- [x] 4.1 `app/web/` page, modules and stylesheet (D3, D4): form, names parsing with the JSON preview, call to `/api/extract`, error display, pdf.js rendering with `isEvalSupported: false`, box elements in the displayed frame, side list, selection, filter, zoom, details, raw JSON download, lazy rendering; text-only insertion of response values; verify manually in the browser on the sample PDFs

## 5. Browser end-to-end tests

- [x] 5.1 `e2e` image stage and `docker-compose.e2e.yml` (D7): pinned Playwright and Chromium; position check at two zoom levels; behaviour checks (selection, filter, zoom, keyboard, labels, errors); CSP violations and outgoing requests recorded
- [x] 5.2 Frame tests: synthetic PDFs with `/Rotate` 90/180/270, an offset crop box and mixed page sizes; the ink check passes on every box

## 6. Calibration and measurements

- [x] 6.1 Ink check on the 60-document sample (D8 item 2): set the threshold, record the pass share, inspect and classify every failure; fix fixable causes
- [x] 6.2 Performance (D8 items 4 and 5): load, first render, all boxes for 1/10/50 pages, memory for 50 pages, render time per page at 100 % and 200 %, image sizes; record in `docs/BENCHMARKS.md`

## 7. Documentation

- [x] 7.1 README (how to open the viewer, with and without Docker, what the line styles and labels mean, the setting), `.env.example`; unfixable findings in `docs/KNOWN_LIMITATIONS.md`

## 8. Review and wrap-up

- [x] 8.1 Independent review (coordinate correctness on unusual PDFs, XSS through names and errors, CSP, path handling, memory on long documents, accessibility, behaviour when the API is slow or fails); fix every fixable finding with tests
- [x] 8.2 Final verification: host `pytest --cov`, ruff, mypy, all pre-commit hooks, `uv-audit`, `openspec validate --all --strict`, full compose test suite, smoke and hygiene against the stack, the e2e suite

## 9. Low-confidence OCR names (follow-up of 6.1)

- [x] 9.1 Paired comparison on the 600-document corpus of a minimum word confidence before recognition and of rules on candidate names (D9); adopt only a variant whose recall loss CI includes 0
- [x] 9.2 Words carry their confidence (`page-ocr`), the Dconf rule (`name-extraction`), tests; end-to-end re-run of the production chain on the 600 documents with the image under test
- [x] 9.3 `docs/BENCHMARKS.md` section 21 and sections 11, 20.3 and 20.6 (the README does not describe the span rules: unchanged); what the confidence cannot fix in `docs/KNOWN_LIMITATIONS.md`
- [x] 9.4 Verification: host gates, compose test suite, smoke and hygiene, the e2e suite
