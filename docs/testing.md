# Testing

The suite is split by what a test needs: the host runs everything that needs no external tool, the test image adds Tesseract and the real models, the compose files add a Qdrant server, a running stack and a browser. A marker on each test that needs something (`tesseract`, `ner_model`, `embedding_model`, `injection_model`, `qdrant_server`, `llm_server`, `deployed_service`, `e2e`, `cuda`) skips it, with the command to run it, where that is missing (`tests/conftest.py`).

## On the host

```bash
uv run pytest -q --cov
```

The coverage gate is 90 % of `app/` (98.5 % today). The pre-commit hooks run ruff, mypy, the lockfile check and the spec validation on each commit, and this suite and `uv audit` on each push. CI (`.github/workflows/ci.yml`) runs the same checks on every push and pull request.

## What the tests cover

| Folder | What |
|---|---|
| `tests/domain/` | Geometry, span rules, matching (with decoys), chunking, prompting, guardrails, injection patterns |
| `tests/application/` | The use cases over in-memory fakes, and the fakes against the port contracts |
| `tests/adapters/` | Each adapter, its config, contract tests shared with the fakes, tiny models for the torch adapters |
| `tests/api/` | Every endpoint, error body, limit, cancellation, back-pressure and the Uvicorn edge cases |
| `tests/integration/` | Real OCR, NER, embeddings and Qdrant on the sample PDFs and drawn pages with known boxes |
| `tests/adversarial/` | Prompt injections in five languages, hostile PDFs, rate-limit floods |
| `tests/architecture/` | The import rules between layers and the fixed HTTP contract |
| `tests/deploy/` | Compose wiring, secrets, hardening, `.env.example`, and the smoke tests of a running stack |
| `tests/e2e/` | The audit viewer in headless Chromium |

## Tests in the production-like container

The `Dockerfile` `test` stage (Python 3.12 on Debian 13, Tesseract 5.5 with
eng/spa/por/fra/deu) runs the full suite, including the tests marked `tesseract`
that are skipped on a host without the binary. With Docker running:

```
docker build --target test -t docintel-test .
docker run --rm docintel-test
```

Append `pytest -q --cov` to the run command for a coverage report. The git-history
cross-check in `tests/architecture` is skipped in the container (no `.git` in the image).
The build needs BuildKit (the default in Docker Desktop): the uv cache is a build cache
mount, so it is reused between builds but never stored in the image. Tesseract,
Leptonica, the language packs and `tini` (the container's PID 1, which reaps orphaned
processes) are pinned to exact Debian versions; the `Dockerfile` comment explains how
to bump them.

## Tests with Qdrant (compose)

`docker-compose.test.yml` runs the same `docintel-test` image next to a Qdrant server
(`qdrant/qdrant:v1.20.0`, pinned by digest, storage on tmpfs, no ports published). The
`tests` service sets `QDRANT_URL=http://qdrant:6333` and starts only once Qdrant answers
`/readyz`, so the tests marked `qdrant_server` (the `VectorStore` contract on a real server,
skipped on the host where `QDRANT_URL` is unset) run with the rest of the suite:

```
docker compose -f docker-compose.test.yml run --rm tests     # builds the image if missing
docker compose -f docker-compose.test.yml build tests        # rebuild after source changes
docker compose -f docker-compose.test.yml down -v            # remove the Qdrant container
```

On the host, the same contract runs against qdrant-client's embedded `:memory:` mode.

## Smoke tests against the running stack

```bash
docker compose -f docker-compose.yml -f docker-compose.smoke.yml run --rm smoke
```

```bash
docker compose -f docker-compose.yml -f docker-compose.smoke.yml run --rm hygiene
```

With `COMPOSE_PROFILES=gpu` set, both include the local model. `smoke` builds the test image,
starts the stack if needed, waits until the service is healthy and runs `tests/deploy` (marker
`deployed_service`) against it: health and readiness, extraction on `sample_pdfs/`, ingestion twice,
an answerable question (answered with a model, 503 `llm_unavailable` without one), an off-topic
question, an invalid upload and the request id. It ingests the sample PDFs into the stack's store.
`hygiene` checks that the runtime image has no tests, development packages, uv, `.env` files, Playwright or browser,
runs as uid 10001 under tini, is offline for Hugging Face, carries exactly the pdf.js files pinned in
`deploy/pdfjs.lock` (each with its hash) and cannot modify its own code, models or pdf.js files. On the host the
smoke tests are skipped unless `SERVICE_URL` is set
(`SERVICE_URL=http://127.0.0.1:8000 uv run pytest -m deployed_service tests/deploy`).

## Browser tests of the viewer (e2e)

```bash
docker compose -f docker-compose.yml -f docker-compose.e2e.yml run --rm e2e
```

`e2e` builds the `e2e` image stage, starts the stack if it is not running, waits until the service
is healthy and runs `tests/e2e` (marker `e2e`) in headless Chromium on the stack's internal network,
which has no route out. The suite (48 tests, about 50 s once the stack is up) sends the sample PDFs
and a synthetic PDF with `/Rotate` 90/180/270 pages, offset crop boxes, mixed page sizes, a
`/UserUnit 2` page and JPEG 2000 and CCITT G4 scans to `/api/extract` (nothing is ingested) and
checks every box's position against the response within 1 px at several zooms, every rendered
canvas against its page, the rendered ink inside each box (thresholds calibrated on the 60-document
sample, [benchmark record](BENCHMARKS.md) 20.3), the page audit warnings, selection, filter, zoom, keyboard,
labels, what is announced to screen readers, locked inputs and Cancel during a slow request, a
60-page document (canned response: lazy rendering, no growth over repeated extractions), error
display and odd error bodies, markup in names shown as text, revalidation with 304, no CSP violation
and no request other than the page, its files and `/api/extract`. The browser opens `http://docintel-app:8000/audit`, not `http://app:8000`: Chromium
sends every request for the host `app` over HTTPS (`app` is a top-level domain on its HSTS preload
list), so `docker-compose.e2e.yml` gives the service the network alias `docintel-app`
(KNOWN_LIMITATIONS V2).

Artefacts (screenshots, the synthetic frame PDF, the responses, `junit.xml`) go to the named volume
`e2e_artifacts`. To keep them on the host, mount a folder there instead (as written for a Linux or
macOS shell and for PowerShell; on Linux the folder must be writable by uid 10001):

```bash
docker compose -f docker-compose.yml -f docker-compose.e2e.yml run --rm -v "$PWD/e2e-artifacts:/e2e-artifacts" e2e
```

`docker compose -f docker-compose.yml -f docker-compose.e2e.yml --profile e2e down` stops the stack;
with `-v` it also deletes the artefacts volume and the stack's indexed documents
(`docintel_qdrant_data`); a plain `down -v` without `--profile e2e` keeps the artefacts volume
(`docker volume rm docintel_e2e_artifacts` removes it alone).

The image (`${TEST_IMAGE}-e2e`, default `docintel-test-e2e`) is test-only: the test image plus
Playwright 1.64.0 (dependency group `e2e`) and the Chrome Headless Shell that Playwright pins
(156.0.8078.4), fetched by the Dockerfile's `chromium` stage and checked against its sha256, with the
system libraries `playwright install-deps` adds through apt (not version-pinned, KNOWN_LIMITATIONS
V6); 6.80 GB, 1.05 GB over the test image. Neither the test nor the runtime image contains Playwright or a browser
(`hygiene` checks it), and the runtime image is unchanged by it. On the host the browser tests are
skipped (no Playwright); their helper tests (`tests/e2e/test_fixtures.py`) run with the rest of the
suite.

## GPU tests

Tests marked `cuda` compare the three model adapters on the GPU with the CPU and are skipped without a visible GPU. [Running the stack](operations.md) (section Torch builds and GPU tests) explains the second host environment they need and the CUDA test image.
