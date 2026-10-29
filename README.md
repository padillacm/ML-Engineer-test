# ML Engineer Take-Home — Document Intelligence Service

Build, from scratch, a small production-quality service that extracts person
names (with their locations) from scanned PDFs, fuzzy-matches them against a
provided list, and answers questions about ingested documents using RAG
(Retrieval-Augmented Generation).

This repository is an **empty skeleton**. There is no working code to fix — you
implement the whole thing. We care as much about *how* you build it (design,
tests, engineering practices, decisions) as about whether it runs.

## Why this exists (the problem)

Let's imagine a company that processes large volumes of scanned documents (contracts, KYC forms,
filings) that mention people. For **compliance**, it isn't enough to know *that*
a name appears — a reviewer has to be able to **find it on the page** and
**double-check that specific name against other records** (an ID, a sanctions
list, an internal system). A name with no location is not auditable.

So the core requirement is: **given a PDF, return each person's name together
with exactly where it sits on the page** (page number + bounding box). That
location is what lets a human or a downstream system verify the match against
other information. The fuzzy-matching and RAG pieces build on the same extracted
content. Keep this "locate it so it can be verified" purpose in mind — it's why
bounding-box accuracy and correct pages matter more than they might first seem.

---

## What you're building

A FastAPI service exposing four endpoints. The request/response **contract is
fixed** by `app/models/schemas.py` — do not change those shapes (add fields only
if you have a reason and document it).

### 1. `GET /health`
Liveness probe. Returns `{"status": "ok"}` (plus anything else useful).

### 2. `POST /api/extract`  — names + bounding boxes + fuzzy match
`multipart/form-data`:
- `pdf_file`: a (scanned) PDF.
- `names`: a JSON string — a list of `{"first_name": ..., "last_name": ...}`.

Pipeline:
1. **OCR** the PDF (it is scanned — there is no selectable text layer).
2. **NER**: extract **person** names only (no organizations, places, dates).
3. **Bounding boxes**: locate each extracted name in the page and return its box
   in **PDF coordinate space** (points, 72 per inch — *not* raw image pixels),
   with a **0-based page number**.
4. **Fuzzy match** the extracted names against the provided `names`.

Response: `ExtractionResponse` (see `schemas.py`) — `extracted_names[]` with a
`bounding_box`, and `fuzzy_matches[]`.

### 3. `POST /api/ingest` — index a document for RAG
`multipart/form-data` with `pdf_file`. OCR → chunk → embed → store in a vector
database. Response: `IngestResponse` (`status`, `chunks_stored`).

### 4. `POST /api/ask` — answer a question with RAG
JSON `{"question": "..."}`. Embed the question, retrieve the most relevant
chunks, call an LLM to answer **grounded in the retrieved context**, and return
`RAGResponse` (`answer`, `sources`).

---

## Requirements that matter (this is the actual evaluation)

> **Scope & time.** This is time-boxed — we do **not** expect fully isolated
> infrastructure. An **embedded / in-process vector store (e.g. Chroma) running
> locally is completely fine**, and you do not need Docker to pass. Things like a
> `docker-compose.yml` with an isolated vector DB, a hardened container, or a
> clean separation of services are **not required** — but if you add them, they
> count as **positive extras** in the evaluation. Prioritize a correct,
> well-structured core; reach for the infra polish only if you have time.

Anyone can wire libraries together. We're hiring an **ML Engineer**, so we look
for production judgment:

- **Architecture** — SOLID, dependency injection, separation of concerns.
  Services depend on abstractions (Protocols/ABCs), not concrete singletons.
  HTTP routers stay thin; business logic is testable without HTTP.
- **Configuration** — no hardcoded hosts/ports/model names/thresholds/DPI.
  Centralized, environment-overridable config.
- **Resilience** — timeouts and retries on external calls (LLM, vector store);
  meaningful failure modes; no silent `except: pass`.
- **Validation** — reject non-PDFs (by content, not just extension), empty
  files, oversized uploads; validate request bodies.
- **Resource management** — no leaked file handles or temp files.
- **Observability** — structured logging with useful context.
- **Correctness details we will check**, e.g.:
  - OCR must process **every** page (including the first).
  - Bounding boxes must be in **PDF space**, not image pixels.
  - Name matching must be **case-insensitive** and robust to OCR noise and
    accents (`María González` ~ `Maria Gonzalez`); prefer full-string
    similarity over substring tricks.
  - RAG chunking must not cut words in half; retrieval must be grounded (no
    hallucinated `sources`).
  - Vector-store IDs must be unique across documents (a second ingest must not
    overwrite the first).
- **Tests** — you write them. Cover the tricky logic above with unit tests
  (mock the heavy models / external services) and at least one end-to-end path.
  We look at *what* you chose to test and how.
- **Containerization (bonus, not required)** — if you have time, a `Dockerfile`
  and a `docker-compose.yml` that brings up the app **and its dependencies** with
  `docker compose up` (and `GET /health` passing) is a strong positive. Skipping
  it costs you nothing on the core; a local/embedded setup is fine.
- **Audit viewer (bonus, not required)** — a simple frontend that, given a PDF,
  **renders the pages and draws the returned bounding boxes over the detected
  names**. This is the compliance reviewer's tool: it's how a human confirms a
  name really is where the API says it is. Nothing fancy — a single self-
  contained page (e.g. vanilla JS + pdf.js) that calls `/api/extract` and
  overlays the boxes is plenty. Strong positive if included.
- **DESIGN.md** (≤ 3 pages) — component + sequence view, technology choices and
  trade-offs, how you'd scale to 1000+ PDFs/hour, failure modes and mitigations,
  and — if you were to route between models by query complexity — how.

---

## Suggested stack (substitute freely, but justify it in DESIGN.md)

- Python 3.12, FastAPI, `pytest`
- OCR: Tesseract via `pytesseract` + `PyMuPDF` (PDF → image)
- NER: spaCy (`en_core_web_sm`); handle non-English if you add it
- Fuzzy: `thefuzz`
- Embeddings: `sentence-transformers`
- Vector DB: **Chroma** (embedded/local — simplest) or Qdrant if you prefer a
  separate service
- LLM: any OpenAI-compatible endpoint (configurable base URL + key)

Dependencies live in `pyproject.toml` and are managed with **[uv](https://docs.astral.sh/uv/)** — change them as you see fit (`uv add ...`). Commit the resulting `uv.lock`.

---

## Deliverables

1. A **branch + pull request** against this repo with your implementation.
2. `DESIGN.md`.
3. A short **`DECISIONS.md`** — notable engineering decisions and trade-offs
   (bullets are fine). This is where your reasoning earns credit.
4. Tests (required). `Dockerfile` / `docker-compose.yml` are **optional bonus**.

## How we evaluate (weights)

| Criterion | Weight |
|---|---|
| Correctness — the pipeline works on the sample PDFs | 30% |
| Architecture & design (SOLID, DI, abstractions, thin routers) | 20% |
| Engineering practices (config, logging, resilience, validation) | 20% |
| Code quality & decisions | 15% |
| Tests you wrote (coverage of the tricky logic) | 10% |
| Design document | 5% |

**Positive extras (bonus, on top of the above):** an **audit viewer** that
overlays the bounding boxes on the PDF, containerization (`Dockerfile` /
`docker-compose.yml`), a truly isolated vector DB, multi-language support,
batching, or other production hardening. Not required — credited when present.

## Getting started

```bash
uv sync --extra dev --extra cpu             # create .venv: dev tools and the CPU build of torch
# Vector DB: Qdrant server (see "Tests with Qdrant (compose)" below); unit tests use an in-memory Qdrant.
uv run uvicorn app.main:app --reload         # needs the variables of "Running the service"
```

Sample scanned PDFs are in `sample_pdfs/`.

### Project layout

```
app/domain/       pure logic: geometry, name matching, chunking, uploads, errors (stdlib + rapidfuzz only)
app/application/  ports (typing.Protocol) and the use cases that depend on them
app/adapters/     real OCR / NER / embedding / vector-store / LLM implementations
app/api/          HTTP edge: thin routers, error mapping, middleware, form parsing (no adapters)
app/web/          the audit viewer page (HTML, ES modules, one stylesheet; served by app/api/viewer.py)
app/bootstrap.py  composition root: builds every adapter and use case once from the settings
app/main.py       create_app(): routers, handlers, middleware, lifespan; `app` is the Uvicorn target
app/settings.py   every tunable, read from the environment (and .env)
app/observability.py   JSON/text logs on stdout with the request id
app/models/schemas.py  the fixed API contract
tests/            unit and HTTP tests, in-memory fakes (tests/fakes), architecture guard;
                  tests/e2e: browser tests of the audit viewer (run in the e2e image only)
deploy/pdfjs.lock, scripts/fetch_pdfjs.py   the pdf.js pin of the audit viewer and its fetcher
docs/BENCHMARKS.md, benchmarks/   measurements behind the defaults and how to reproduce them
openspec/         specs and change proposals the implementation follows
```

Run the tests and the linter with `uv run pytest -q` and `uv run ruff check app tests`.
Install the git hooks once with `uv run pre-commit install` (see `.pre-commit-config.yaml`).
`tests/architecture` fails if the domain or application layers import a framework or vendor
library, if `app/api` imports an adapter or a vendor library, or if anything but
`app/bootstrap.py` imports an adapter.

### Running the service

All configuration comes from environment variables, read once at start-up by `app/settings.py`
(names are case-insensitive; an empty value counts as unset). A `.env` file in the working
directory is read too; it is git- and docker-ignored, and API keys belong there or in the
environment, never in a committed file. An invalid or missing value stops the service before
it accepts requests, with a message naming the variable and the rule (never its value).
Required variables are marked **R**; every default is the benchmarked value (`docs/BENCHMARKS.md`).

| Group | Variable (default) |
|---|---|
| Models | `NER_MODEL_PATH` **R**, `EMBEDDING_MODEL_PATH` **R**, `EMBEDDING_MODEL_REVISION` **R**, `TORCH_THREADS` (2, at most the CPUs available: torch intra-op threads, process-wide, shared by the recogniser and the embedder; the former `NER_THREADS`/`EMBEDDING_THREADS` stop start-up), `NER_POOL_SIZE` (derived*: texts recognised at once, sharing one model), `EMBEDDING_POOL_SIZE` (derived*: passage batches embedded at once, sharing one model; one more slot is reserved for questions' queries), `EMBEDDING_BATCH_SIZE` (16); start-up warns when a pool x `TORCH_THREADS` exceeds the CPUs available (CPU only) |
| Device | `DEVICE` (auto: the GPU when torch sees one, else the CPU; `cpu`; `cuda`, which stops start-up without a GPU): where the recogniser, the embedder and the detector run, logged in the `concurrency in use` line with the GPU's name and memory. On the GPU the embedder and the detector run in fp16 and the recogniser in fp32; the image must be the CUDA build (below) |
| OCR | `OCR_DPI` (200), `OCR_WORKERS` (derived*: Tesseract processes per document), `OCR_TIMEOUT_S` (60), `OCR_FLATTEN` (true), `MAX_PAGES` (50), `OCR_RENDER_TIMEOUT_S` (10: time budget of one page render, then `pdf_unreadable`), `OCR_MAX_IMAGE_PIXELS` (80,000,000: decoded image pixels a page may declare, checked before rendering, then `too_large`), `MAX_PAGE_CHARS` (30,000: longest recognised text of one page, then `too_large`; the densest sample page has about 1,000 characters) |
| Matching | `MATCH_THRESHOLD` (0.90) |
| Chunking | `CHUNK_SIZE` (800 characters), `CHUNK_OVERLAP` (0.12, a ratio of the size) |
| Retrieval | `RAG_THRESHOLD` (0.808), `RAG_TOP_K` (5), `RAG_MAX_CHUNKS` (3), `RAG_MAX_CONTEXT_CHARS` (6000), `RAG_MAX_CHUNKS_PER_DOCUMENT` (2: context chunks one document may give when chunks of other documents also matched) |
| Guardrails | `INJECTION_MODEL_PATH` (unset: no detector, every score 0; the images set `/models/prompt-guard` when built with the model), `INJECTION_THRESHOLD` (0.5: score at or above which a page's chunks are stored flagged and never retrieved, or a question is refused), `INJECTION_POOL_SIZE` (unset: 2 on the CPU, 1 on the GPU; concurrent page screenings on the one loaded model, plus one slot reserved for questions), `INJECTION_PATTERNS` (true: instruction patterns in the five languages, combined with the model's score by the higher one), `RAG_SUPPORT_MIN_SHARE` (0.6: share of an answer's content tokens that must appear in the context, else the fixed refusal) |
| Vector store | `QDRANT_URL` or `QDRANT_LOCATION` **R** (exactly one; `:memory:` is the embedded, non-persistent mode), `QDRANT_API_KEY` or `QDRANT_API_KEY_FILE` (a file holding the key, as for the language model), `QDRANT_COLLECTION_PREFIX` (chunks), `QDRANT_TIMEOUT_S` (10), `QDRANT_ATTEMPTS` (3), `QDRANT_DEADLINE_S` (20) |
| Language model | `LLM_BASE_URL` (unset: no model), `LLM_MODEL` (required with a base URL), `LLM_API_KEY`, `LLM_API_KEY_FILE` (unset; a file holding the key, read once at start-up, an empty file meaning no key; not together with `LLM_API_KEY`; compose sets it to its secret), `LLM_CONNECT_TIMEOUT_S` (5), `LLM_READ_TIMEOUT_S` (60), `LLM_TOTAL_TIMEOUT_S` (90), `LLM_ATTEMPTS` (3), `LLM_MAX_TOKENS` (512), `LLM_REASONING_EFFORT` (unset: not sent), `LLM_MAX_TOKENS_FIELD` (`max_tokens`), `LLM_SEND_TEMPERATURE` (true) |
| Limits | `MAX_UPLOAD_MB` (20), `MAX_NAMES` (1000), `MAX_CONCURRENT_JOBS` (derived*), `MAX_CONCURRENT_QUESTIONS` (16), `BUSY_WAIT_S` (10, at most 300), `READINESS_TIMEOUT_S` (2, at most 30) |
| Rate limits (per client address) | `RATE_LIMIT_UPLOADS_PER_MINUTE` (60) and `RATE_LIMIT_UPLOADS_BURST` (20) for `/api/extract` and `/api/ingest`, `RATE_LIMIT_QUESTIONS_PER_MINUTE` (240) and `RATE_LIMIT_QUESTIONS_BURST` (60) for `/api/ask`, `MAX_JOBS_PER_CLIENT` (4: extraction and ingestion jobs one address may hold at once); a rate or cap of 0 turns that limit off; over a limit the client gets 429 `rate_limited` with `Retry-After` |
| API docs | `EXPOSE_DOCS` (false: `/docs`, `/redoc` and `/openapi.json` answer 404; true serves them) |
| Audit viewer | `AUDIT_VIEWER_ENABLED` (true: the page at `/audit` and its files; false: they answer 404, the API is unaffected), `PDFJS_DIR` (unset: `vendor/pdfjs` at the project root, where `uv run python scripts/fetch_pdfjs.py` puts the pinned pdf.js files; the images set `/app/vendor/pdfjs`) |
| Logging | `LOG_LEVEL` (INFO), `LOG_FORMAT` (json, or text), `LOG_EXCEPTION_MESSAGES` (false; local debugging only: logs exception messages, which may contain document text) |

\* Derived at start-up when unset, from the CPUs available to the process (affinity and a
container's CPU quota, cgroup v2 or v1), and logged in one `concurrency in use` line; a set value
always wins.
The rule (`app/bootstrap.py` `concurrency_for`, with `TORCH_THREADS` T): jobs = 3/4 of the CPUs
rounded up, at most 12; NER pool = CPUs / T, at most 8; embedder pool = CPUs / (2 T), at most 4;
OCR workers = CPUs, at most 4; each at least 1. On the GPU the NER, embedder and detector pools
are 1 each instead (every size tried came within 3 %, `docs/BENCHMARKS.md` 23.5); jobs and OCR
workers keep the CPU rule, because OCR stays on the CPUs. With T = 2:

| CPUs | 1 | 2 | 4 | 6 | 8 | 12 | 16 or more |
|---|---|---|---|---|---|---|---|
| `MAX_CONCURRENT_JOBS` | 1 | 2 | 3 | 5 | 6 | 9 | 12 |
| `NER_POOL_SIZE` | 1 | 1 | 2 | 3 | 4 | 6 | 8 |
| `EMBEDDING_POOL_SIZE` | 1 | 1 | 1 | 1 | 2 | 3 | 4 |
| `OCR_WORKERS` | 1 | 2 | 4 | 4 | 4 | 4 | 4 |

The 4-, 8- and 16-CPU columns are measured (`docs/BENCHMARKS.md` 17.4 and 18); the others follow
the rule.

The image also sets `MALLOC_ARENA_MAX=2` (glibc), which keeps the service's memory flat under
load (`docs/BENCHMARKS.md` section 15); set it too when running the service outside Docker on Linux.

```bash
# .env (example values; never commit this file)
NER_MODEL_PATH=/models/xlm-roberta-base-ner-hrl
EMBEDDING_MODEL_PATH=/models/multilingual-e5-base
EMBEDDING_MODEL_REVISION=<commit of the e5 files>
QDRANT_URL=http://localhost:6333
LLM_BASE_URL=http://localhost:11434/v1     # any OpenAI-compatible server, e.g. Ollama
LLM_MODEL=qwen2.5:7b-instruct
# LLM_API_KEY=...                          # only if the server needs one

uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Models and clients are built once, when the application starts (not on the first request),
and closed when it stops; one process holds one copy of the models, so scale with processes or
replicas, not threads. Without `LLM_BASE_URL` the service still starts: `/api/ask` refuses
questions that match nothing and answers 503 `llm_unavailable` when it would need the model.
If Qdrant is down the service starts too (readiness reports it) and recovers when Qdrant is
back. The test image already contains Tesseract and both models; to run the service from the
runtime image with Qdrant and, optionally, a local model, see "Running the stack" below.

Endpoints (interactive documentation at `/docs`, schema at `/openapi.json`, the audit viewer at
`/audit`: see "Audit viewer" below):

```bash
# Liveness (never touches a dependency) and readiness (200, or 503 naming the failing check)
curl http://localhost:8000/health
curl http://localhost:8000/health/ready
# -> {"status":"ready","checks":{"vector_store":"ok","llm":"configured"}}

# Names, boxes (PDF points, 0-based pages) and fuzzy matches; `names` and `language` are optional
curl -F pdf_file=@sample_pdfs/meeting_minutes.pdf \
     -F 'names=[{"first_name": "Maria", "last_name": "Gonzalez"}]' \
     -F language=es \
     http://localhost:8000/api/extract

# Index a document: {"status": "indexed" | "already_indexed", "chunks_stored": n, "document_id": "..."}
curl -F pdf_file=@sample_pdfs/meeting_minutes.pdf http://localhost:8000/api/ingest

# Ask: {"answer": "...", "sources": ["exact stored chunk text", ...]}
curl -H 'Content-Type: application/json' -d '{"question": "Who signed the contract?"}' \
     http://localhost:8000/api/ask
```

Uploads are judged by their content (a `%PDF-` header within the first 1,024 bytes), not by
their name or content type. At most `MAX_CONCURRENT_JOBS` extractions and ingestions run at
once; a request that cannot start within `BUSY_WAIT_S` gets 503 `service_busy` (questions and
health checks never wait for this limit: waiting requests hold no thread, questions have their
own `MAX_CONCURRENT_QUESTIONS` slots (503 `service_busy` after `BUSY_WAIT_S`) and readiness checks
run on their own threads with a `READINESS_TIMEOUT_S` timeout). A job whose client disconnects
stops after the page in progress (logged as 499 `request_cancelled`).
The derived sizes were measured on a 16-CPU host, with CPU quotas of 8 and 4
(`docs/BENCHMARKS.md` sections 14, 15, 17 and 18); re-measure on other hardware. Every error has the body
`{"detail": "...", "code": "...", "request_id": "..."}`:

| Status | Codes |
|---|---|
| 400 | `empty_file`, `http_error` (a form the multipart parser refuses, e.g. a field over 1 MiB) |
| 413 | `too_large` (file over `MAX_UPLOAD_MB`, refused while reading or from `Content-Length`, or a body over the endpoint's limit: 64 KiB for `/api/ask` and other routes without uploads; a body up to 1 MiB over is read to its end so the client can read the 413), `too_many_pages` (over `MAX_PAGES`, refused before any page is rendered) |
| 415 | `not_a_pdf` |
| 429 | `rate_limited` (the client address is over its rate, burst or job cap; `Retry-After` in seconds) |
| 499 | `request_cancelled` (the client disconnected; logged only, no response reaches it) |
| 422 | `pdf_unreadable`, `invalid_names` (names the wrong entry, e.g. `names[1]`, including a name over 100 characters), `unsupported_language` (not en, es, pt, fr, de), `invalid_question` (blank), `invalid_request` (validation `detail` list of `{type, loc, msg}`; the rejected value is never echoed) |
| 500 | `ocr_failed`, `ner_failed`, `embedding_failed`, `detector_failed`, `geometry_error`, `incompatible_embedding`, `invalid_configuration`, `internal_error` (anything unexpected; no internal detail is returned) |
| 503 | `vector_store_unavailable`, `llm_unavailable`, `service_busy`, with `Retry-After` (5 s; `BUSY_WAIT_S` rounded up for `service_busy`) |
| 504 | `vector_store_timeout`, `llm_timeout`, with `Retry-After: 5` |

Every response carries an `X-Request-ID` header: the client's own value when it is 1 to 64
letters, digits, `.`, `_` or `-`, otherwise a generated one. Logs go to stdout, one JSON object
per line (`LOG_FORMAT=text` for a readable line) with `timestamp`, `level`, `logger`,
`message` and, inside a request, `request_id`. Each request writes one access line (`method`,
route template, `status`, `duration_ms`; for a successful `/health` or `/health/ready` probe at
DEBUG, so the container's 15 s healthcheck does not flood the log) and each use case one outcome line of counts, ids and
durations; failures are logged with their `code` and `category`, unexpected exceptions with
their stack trace. Document text, names, questions, answers, file names and keys are never
logged.

#### Tests in the production-like container

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

The NER model (`Davlan/xlm-roberta-base-ner-hrl`, about 1.1 GB) is baked into the image:
a separate `model` build stage installs a pinned `huggingface_hub`, downloads only the
files `transformers` loads (config, tokenizer files and `model.safetensors`) plus the model
card (`README.md`) at a pinned commit, checks every file's sha256, writes a `NOTICE` with the
licences (model AFL-3.0, base XLM-RoBERTa MIT) and is copied read-only to
`/models/xlm-roberta-base-ner-hrl` (`NER_MODEL_PATH`). The image sets `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`, so nothing is fetched at run time; to check that the whole suite
needs no network at all, run it with networking disabled:

```
docker run --rm --network none docintel-test
```

Tests marked `ner_model` need the model directory and are skipped on a host without it.
Loading refuses weights that do not fit the model's configuration (missing, unexpected or
mismatched tensors) instead of running with random weights. One text sent to the model is
capped at `max_chars` (default 50,000 characters): inference costs about 0.25 s per
1,000 characters at 2 threads and holds the recogniser's lock. To bump the model, set
`NER_MODEL_REVISION` and every `*_SHA256` build arg in the `Dockerfile` to the new commit and
its file hashes (the comment above the `model` stage says where to get them), rebuild and
rerun the suite.

The embedding model (`intfloat/multilingual-e5-base`, MIT, 768 dimensions, about 1.1 GB) is
baked in the same way: the `model` stage downloads `config.json`, `model.safetensors`,
`tokenizer.json`, `sentencepiece.bpe.model`, the two tokenizer configs and the model card
at a pinned commit (never `pytorch_model.bin` or the ONNX/OpenVINO variants), checks every
file's sha256, writes a `NOTICE` (model MIT, base XLM-RoBERTa MIT) and copies it read-only
to `/models/multilingual-e5-base`. The image sets `EMBEDDING_MODEL_PATH` to that directory
and `EMBEDDING_MODEL_REVISION` to the commit (part of the embedding identity, so a new
revision opens a new vector collection). It loads offline in about 5 s; one 800-character
passage (~180 tokens) takes about 0.16 s at 2 threads. To bump it, set the global
`E5_MODEL_REVISION` build arg and every `E5_MODEL_*_SHA256` arg in the `Dockerfile` (same
sources as the NER model), rebuild and rerun the suite.

The injection detector (`meta-llama/Llama-Prompt-Guard-2-86M`, about 1.1 GB) is baked in the
same way by the `prompt-guard` stage (config, `model.safetensors`, the tokenizer files, the
model card, `LICENSE` and `USE_POLICY.md` at a pinned commit, every sha256 checked, a `NOTICE`
with the licence) and copied read-only to `/models/prompt-guard` (`INJECTION_MODEL_PATH`). The
model is gated: the account that builds the image must have accepted the Llama 4 Community
License on its Hugging Face page, and a read token must be in `.env` as `HF_TOKEN` (never
committed). Compose hands it to the build as a BuildKit secret, so it is in no layer and in no
container (`docker history` and `docker inspect` show none). `INJECTION_MODEL=0` (in `.env` or
`--build-arg`) builds every image without the model; the service then runs with injection
detection off. To bump it, run `uv run python scripts/resolve_model_hashes.py
meta-llama/Llama-Prompt-Guard-2-86M <commit>` with `HF_TOKEN` in the environment, paste the
printed `ARG` lines over the `PROMPT_GUARD_*` ones in the `Dockerfile`, rebuild and rerun the
suite. Tests marked `injection_model` need the model directory and are skipped without it.

This service is **built with Llama**: Llama 4 is licensed under the Llama 4 Community
License, Copyright © Meta Platforms, Inc. All Rights Reserved (the `NOTICE` next to the model
carries the full attribution; the licence is not OSI-approved, see `docs/KNOWN_LIMITATIONS.md`).
`torch` comes in two builds, chosen by exactly one of two extras that uv declares conflicting:
`cpu` from the PyTorch CPU wheel index (no NVIDIA libraries) and `cuda` from the CUDA 12.6 index.
A sync needs one of them (`uv sync --extra dev --extra cpu` on the host; without either, no torch is
installed). The images take it from the `TORCH_VARIANT` build argument (`cpu` by default); in
compose, `TORCH_VARIANT` empty means `cuda` with `COMPOSE_PROFILES=gpu` and `cpu` otherwise, and
the CUDA build also reserves the GPU for the service (`docker-compose.torch.yml`). The test image
stays `cpu` unless `TORCH_VARIANT=cuda`. To run the GPU tests on the host, keep a second
environment beside `.venv` so the two builds never replace each other:

```bash
UV_PROJECT_ENVIRONMENT=.venv-cuda uv sync --extra dev --extra cuda
```

```bash
.venv-cuda/Scripts/python.exe -m pytest -m cuda
```

(`.venv-cuda/bin/python` on Linux.) Tests marked `cuda` compare the three adapters on the GPU with
the CPU and are skipped without a visible GPU; with `NER_MODEL_PATH`, `EMBEDDING_MODEL_PATH` and
`INJECTION_MODEL_PATH` set they also compare the real models. In the CUDA test image:
`TORCH_VARIANT=cuda docker compose -f docker-compose.test.yml run --rm tests pytest -m cuda`.

#### Tests with Qdrant (compose)

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

#### Defences of a service without authentication

The service is a proof of concept with no authentication, so the application itself bounds
what one client and one document can cost it, and what a document can make the language
model say (`openspec/specs/client-rate-limiting`, `injection-detection` and the deltas of the
other specs; design `add-defensive-hardening`):

- **Rate limits and caps per client address.** A token bucket per peer address and route
  class (uploads, questions) and a cap on the jobs one address may hold at once, inside the
  process (there is no proxy): over a limit the request is answered 429 `rate_limited` with
  `Retry-After` before its body is read. The defaults are above the measured single-client
  throughput, so a well-behaved client is never limited; `X-Forwarded-For` is ignored (without
  a proxy it would let a client choose its key), so behind a NAT or a proxy every client
  shares one address: raise the limits or set them to 0 there. Uvicorn runs with
  `--limit-concurrency 64` and `--timeout-keep-alive 5`; `/docs` and `/openapi.json` are off
  unless `EXPOSE_DOCS=true`; API responses carry `Cache-Control: no-store`,
  `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer`.
- **Bounded PDF rendering.** Each page is rendered through PDFium's progressive API under a
  time budget (`OCR_RENDER_TIMEOUT_S`): a page that takes longer fails the document with
  `pdf_unreadable` instead of holding the renderer (a hostile page that took 4.3 s to draw is
  stopped in about 70 ms over a 50 ms budget). Before rendering, the declared pixels of a
  page's images (nested form XObjects included) are summed and a page over
  `OCR_MAX_IMAGE_PIXELS` is refused `too_large` without decoding anything; a page whose
  recognised text exceeds `MAX_PAGE_CHARS` is refused `too_large` before NER, chunking or
  embedding run.
- **Injection detection.** With the model built in (`INJECTION_MODEL_PATH`), the text of
  every page is scored by Meta's Llama Prompt Guard 2 (86M, en/es/pt/fr/de, on the CPU)
  at ingestion and every question at answering. Every chunk of a page at or above
  `INJECTION_THRESHOLD` is stored flagged: the document stays complete (the ingest succeeds and
  reports every chunk; `chunks_flagged` is logged) but a flagged chunk is never returned by a
  search, so neither it nor its page reaches the model. The page is the unit because the
  context holds the whole parent page of a retrieved chunk. A flagged question gets the fixed
  refusal without a search or a model call. Next to the model, a rule detector
  (`INJECTION_PATTERNS`) matches instruction patterns in the five languages (ignore previous
  instructions, a new role, reveal the prompt, "whatever the documents say, answer that...",
  forged labels), tolerant of case, spacing and OCR substitutions; the higher of the two scores
  counts. On the adversarial suite the model alone flags 75 % of direct injections and about a
  third of those written into documents (none of the benign controls); with the patterns every
  case of the suite is flagged and the benign controls and the 2,776 sentences of the corpus
  generator still match nothing (the patterns were written against that suite, so those rates
  are a floor check, not a generalisation; `docs/BENCHMARKS.md` 22). Without the model the
  service runs with the patterns alone and says so at start-up.
- **The model sees passages as data.** Each context passage is enclosed between
  `<<<passage n: document <id>, page <p>>>>` and `<<<end passage n>>>` (a run of `<` or `>`
  inside the page text gets a zero-width space, so a document cannot forge a boundary) and
  the instructions say that anything inside a passage is document content, never an
  instruction. After the model answers, the answer is returned only if at least
  `RAG_SUPPORT_MIN_SHARE` of its content tokens (accent-folded, stop words removed) appear in
  the context (`unsupported` otherwise) and it neither quotes the instructions nor names a
  URL, a document or a page that is not in the context (`output_filtered`); the refusal
  reason and the share are logged, never the answer. When chunks of several documents pass
  the threshold, one document gives at most `RAG_MAX_CHUNKS_PER_DOCUMENT` of them.
- **Hardened containers.** Under compose the service and Qdrant run with a read-only root
  filesystem (scratch on tmpfs), every capability dropped, `no-new-privileges`, a PID limit
  and a memory limit; the smoke suite checks them on the running stack. `uv audit` runs on
  every push.

What these defences do not do: they do not authenticate anyone (a reverse proxy's job before
the service leaves the host), and a general-purpose language model is never immune to
injection: the detector, the delimiters and the output checks reduce the surface and make an
attack visible in the logs. `docs/SECURITY.md` summarises the threat model, the defences, their measured effect and the residual risk; `docs/KNOWN_LIMITATIONS.md` records what remains.

### Running the stack

The service runs from the `runtime` image (no tests or development tools, non-root, models baked in)
next to a Qdrant server whose data lives in a named volume. Only the service publishes a port, on
`127.0.0.1:8000` by default: it has no authentication, so put a reverse proxy with TLS and
authentication in front of it before exposing it beyond the host.

Configuration is optional: `cp .env.example .env` and uncomment what you change. Every setting is
listed there with its default; `.env` is git- and docker-ignored and never enters an image. Compose
reads `.env` and passes the service only the settings it reads (and its keys as secret files), so
other variables in `.env` never reach the container. Put a value that contains `$` in single
quotes (`LLM_API_KEY='abc$def'`): compose reads `$def` in an unquoted or double-quoted value as a
variable and silently replaces it.

#### Without a GPU

```bash
docker compose up -d --wait
```

The first run builds the image; the command returns when the service is healthy
(`curl http://127.0.0.1:8000/health/ready`). Extraction and ingestion work. With no language model
configured, readiness reports `"llm": "not_configured"` and questions that need a model answer 503
`llm_unavailable` (questions the documents cannot answer are still refused with 200).

#### With a hosted OpenAI-compatible model

Uncomment the hosted block in `.env` and fill it in, and start the stack without the `gpu` profile
(`LLM_MODEL` names the model of both, so with the profile `ollama-pull` stops with an error
instead of trying to pull the hosted model's name):

```dotenv
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=<model name>
LLM_API_KEY='<your key>'
```

then `docker compose up -d --wait`. The key stays in `.env` only: compose hands it to the service as
a secret file (`/run/secrets/llm_api_key`, read through `LLM_API_KEY_FILE`), so it is not in the
container's environment (`docker inspect` shows `LLM_API_KEY=` empty), not in `docker compose config`
and never in the logs. Without a key the service gets an empty file and runs without one. Anyone who
can use the Docker socket can still read the file inside the container. Outside compose,
`LLM_API_KEY` works as before. `QDRANT_API_KEY` in `.env` takes the same path
(`/run/secrets/qdrant_api_key`), and the stack's Qdrant then requires it.

Reasoning models (for example `gpt-6-luna`) reject the request's `max_tokens` and its temperature 0
with HTTP 400; for them also set

```dotenv
LLM_MAX_TOKENS_FIELD=max_completion_tokens
LLM_SEND_TEMPERATURE=false
```

(the model then runs at its default temperature, so answers are not deterministic).
`LLM_REASONING_EFFORT` (`none` ... `xhigh`) is sent only when set. A rejected request answers 503
`llm_unavailable`, and the service logs `LLM request failed (status 400, unsupported_parameter)`:
the status and the provider's error code, never its message. A reply cut at the token limit with
no content (a thinking model that spent `LLM_MAX_TOKENS` reasoning) is logged as a warning with the
`model` and `max_tokens` fields.

#### With a local model on an NVIDIA GPU (Ollama)

```bash
COMPOSE_PROFILES=gpu docker compose up -d --wait
```

(PowerShell: `$env:COMPOSE_PROFILES = "gpu"; docker compose up -d --wait`; or put
`COMPOSE_PROFILES=gpu` in `.env`, after which `docker compose up -d --wait` is enough.) This also
starts Ollama with one NVIDIA GPU, downloads `LLM_MODEL` into the `ollama_models` volume unless it
is already there (a warm volume starts offline), runs one short prompt to load it and compile its
GPU kernels (cached in the same volume), and only then starts the service, pointed at Ollama. The
first start therefore takes a while (download plus about a minute of warm-up); later starts reuse
the volume. If the download fails (wrong model name, full volume, no network) or the model is
missing, `up` fails and the service is not started; `docker compose logs ollama-pull llm-warmup`
says why. `--profile gpu` on its own is refused with a message: compose does not show the command
line's profiles to the file, so it could not point the service at Ollama. Context length, parallel
requests and keep-alive are `OLLAMA_CONTEXT_LENGTH`, `OLLAMA_NUM_PARALLEL` and `OLLAMA_KEEP_ALIVE`.

The service reaches Ollama at a fixed address of the internal network (`10.213.47.250`), never by
name, so no question or document text can be sent to a host that answers a DNS lookup for
`ollama` while Ollama is down: questions then answer 503 `llm_unavailable` until it is back.
Ollama itself has no route out; the download runs in the separate `ollama-pull` container. If
compose reports that the network `networks have overlapping IPv4` (another network uses
`10.213.47.0/24`, or a second copy of the stack runs), set `BACKEND_NETWORK_PREFIX` (for example
`10.213.48`) in `.env`.

Requirements: Docker Desktop with WSL2 GPU support on Windows, or on Linux the NVIDIA driver plus the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
configured for Docker (`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`).
Check with `docker run --rm --gpus all ubuntu nvidia-smi`. Without a usable GPU, run the stack without
the profile.

#### Sizing

Measured on the deployed stack (`docs/BENCHMARKS.md` sections 17-19): one host (Ryzen 7 7700X, 16
threads, 32 GB, RTX 4070 Ti 12 GB); 8 and 4 CPUs are CPU quotas on that host (`cpus:`), not smaller
machines; short synthetic documents (1.8 pages per extraction, 1.7 per ingestion); every setting at
its default, so the concurrency is derived from the CPUs.

| CPUs available to the service | 16 | 8 | 4 |
|---|---|---|---|
| Derived jobs / NER pool / embedder pool / OCR workers | 12 / 8 / 4 / 4 | 6 / 4 / 2 / 4 | 3 / 2 / 1 / 4 |
| `/api/extract`, 16 clients: docs/min, p95 | 84.0, 13.3-14.0 s | 62.0, 16.9 s | 37.3, 26.9-27.0 s |
| `/api/ingest`, 16 clients: docs/min, p95 | 90.7, 12.9-13.2 s | 65.9, 16.4-16.7 s | 38.1, 27.9 s |
| Question, `gpu` profile (e4b): p50 / p95 alone | 0.60 / 1.84 s | 0.56 / 1.86 s | not measured |
| ... while 16 clients ingest | 0.84 / 2.31 s | 0.55 / 1.99 s | not measured |
| Service memory peak (extract / ingest) | 2.6 / 3.0 GB | 2.3 / 2.7 GB | 2.2 / 2.5 GB |

With the `gpu` profile, 16 concurrent questions give 128-136 questions/min at p95 9.6-9.7 s
(16 CPUs); `gemma4:e4b-it-qat` with an 8,192-token context and 2 parallel requests uses 5.9-6.5 GB of
VRAM (device total, including about 1.2 GB taken by the desktop). Start (`up -d --wait`): 13.4-13.6 s
without the profile, 15.8-19.4 s with it once the model and its compiled GPU kernels are in the
volume (17.7-17.9 s after the review's changes), about 85 s on the very first start (79.5 s after
the review; warm-up) plus the model download (about 6 GB). The runtime image is 5.76 GB (the audit
viewer's page and pdf.js add 3.2 MB; the browser tests' image is separate). Above 16
CPUs the derived values stay at the 16-CPU ones: scale with replicas.

#### Models on the GPU

With `COMPOSE_PROFILES=gpu` the service is the CUDA build and `DEVICE=auto` puts the recogniser,
the embedder and the detector on the GPU beside Ollama. Measured on the same host with every
defence on (`docs/BENCHMARKS.md` section 23, 60-document sample, 16 CPUs):

| | CPU build | CUDA build, models on the GPU |
|---|---|---|
| `/api/ingest`: docs/min sequential / 12 clients | 9.7 / 35.3 | 20.0 / 132.9 |
| `/api/extract`: docs/min sequential / 12 clients | 15.4 / 86.0 | 19.3 / 127.9 |
| Detector per page / per question | 1,541 / 128 ms | 94 / 19 ms |
| Embedder per page / per query | 551 / 50 ms | 16 / 5.6 ms |
| Recogniser per page (fp32 on both) | 563 ms | 102 ms |
| Question p50 / p95 under 12-client ingestion (Gemma on the GPU) | 1.0 / 2.0 s | 0.9 / 1.6 s (0.6 / 1.1 s alone) |
| VRAM: the three models in service / with Gemma, worst case | none | 2.7-2.9 GB / 9.4 of 12.3 GB |
| Image / service RAM / start to ready | 7.89 GB / 1.2 GB / 8.0-9.8 s | 17.2 GB / 2.2 GB / 9.7-12.0 s |

OCR, which stays on the CPUs, is then 94 % of the ingestion time and the GPU averages 7-16 %
busy, which is why the pool sizes no longer matter there and turning the detector or the
patterns off moves throughput by 0-5 %. The worst case leaves 2.9 GB of the 12 GB free with a desktop on
the GPU. The results equal the CPU build's (the recogniser's spans on all 1,030 corpus pages,
extraction and ingestion bodies on the sample, the detector's flags; e5 vectors within a cosine
of 0.99997). `DEVICE=cpu` makes the CUDA build behave like the
CPU one.

#### Storage and failure modes

- `qdrant_data` holds the index: each chunk is a 768-dimension float vector (3 KB) plus its text
  (up to 800 characters) and payload, roughly 5-10 KB per chunk (an estimate, not measured; the
  short test documents give about 5 chunks each), plus Qdrant's write-ahead log, which needs room
  of its own: on a 40 MB volume the first write already failed ("WAL buffer size exceeds available
  disk space"). When the volume is full, ingestion answers 503 `vector_store_unavailable` with "the
  vector store has no disk space left" and the service logs that line; readiness stays 200, because
  Qdrant itself is up.
- `ollama_models` holds the model (`gemma4:e4b-it-qat`: 6.1 GB) and the compiled GPU kernels (at
  most 4 GiB, `CUDA_CACHE_MAXSIZE`): plan about 11 GB, more for other models. When it is full the
  download fails and the stack does not start (above).
- Qdrant down: the service starts, readiness answers 503 naming `vector_store`, requests that need
  the store answer 503 and recover when it is back. Ollama down: questions answer 503
  `llm_unavailable` and recover when it is back.
- After a host reboot Docker restarts the containers in any order and without the start-up
  check; questions answer 503 until Ollama is up.

#### Stopping

```bash
docker compose --profile gpu down
```

This keeps indexed documents and downloaded models; add `-v` to delete them too (volumes
`docintel_qdrant_data`, `docintel_ollama_models`). A stop gives running requests up to 30 s to finish.

#### Smoke tests against the running stack

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

### Audit viewer

`/audit` is a page for checking every box of a real `/api/extract` response by eye. It sends the
chosen PDF to `POST /api/extract` like any client, renders the same file in the browser with pdf.js
(an independent renderer, not the service's PDFium) and draws each returned box at its points x the
zoom scale over the page as displayed (`/Rotate` applied, relative to the crop box, top-left origin,
no correction in the browser). A box in image pixels, measured from the wrong origin or on a wrong
frame visibly misses its name.

With the stack running (`docker compose up -d --wait`; the images carry pdf.js), open
`http://127.0.0.1:8000/audit`. On the host, fetch the pinned pdf.js files once with
`uv run python scripts/fetch_pdfjs.py` (into `vendor/pdfjs`, git-ignored), start the service as in
"Running the service" and open the same address. Without the files the page says so and the pdf.js
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
  (`docs/BENCHMARKS.md` 20.4).
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

#### pdf.js

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

#### Browser tests of the viewer (e2e)

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
sample, `docs/BENCHMARKS.md` 20.3), the page audit warnings, selection, filter, zoom, keyboard,
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

## Use of AI tools

Using AI assistants (Copilot, Claude, ChatGPT, agents) is **allowed and
expected** — that's how modern engineering works. Two conditions:

1. **Disclose it** in `DECISIONS.md` (what you used it for).
2. **Own every line.** In a follow-up conversation we will ask you to explain
   design choices, walk through your code, and make a live change. Code you
   can't reason about will not count in your favor.

We are evaluating *your* engineering judgment, not an agent's output. Submitting
something you don't understand is the one thing that fails this test.
