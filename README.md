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
uv sync --extra dev                          # create .venv and install deps from pyproject.toml
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
app/bootstrap.py  composition root: builds every adapter and use case once from the settings
app/main.py       create_app(): routers, handlers, middleware, lifespan; `app` is the Uvicorn target
app/settings.py   every tunable, read from the environment (and .env)
app/observability.py   JSON/text logs on stdout with the request id
app/models/schemas.py  the fixed API contract
tests/            unit and HTTP tests, in-memory fakes (tests/fakes), architecture guard
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
| Models | `NER_MODEL_PATH` **R**, `EMBEDDING_MODEL_PATH` **R**, `EMBEDDING_MODEL_REVISION` **R**, `TORCH_THREADS` (2, at most the CPUs available: torch intra-op threads, process-wide, shared by the recogniser and the embedder; the former `NER_THREADS`/`EMBEDDING_THREADS` stop start-up), `NER_POOL_SIZE` (derived*: texts recognised at once, sharing one model), `EMBEDDING_POOL_SIZE` (derived*: passage batches embedded at once, sharing one model; one more slot is reserved for questions' queries), `EMBEDDING_BATCH_SIZE` (16); start-up warns when a pool x `TORCH_THREADS` exceeds the CPUs available |
| OCR | `OCR_DPI` (200), `OCR_WORKERS` (derived*: Tesseract processes per document), `OCR_TIMEOUT_S` (60), `OCR_FLATTEN` (true), `MAX_PAGES` (50) |
| Matching | `MATCH_THRESHOLD` (0.90) |
| Chunking | `CHUNK_SIZE` (800 characters), `CHUNK_OVERLAP` (0.12, a ratio of the size) |
| Retrieval | `RAG_THRESHOLD` (0.808), `RAG_TOP_K` (5), `RAG_MAX_CHUNKS` (3), `RAG_MAX_CONTEXT_CHARS` (6000) |
| Vector store | `QDRANT_URL` or `QDRANT_LOCATION` **R** (exactly one; `:memory:` is the embedded, non-persistent mode), `QDRANT_API_KEY` or `QDRANT_API_KEY_FILE` (a file holding the key, as for the language model), `QDRANT_COLLECTION_PREFIX` (chunks), `QDRANT_TIMEOUT_S` (10), `QDRANT_ATTEMPTS` (3), `QDRANT_DEADLINE_S` (20) |
| Language model | `LLM_BASE_URL` (unset: no model), `LLM_MODEL` (required with a base URL), `LLM_API_KEY`, `LLM_API_KEY_FILE` (unset; a file holding the key, read once at start-up, an empty file meaning no key; not together with `LLM_API_KEY`; compose sets it to its secret), `LLM_CONNECT_TIMEOUT_S` (5), `LLM_READ_TIMEOUT_S` (60), `LLM_TOTAL_TIMEOUT_S` (90), `LLM_ATTEMPTS` (3), `LLM_MAX_TOKENS` (512), `LLM_REASONING_EFFORT` (unset: not sent), `LLM_MAX_TOKENS_FIELD` (`max_tokens`), `LLM_SEND_TEMPERATURE` (true) |
| Limits | `MAX_UPLOAD_MB` (20), `MAX_NAMES` (1000), `MAX_CONCURRENT_JOBS` (derived*), `MAX_CONCURRENT_QUESTIONS` (16), `BUSY_WAIT_S` (10, at most 300), `READINESS_TIMEOUT_S` (2, at most 30) |
| Logging | `LOG_LEVEL` (INFO), `LOG_FORMAT` (json, or text), `LOG_EXCEPTION_MESSAGES` (false; local debugging only: logs exception messages, which may contain document text) |

\* Derived at start-up when unset, from the CPUs available to the process (affinity and a
container's CPU quota, cgroup v2 or v1), and logged in one `concurrency in use` line; a set value
always wins.
The rule (`app/bootstrap.py` `concurrency_for`, with `TORCH_THREADS` T): jobs = 3/4 of the CPUs
rounded up, at most 12; NER pool = CPUs / T, at most 8; embedder pool = CPUs / (2 T), at most 4;
OCR workers = CPUs, at most 4; each at least 1. With T = 2:

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

Endpoints (interactive documentation at `/docs`, schema at `/openapi.json`):

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
| 499 | `request_cancelled` (the client disconnected; logged only, no response reaches it) |
| 422 | `pdf_unreadable`, `invalid_names` (names the wrong entry, e.g. `names[1]`, including a name over 100 characters), `unsupported_language` (not en, es, pt, fr, de), `invalid_question` (blank), `invalid_request` (validation `detail` list of `{type, loc, msg}`; the rejected value is never echoed) |
| 500 | `ocr_failed`, `ner_failed`, `embedding_failed`, `geometry_error`, `incompatible_embedding`, `invalid_configuration`, `internal_error` (anything unexpected; no internal detail is returned) |
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
`torch` is CPU-only on every platform: `pyproject.toml` takes it from the PyTorch CPU
wheel index (`[tool.uv.sources]`), so neither the host nor the image installs CUDA.

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
the review; warm-up) plus the model download (about 6 GB). The runtime image is 5.76 GB. Above 16
CPUs the derived values stay at the 16-CPU ones: scale with replicas.

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
`hygiene` checks that the runtime image has no tests, development packages, uv or `.env` files, runs as uid 10001
under tini, is offline for Hugging Face and cannot modify its own code or models. On the host the
smoke tests are skipped unless `SERVICE_URL` is set
(`SERVICE_URL=http://127.0.0.1:8000 uv run pytest -m deployed_service tests/deploy`).

## Use of AI tools

Using AI assistants (Copilot, Claude, ChatGPT, agents) is **allowed and
expected** — that's how modern engineering works. Two conditions:

1. **Disclose it** in `DECISIONS.md` (what you used it for).
2. **Own every line.** In a follow-up conversation we will ask you to explain
   design choices, walk through your code, and make a live change. Code you
   can't reason about will not count in your favor.

We are evaluating *your* engineering judgment, not an agent's output. Submitting
something you don't understand is the one thing that fails this test.
