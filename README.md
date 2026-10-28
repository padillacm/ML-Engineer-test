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
| Models | `NER_MODEL_PATH` **R**, `EMBEDDING_MODEL_PATH` **R**, `EMBEDDING_MODEL_REVISION` **R**, `TORCH_THREADS` (2: torch intra-op threads, process-wide, shared by the recogniser and the embedder; the former `NER_THREADS`/`EMBEDDING_THREADS` stop start-up), `NER_POOL_SIZE` (8: texts recognised at once, sharing one model; start-up warns when `NER_POOL_SIZE` x `TORCH_THREADS` exceeds the CPUs available), `EMBEDDING_BATCH_SIZE` (16) |
| OCR | `OCR_DPI` (200), `OCR_WORKERS` (min(4, CPUs)), `OCR_TIMEOUT_S` (60), `OCR_FLATTEN` (true), `MAX_PAGES` (50) |
| Matching | `MATCH_THRESHOLD` (0.90) |
| Chunking | `CHUNK_SIZE` (800 characters), `CHUNK_OVERLAP` (0.12, a ratio of the size) |
| Retrieval | `RAG_THRESHOLD` (0.808), `RAG_TOP_K` (5), `RAG_MAX_CHUNKS` (3), `RAG_MAX_CONTEXT_CHARS` (6000) |
| Vector store | `QDRANT_URL` or `QDRANT_LOCATION` **R** (exactly one; `:memory:` is the embedded, non-persistent mode), `QDRANT_API_KEY`, `QDRANT_COLLECTION_PREFIX` (chunks), `QDRANT_TIMEOUT_S` (10), `QDRANT_ATTEMPTS` (3), `QDRANT_DEADLINE_S` (20) |
| Language model | `LLM_BASE_URL` (unset: no model), `LLM_MODEL` (required with a base URL), `LLM_API_KEY`, `LLM_CONNECT_TIMEOUT_S` (5), `LLM_READ_TIMEOUT_S` (60), `LLM_TOTAL_TIMEOUT_S` (90), `LLM_ATTEMPTS` (3), `LLM_MAX_TOKENS` (512) |
| Limits | `MAX_UPLOAD_MB` (20), `MAX_NAMES` (1000), `MAX_CONCURRENT_JOBS` (12), `MAX_CONCURRENT_QUESTIONS` (16), `BUSY_WAIT_S` (10, at most 300), `READINESS_TIMEOUT_S` (2, at most 30) |
| Logging | `LOG_LEVEL` (INFO), `LOG_FORMAT` (json, or text), `LOG_EXCEPTION_MESSAGES` (false; local debugging only: logs exception messages, which may contain document text) |

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
back. The test image already contains Tesseract and both models; the runtime image and a
`docker-compose.yml` that starts the service with Qdrant and a local model come in a later
block.

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
The defaults of 12 jobs and 8 NER pipelines were measured on a 16-CPU host
(`docs/BENCHMARKS.md` sections 14 and 15); re-measure on other hardware. Every error has the body
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
route template, `status`, `duration_ms`) and each use case one outcome line of counts, ids and
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

## Use of AI tools

Using AI assistants (Copilot, Claude, ChatGPT, agents) is **allowed and
expected** — that's how modern engineering works. Two conditions:

1. **Disclose it** in `DECISIONS.md` (what you used it for).
2. **Own every line.** In a follow-up conversation we will ask you to explain
   design choices, walk through your code, and make a live change. Code you
   can't reason about will not count in your favor.

We are evaluating *your* engineering judgment, not an agent's output. Submitting
something you don't understand is the one thing that fails this test.
