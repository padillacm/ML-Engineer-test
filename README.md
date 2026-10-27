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
# Vector DB (planned): Qdrant, via docker compose; tests use an in-memory instance.
uv run uvicorn app.main:app --reload
```

Sample scanned PDFs are in `sample_pdfs/`.

### Project layout (work in progress)

```
app/domain/       pure logic: geometry, name matching, chunking, errors (stdlib + rapidfuzz only)
app/application/  ports (typing.Protocol) the use cases depend on
app/adapters/     real OCR / NER / embedding / vector-store / LLM implementations
app/api/, app/models/schemas.py   HTTP edge and the fixed API contract
tests/            unit tests, in-memory fakes (tests/fakes), architecture guard
docs/BENCHMARKS.md, benchmarks/   measurements behind the defaults and how to reproduce them
openspec/         specs and change proposals the implementation follows
```

Run the tests and the linter with `uv run pytest -q` and `uv run ruff check app tests`.
Install the git hooks once with `uv run pre-commit install` (see `.pre-commit-config.yaml`).
`tests/architecture` fails if the domain or application layers import a
framework or vendor library.

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

## Use of AI tools

Using AI assistants (Copilot, Claude, ChatGPT, agents) is **allowed and
expected** — that's how modern engineering works. Two conditions:

1. **Disclose it** in `DECISIONS.md` (what you used it for).
2. **Own every line.** In a follow-up conversation we will ask you to explain
   design choices, walk through your code, and make a live change. Code you
   can't reason about will not count in your favor.

We are evaluating *your* engineering judgment, not an agent's output. Submitting
something you don't understand is the one thing that fails this test.
