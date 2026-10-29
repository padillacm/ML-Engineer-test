# Proposal

## Why

The core is complete and measured (OCR, NER, matching, ingestion, grounded answering), but nothing serves it: `app/api/*` and `app/main.py` are the original skeleton and `app/services/*` are stubs. The evaluation checks the HTTP edge directly: thin routers over abstractions, centralised environment-overridable configuration (no hardcoded hosts, model names, thresholds or DPI), upload validation by content (non-PDF, empty, oversized), validated request bodies, meaningful status codes for failures (bad upload, vector store down, LLM error), structured logging with context, no leaked handles or temp files, models loaded once, and at least one end-to-end path. Known gaps carried from block 4: a whitespace-only question raises a plain `ValueError` (it would become a 500), and the page limit agreed with the user (50) is not enforced anywhere.

## What Changes

- Archive `add-rag-pipeline` so its specs join the baseline.
- **Settings**: one `Settings` object read from the environment (and an optional `.env`) with `pydantic-settings`: model paths and revisions, Qdrant URL/key/timeouts, LLM base URL/model/key (secret), OCR DPI/workers/timeout, NER and embedding threads, matching and refusal thresholds, chunking, upload limits (`MAX_UPLOAD_MB=20`, `MAX_PAGES=50`, `MAX_NAMES`), concurrency, logging. Defaults are the benchmarked ones; invalid values stop the service at start-up with a message that names the variable and never shows a secret.
- **Composition root** (`app/bootstrap.py`): builds every adapter and use case once from `Settings` in the FastAPI lifespan and closes them on shutdown; the only module (with `app/main.py`) allowed to import adapters. `create_app(settings, container)` lets tests inject fakes.
- **Endpoints** (async handlers; blocking work on dedicated thread pools):
  - `GET /health` liveness `{"status": "ok"}`; `GET /health/ready` readiness (vector store reachable, LLM configured), 503 when not ready.
  - `POST /api/extract`: `pdf_file` + `names` (JSON list of `{first_name, last_name}`) + optional `language` hint (`en|es|pt|fr|de`) -> `ExtractionResponse`.
  - `POST /api/ingest`: `pdf_file` -> `IngestResponse` with `status` `indexed` or `already_indexed`, plus an optional `document_id` field (allowed by the contract; recorded in `DECISIONS.md`).
  - `POST /api/ask`: `RAGRequest` -> `RAGResponse`.
- **Validation at the edge**: PDF by content (`%PDF-` header within the first 1,024 bytes, as PDF readers accept), empty file, size limit enforced while reading (and early from `Content-Length`), `names` JSON parsed and validated against `NamePair` with a count limit, whitespace-only question rejected; page count enforced by the OCR adapter before any page is rendered (`too_many_pages`).
- **Error mapping**: one handler maps every core error code to an HTTP status and a uniform body `{"detail", "code", "request_id"}`; dependency failures are 503/504 with `Retry-After`; unexpected exceptions are 500 without internal details; request-validation errors keep FastAPI's 422 shape plus a code. A test fails if a code has no mapping.
- **Back-pressure**: a bounded number of heavy requests (extract, ingest) run at once; a request that cannot start within a short wait gets 503 `service_busy` with `Retry-After` instead of queueing without limit. The default comes from a concurrency measurement.
- **Observability**: JSON logs on stdout (stdlib `logging`, no new dependency), a request id per request (`X-Request-ID` accepted when well-formed, otherwise generated; echoed in the response), one access line per request (method, route, status, duration) and one outcome line per use case (pages, names found, chunks, refusal, top score, durations). Logs never contain document text, names, questions, answers or secrets.
- **Core changes**: `AskQuestion` raises `InvalidInput("invalid_question")` instead of `ValueError`; new codes `invalid_question` and `service_busy`; `TesseractOcrConfig.max_pages`.
- **Cleanup**: delete `app/services/`; lint and type-check `app/api`, `app/main.py`, `app/bootstrap.py`, `app/settings.py` strictly and include them in coverage; architecture guard: routers depend on use cases, never on adapters.
- **Tests**: HTTP tests with fakes on the host (every validation and error path, request id, log content); HTTP end-to-end tests in the container with the real engines, models and Qdrant (sample PDFs through `/api/extract`, `/api/ingest`, `/api/ask` with a scripted LLM).
- Dependencies: `pydantic-settings`, `python-multipart` (FastAPI needs it to read forms).

## Capabilities

### New Capabilities
- `http-api`: the four endpoints, readiness, upload and body validation, error-to-status mapping, back-pressure and resource handling at the HTTP edge.
- `service-configuration`: one environment-driven settings object, its defaults and start-up validation, and single construction of models and clients.
- `observability`: structured logs, request ids and what may and may not be logged.

### Modified Capabilities
- `grounded-answering`: a blank question is an invalid-input error.
- `page-ocr`: documents above a configurable page count are rejected before rendering.
- `architecture-boundaries`: the HTTP layer depends on use cases and core types only; adapters are built in the composition root.

## Impact

- New: `app/settings.py`, `app/bootstrap.py`, `app/observability.py`, `app/api/{errors,uploads,dependencies,middleware,health}.py`, `app/domain/uploads.py`; rewritten `app/main.py`, `app/api/extract.py`, `app/api/rag.py`; deleted `app/services/`.
- Changed: `app/domain/errors.py` (two codes), `app/application/ask_question.py`, `app/adapters/ocr/engine.py` (`max_pages`), `pyproject.toml`/`uv.lock`, `tests/architecture`, README (how to run the service).
- `app/models/schemas.py` stays byte-identical (hash-guarded); the optional `document_id` is added in a response subclass.
- Out of scope: the runtime image, `docker-compose.yml` and Ollama (block 6), the audit viewer (block 7), prompt-injection and further input hardening (security pass), authentication and rate limiting per client (documented as future work).
