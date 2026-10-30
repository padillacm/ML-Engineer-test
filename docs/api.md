# HTTP API

The request and response shapes are fixed by `app/models/schemas.py`; what this service adds to them is listed in [Decisions](../DECISIONS.md). With `EXPOSE_DOCS=true` the interactive documentation is served at `/docs` and the schema at `/openapi.json`.

## Endpoints

The audit viewer's page and files are under `/audit` ([Audit viewer](audit-viewer.md)).

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

## Uploads, limits and back-pressure

Uploads are judged by their content (a `%PDF-` header within the first 1,024 bytes), not by
their name or content type. At most `MAX_CONCURRENT_JOBS` extractions and ingestions run at
once. Questions have their own `MAX_CONCURRENT_QUESTIONS` slots. A request of either kind that
cannot start within `BUSY_WAIT_S` gets 503 `service_busy`; waiting requests hold no thread.
Readiness checks run on their own threads, with a `READINESS_TIMEOUT_S` timeout, and wait for no
slot. An extraction or ingestion whose client disconnects stops after the page in progress (logged
as 499 `request_cancelled`); a question runs to its end.
The derived sizes were measured on a 16-CPU host, with CPU quotas of 8 and 4
([benchmark record](BENCHMARKS.md) sections 14, 15, 17 and 18); re-measure on other hardware.

## Errors

Every error has the body `{"detail": "...", "code": "...", "request_id": "..."}`:

| Status | Codes |
|---|---|
| 400 | `empty_file`, `http_error` (a form the multipart parser refuses, e.g. a field over 1 MiB) |
| 413 | `too_large` (file over `MAX_UPLOAD_MB`, refused while reading or from `Content-Length`, or a body over the endpoint's limit: 64 KiB for `/api/ask` and other routes without uploads), `too_many_pages` (over `MAX_PAGES`, refused before any page is rendered) |
| 415 | `not_a_pdf` |
| 422 | `pdf_unreadable`, `invalid_names` (names the wrong entry, e.g. `names[1]`, including a name over 100 characters), `unsupported_language` (not en, es, pt, fr, de), `invalid_question` (blank), `invalid_request` (validation `detail` list of `{type, loc, msg}`; the rejected value is never echoed) |
| 429 | `rate_limited` (the client address is over its rate, burst or job cap; `Retry-After` in seconds) |
| 499 | `request_cancelled` (the client disconnected; logged only, no response reaches it) |
| 500 | `ocr_failed`, `ner_failed`, `embedding_failed`, `detector_failed`, `geometry_error`, `incompatible_embedding`, `invalid_configuration`, `internal_error` (anything unexpected; no internal detail is returned) |
| 503 | `vector_store_unavailable`, `llm_unavailable`, `service_busy`, with `Retry-After` (5 s; `BUSY_WAIT_S` rounded up for `service_busy`) |
| 504 | `vector_store_timeout`, `llm_timeout`, with `Retry-After: 5` |

## Request id and logs

Every response carries an `X-Request-ID` header: the client's own value when it is 1 to 64
letters, digits, `.`, `_` or `-`, otherwise a generated one. Logs go to stdout, one JSON object
per line (`LOG_FORMAT=text` for a readable line) with `timestamp`, `level`, `logger`,
`message` and, inside a request, `request_id`. Each request writes one access line (`method`,
route template, `status`, `duration_ms`; for a successful `/health` or `/health/ready` probe at
DEBUG, so the container's 15 s healthcheck does not flood the log) and each use case one outcome line of counts, ids and
durations; failures are logged with their `code` and `category`, unexpected exceptions with
their stack trace. Document text, names, questions, answers, file names and keys are never
logged.
