# Spec Delta

## Purpose

Defines the service's HTTP edge: the four endpoints of the fixed contract, readiness, how uploads and request bodies are validated, how core errors become status codes, how load is bounded, and how request resources are released.

## ADDED Requirements

### Requirement: Endpoints follow the fixed contract
The service SHALL expose `GET /health`, `POST /api/extract`, `POST /api/ingest` and `POST /api/ask` with the request and response shapes of `app/models/schemas.py`, unchanged. Optional additions SHALL be limited to an optional `language` form field on extract and an optional `document_id` response field on ingest. Each handler SHALL only parse the request, call one use case and translate its result.

#### Scenario: Liveness
- **WHEN** `GET /health` is called while the process is running
- **THEN** it answers 200 with `{"status": "ok"}` without calling any model or dependency

#### Scenario: Extraction
- **WHEN** a scanned PDF and `names='[{"first_name": "Maria", "last_name": "Gonzalez"}]'` are posted to `/api/extract`
- **THEN** the response is an `ExtractionResponse` whose boxes are in PDF points with 0-based pages and whose matches have scores between 0 and 1

#### Scenario: Ingestion reports duplicates
- **WHEN** the same PDF is posted to `/api/ingest` twice
- **THEN** the first response has `status` `indexed` and the second `already_indexed`, both with the same `chunks_stored` and `document_id`

#### Scenario: Question
- **WHEN** `{"question": "..."}` is posted to `/api/ask`
- **THEN** the response is a `RAGResponse` with the answer and the sources the use case returned

### Requirement: Readiness reports dependencies
`GET /health/ready` SHALL answer `{"status": "ready" | "not_ready", "checks": {...}}` reporting whether the vector store answers and whether a language model is configured, and SHALL answer 503 when the vector store does not answer, without contacting the language model.

#### Scenario: Vector store down
- **WHEN** the vector store cannot be reached
- **THEN** readiness answers 503 and names the failing check, while `/health` still answers 200

### Requirement: Uploads are validated by content
An uploaded `pdf_file` SHALL be rejected when it is missing, empty, or lacks the PDF header `%PDF-` within its first 1,024 bytes, whatever its file name or declared content type.

#### Scenario: Not a PDF
- **WHEN** a PNG renamed `scan.pdf` with content type `application/pdf` is uploaded
- **THEN** the response is 415 with code `not_a_pdf`

#### Scenario: Empty file
- **WHEN** a zero-byte file is uploaded
- **THEN** the response is 400 with code `empty_file`

#### Scenario: Damaged PDF
- **WHEN** a file with a PDF header that no reader can open is uploaded
- **THEN** the response is 422 with code `pdf_unreadable`

### Requirement: Upload size and page count are limited
Uploads larger than the configured maximum SHALL be rejected while the body is read, so they are never held in memory in full, and as soon as a declared `Content-Length` exceeds it. The upload routes SHALL allow the multipart parser's per-field limit plus 64 KiB of form envelope beyond the file limit. Bodies of routes without uploads, `/api/ask` included, SHALL be limited to 64 KiB. Documents with more pages than the configured maximum SHALL be rejected before any page is recognised.

#### Scenario: Oversized upload
- **WHEN** a file one byte larger than the maximum is uploaded, with or without `Content-Length`
- **THEN** the response is 413 with code `too_large` and no OCR runs

#### Scenario: Oversized question body
- **WHEN** a body over 64 KiB is posted to `/api/ask`, with or without `Content-Length`
- **THEN** the response is 413 with code `too_large` and nothing is embedded

#### Scenario: Too many pages
- **WHEN** a valid PDF with one page more than the maximum is uploaded
- **THEN** the response is 413 with code `too_many_pages` and no page is rendered

### Requirement: Request bodies are validated
`names` SHALL be a JSON array of objects with non-empty `first_name` and `last_name` of at most 100 characters each, at most the configured number of entries (an absent field means no names). `language`, when given and not blank, SHALL be one of the supported language codes (a blank value means no hint). The question SHALL be 1 to 2,000 characters and not only whitespace.

#### Scenario: Malformed names
- **WHEN** `names` is not JSON, not an array, or an entry lacks `last_name`
- **THEN** the response is 422 with code `invalid_names` and a message that says which entry is wrong

#### Scenario: Blank question
- **WHEN** the question is `"   "`
- **THEN** the response is 422 with code `invalid_question` and nothing is embedded or searched

#### Scenario: Unsupported language hint
- **WHEN** `language=it` is sent to `/api/extract`
- **THEN** the response is 422 with code `unsupported_language`

### Requirement: Errors map to status codes in one place
Every core error code SHALL map to exactly one HTTP status in a single table: invalid input to 422 except `empty_file` 400, `too_large` and `too_many_pages` 413 and `not_a_pdf` 415; `invalid_configuration`, processing failures and incompatible embedding to 500; dependency unavailable (including `service_busy`) to 503 and dependency timeout to 504, both with a `Retry-After` header.

#### Scenario: Every code has a status
- **WHEN** the mapping is checked against the core's closed list of codes
- **THEN** each code has a status, and adding a code without one fails the test suite

#### Scenario: Vector store down
- **WHEN** the vector store is unreachable during `/api/ingest`
- **THEN** the response is 503 with code `vector_store_unavailable` and a `Retry-After` header

#### Scenario: Language model timeout
- **WHEN** the language model does not answer within its deadline during `/api/ask`
- **THEN** the response is 504 with code `llm_timeout`

#### Scenario: No language model configured
- **WHEN** `/api/ask` reaches the model step and no language model is configured
- **THEN** the response is 503 with code `llm_unavailable`, while extraction and ingestion keep working

### Requirement: Error bodies are uniform and safe
Error bodies SHALL be `{"detail": <message>, "code": <code>, "request_id": <id>}`. Any exception that is not a core error SHALL produce 500 with code `internal_error` and a generic message, and SHALL be logged with its stack trace. Messages SHALL never contain secrets, document text or stack traces, and validation details SHALL NOT contain the rejected input.

#### Scenario: Unexpected failure
- **WHEN** a use case raises an exception that is not a core error
- **THEN** the response is 500 with code `internal_error` and the exception text is not in the body

#### Scenario: Validation does not echo input
- **WHEN** a 2,001-character question or a file sent as a plain form field is rejected with `invalid_request`
- **THEN** neither the question text nor the file bytes appear in the response body

### Requirement: Heavy work is bounded
At most a configured number of extraction and ingestion requests SHALL run at once; a request that cannot start within a configured wait SHALL be answered 503 with code `service_busy` and a `Retry-After` header. Questions SHALL have their own limit of the same kind, and health checks SHALL wait for neither.

#### Scenario: Saturated service
- **WHEN** the limit is 1, one extraction is running and a second arrives
- **THEN** the second is answered 503 `service_busy` after the wait, and the first completes normally

### Requirement: Request resources are released
Every uploaded file and temporary buffer SHALL be closed when its request ends, whether it succeeds or fails, and models and clients SHALL be created once per process and closed at shutdown.

#### Scenario: Many requests
- **WHEN** 200 requests, half of them failing validation, are served
- **THEN** the number of open file descriptors and temporary files returns to its level before them

### Requirement: Abandoned jobs stop early
When the client of an extraction or ingestion disconnects, the job SHALL stop after the page being recognised, free its slot, and be logged with code `request_cancelled`.

#### Scenario: Client disconnects during OCR
- **WHEN** the client of a 10-page extraction disconnects while page 2 is recognised
- **THEN** no page after the current one is recognised and the job slot is free again

### Requirement: Early rejections are readable
After an early 413, the service SHALL read and discard a bounded amount of the remaining body so that a client still sending can read the response.

#### Scenario: Slightly oversized upload
- **WHEN** a client sends a body a few kilobytes over the limit with its `Content-Length`
- **THEN** the client receives the 413 response instead of a connection reset

#### Scenario: Far oversized upload
- **WHEN** a client declares a body more than 1 MiB over the limit
- **THEN** the 413 is sent and the connection is closed without reading the body

#### Scenario: Slowly sent rejected body
- **WHEN** a client keeps sending a rejected body a few bytes at a time on a keep-alive connection
- **THEN** the connection is closed within 5 s of the 413
