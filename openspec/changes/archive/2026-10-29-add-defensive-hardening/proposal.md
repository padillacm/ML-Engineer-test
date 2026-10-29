# Proposal

## Why

The service is a proof of concept exposed without authentication (out of scope for now), so the application itself is the only thing standing between a client and its CPU, its vector store and its language model. Two of its documented limitations are security gaps: a hostile PDF page can hold the PDFium lock for 33 s (`docs/KNOWN_LIMITATIONS.md` P5) and document text reaches the model verbatim, so instructions hidden in a scanned document steer the answer (R3). The deployment block deferred "security hardening of ingestion, prompts and retrieval" to this pass.

## What Changes

- **Client rate limiting and per-client job caps** at the HTTP edge, in process (no proxy in the POC): a token bucket per client address and route class (uploads, questions), 429 `rate_limited` with `Retry-After`, and a cap on the jobs one client may hold at once. API documentation (`/docs`, `/openapi.json`) off by default behind a setting; `Cache-Control: no-store`, `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer` on API responses; Uvicorn started with a connection limit and a short keep-alive timeout.
- **Bounded PDF rendering and page text**: each page render runs under a time budget through PDFium's pausable (progressive) rendering, so a hostile page fails with `pdf_unreadable` after the budget instead of holding the lock; the recognised text of a page is capped (`MAX_PAGE_CHARS`) and a denser page is refused as invalid input (`too_large`) before NER, embedding or chunking, which also bounds the chunks of a document. A corpus of hostile PDFs becomes regression tests.
- **Injection guardrails for the RAG chain**:
  - a new `InjectionDetector` port with an optional adapter for Meta's Llama Prompt Guard 2 (86M, multilingual, CPU, in process; `INJECTION_MODEL_PATH` unset means no detector) that scores chunks at ingestion and questions at answering;
  - a flagged chunk is stored with a `flagged` mark and never enters a search result (the ingest succeeds; the count of flagged chunks is logged);
  - a flagged question is answered with the fixed refusal without calling the model;
  - the context passages are delimited and the system prompt declares them data, never instructions;
  - an answer whose content is not supported by the context, or that leaks the system prompt or names documents, pages or URLs that are not in the context, is replaced by the fixed refusal;
  - at most a configurable number of context chunks come from one document when other documents also passed the threshold.
- **Container hardening**: the app and Qdrant containers run with a read-only root filesystem (tmpfs for scratch), all capabilities dropped, `no-new-privileges`, a PID limit and a memory limit; the dependency audit hook runs on push, not only on request.
- **Prompt Guard model in the image**: a `prompt-guard` model stage like the NER and embedding ones (pinned revision, sha256-checked files, NOTICE with the Llama 4 Community License attribution), downloaded with a build-time Docker secret (`HF_TOKEN` from `.env`, never in a layer) and skippable (`--build-arg INJECTION_MODEL=0`) so the stack still builds and runs without a token, with the detector unconfigured.
- **Adversarial suite and measurements**: direct and indirect injection cases in the five languages (including OCR-noisy text), hostile PDFs, limiter load; detector precision and recall by threshold on adversarial and benign text, the prompt change re-measured on the answer-quality sample, the cost per page and per question, all recorded in `docs/BENCHMARKS.md`; `docs/KNOWN_LIMITATIONS.md` R3 and P5 updated.

Out of scope: authentication, TLS and a reverse proxy (still the edge's job); moving the models to the GPU (its own change); content moderation of answers (the service answers from its own documents).

## Capabilities

### New Capabilities

- `client-rate-limiting`: how the service limits what one client address may ask of it over time and at once, without authentication, and how it answers a client that exceeds a limit.
- `injection-detection`: the contract of the injection detector (scores for texts, unavailable means no detection), its adapter for Prompt Guard 2, its configuration and its cost bounds.

### Modified Capabilities

- `http-api`: API documentation is served only when enabled; API responses carry no-store and nosniff headers; the server bounds connections and keep-alive.
- `page-ocr`: a page render has a time budget; a page whose recognised text exceeds the configured maximum is refused.
- `document-ingestion`: chunks are screened and flagged chunks are stored but never retrievable; the flagged count is reported in the outcome line.
- `vector-storage`: search excludes flagged chunks; the flag is part of the chunk payload.
- `grounded-answering`: flagged questions are refused without the model; passages are delimited as data; unsupported or leaking answers become the fixed refusal; context diversity across documents.
- `service-configuration`: the detector, limiter, render budget, page text cap and documentation settings are tunables with defaults; the detector is optional.
- `observability`: guardrail outcomes (flagged chunks, flagged questions, unsupported answers, rate limits) are logged by count and reason, never by content.
- `deployment`: hardened container runtime options; the Prompt Guard model stage with its build secret and opt-out; the audit hook on push.

## Impact

- New: `app/api/rate_limit.py`, `app/domain/guardrails.py` (support check, output filter, delimitation helpers), `app/adapters/injection/` (config, `PromptGuardDetector`, `NoDetector`), `tests/adversarial/`, `tests/fixtures/hostile_pdfs/` (generated by a script, not committed as binaries where avoidable), `benchmarks/adversarial/`, `scripts/resolve_model_hashes.py`.
- Changed: `app/main.py`, `app/api/middleware.py`, `app/api/errors.py` (429 code), `app/settings.py`, `app/bootstrap.py`, `app/application/ports.py`, `app/application/ingest_document.py`, `app/application/ask_question.py`, `app/domain/prompting.py`, `app/domain/types.py` (`Chunk.flagged`, `IngestResult.chunks_flagged`), `app/adapters/ocr/rendering.py` and `engine.py`, `app/adapters/vector_store/qdrant_store.py`, `tests/fakes/vector_store.py`, `Dockerfile`, `docker-compose.yml`, `.env.example`, `.pre-commit-config.yaml`, `README.md`, `docs/BENCHMARKS.md`, `docs/KNOWN_LIMITATIONS.md`.
- Unchanged: `app/models/schemas.py` and the HTTP contract (the flagged count is logged, not returned); the stored vectors and ids (the flag is one more payload field, so an existing collection keeps working: a point without the field counts as not flagged).
- Dependencies: no new Python package (Prompt Guard 2 is a DeBERTa sequence classifier served by the installed `transformers`); the model adds about 1.1 GB to the image when included. Downloads (on approval): the model, once, with the user's Hugging Face token.
- Licensing: Prompt Guard 2 is under the Llama 4 Community License (not OSI); the image's NOTICE and the README carry the required attribution ("Built with Llama"). The user chose it over the Apache-licensed, English-only alternative for its coverage of the five languages.
