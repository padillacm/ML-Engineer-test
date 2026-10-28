# Design

## Context

From earlier blocks: the `Embedder`, `VectorStore` and `LLMClient` ports with fakes and contract tests (`tests/application/contracts.py`); `chunk_document` (800 characters, 12 % overlap, offsets into the cleaned page text); `TesseractOcrEngine` (pages with words and the language used); the `docintel-test` image (Python 3.12, Tesseract 5.5, pinned NER model in a `model` stage, offline, CPU `torch`, `tini`). The defaults below come from `docs/BENCHMARKS.md` sections 6-8 (embedders, chunking, refusal threshold) measured on OCR text of the synthetic corpus. Decisions taken earlier with the user: Qdrant server as the vector database (Docker), Ollama or any OpenAI-compatible provider through one client, Docker as the main way to run the service.

## Goals / Non-Goals

**Goals:** real adapters behind the existing ports; the two use cases; retries and timeouts on every external call; ids that cannot collide; sources that are exactly what was retrieved; tests against a real Qdrant server in Docker.

**Non-Goals:** HTTP, validation and environment settings (block 5); the runtime compose with Ollama on GPU (block 6); hybrid BM25 retrieval and the complexity router (future work, `DESIGN.md`); measuring answer quality of a real LLM (block 6).

## Decisions

### D1. Embedder: `intfloat/multilingual-e5-base` with our own pooling

MIT licence, 768 dimensions, +5.1 answer recall@5 over e5-small on OCR text (BENCHMARKS 6). Implemented with `transformers` `AutoModel` + mean pooling over the attention mask + L2 normalisation, the recipe from the model card, instead of adding `sentence-transformers` (a second wrapper over the same `torch` and `transformers`). Prefixes `query: ` and `passage: ` live in the adapter. Inputs are truncated at 512 tokens (an 800-character chunk is ~200 tokens). Batched (default 16), `torch.inference_mode()`, threads configurable (default 2), one lock per instance, loaded once from a local path, offline, with the same weight-integrity check as the NER adapter (`output_loading_info`). Identity: `EmbeddingIdentity("intfloat/multilingual-e5-base@<revision>", 768, "e5-query-passage", normalized=True, metric="cosine")`. The model is baked into the image in the existing `model` stage with a pinned revision, sha256-verified files and a NOTICE.

### D2. Ports change: page text travels with the chunks

Passing the parent page to the LLM needs the page text at question time. `VectorStore.upsert` gains keyword arguments `page_texts: Mapping[int, str]` and `content_hash: str`, and `SearchHit` gains `page_text: str`. A new method `document_chunk_count(document_id) -> int` lets the ingest use case detect an already indexed file before doing any work. The page text is stored in each chunk's payload (about 3 chunks per page of ~1.75k characters, so ~3x duplication of a few KB per page), which keeps search a single round trip; a separate page collection is the alternative if documents grow large. The fake, the contract and existing tests follow.

### D3. Vector store: Qdrant through `qdrant-client`

- **Collection per identity**: name `{prefix}_{sha256(identity)[:12]}` (prefix configurable, default `chunks`), cosine distance, size from the identity; the identity is written in the collection metadata on creation and checked on `ensure_collection` (mismatch -> `IncompatibleEmbedding`, nothing modified). A model change therefore opens a new collection; re-indexing from stored page text is a documented follow-up.
- **Ids**: `uuid5(NAMESPACE, f"{document_id}:{sequence}")`, deterministic and unique across documents.
- **Payload**: `document_id`, `page`, `sequence`, `text`, `page_text`, `start`, `end`, `content_hash`; payload index on `document_id` for deletion.
- **Upsert (idempotent)**: every point carries `chunk_total`; batches are written with `wait=True` without deleting first (the ids are deterministic, so concurrent writers of the same document write the same points and readers never see it disappear), then only this document's points not written in this call are deleted. A failed batch deletes nothing: another writer may have completed the document, and a partial state is detectable.
- **Completeness**: `document_chunk_count` returns the count only when every point of the document agrees on `chunk_total` and the count equals it, otherwise 0, so an interrupted ingest is not reported as indexed and the next ingest of the same file completes it.
- **Search**: `query_points` with `limit=top_k`, `with_payload=True`; scores clamped to 0..1.
- **Lost collection**: a 404 "collection not found" (or the local mode's equivalent) recreates the collection with the bound identity once and retries the operation.
- **Resilience**: client `timeout` configurable (default 10 s); retries (attempts default 3, exponential backoff 0.2 s x 2^n with full jitter, cap 2 s) on connection errors, timeouts and HTTP 502/503/504; an overall `deadline_s` (default 20 s) bounds each database request with its retries; exhausted -> `DependencyUnavailable("vector_store_unavailable")` / `DependencyTimeout("vector_store_timeout")`, raised from the original. Other HTTP errors fail at once.
- **Modes**: `url` for a server (compose), `location=":memory:"` for unit and contract tests on the host. The server path is the one used in production and in the container tests.

### D4. LLM client: one `httpx` client for every OpenAI-compatible provider

`POST {base_url}/chat/completions` with `{"model", "messages", "temperature": 0, "max_tokens"}`; `Authorization: Bearer` only when a key is configured. `httpx.Client` with `Timeout(connect=5, read=60)` configurable. Retries (default 3 attempts) with exponential backoff and full jitter on `ConnectError`, `TimeoutException`, 429 and 5xx, honouring `Retry-After` (capped); 4xx other than 429 fail at once. Errors: `DependencyTimeout("llm_timeout")` or `DependencyUnavailable("llm_unavailable")`, with status code in the message, never the key, the prompt or the response body. The key is held in a small wrapper whose `repr` is masked. One client works with Ollama (`http://ollama:11434/v1`) and hosted providers; non-OpenAI-compatible APIs (e.g. native Anthropic) would be another adapter behind the same port (future work).

### D5. Prompt and context building in the domain (`app/domain/prompting.py`)

Pure functions: `select_context(hits, max_chunks=3, max_chars=6000)` takes the first `max_chunks` distinct chunks in rank order, deduplicates their pages by `(document_id, page)`, and adds whole pages while the **rendered** context (labels and separators included, `render_context(pages)`) fits the budget; the first page may be cut at a word boundary to the room left after its label, later pages only if at least 200 characters fit, and selection stops at the first page that does not fit (rank order is kept and the budget is never exceeded; if not even the first label fits, no context is built and the use case refuses). `shown_sources(hits, pages, max_chunks)` returns, in rank order, the selected chunks whose text lies inside the included (possibly cut) page text, so a source is always something the model saw. `build_messages(question, pages)` returns a system message (answer only from the numbered context, no outside knowledge, the question's language, do not invent documents, pages or quotes, be concise, and reply with exactly `NOT_IN_DOCUMENTS` when the answer is not in the context) and a user message `Context: … Question: …` with pages labelled `[n] document <id>, page <p>`. `is_refusal(reply)` recognises the sentinel after normalising case, whitespace, quotes and punctuation (and a blank reply or the English refusal); the use case maps it to the fixed English `REFUSAL` with no sources. Injection-resistant framing of the context is deferred to the security pass (below).

### D6. Use cases

- `IngestDocument(ocr, chunk_config, embedder, store)`: `content_hash = sha256(pdf)`, `document_id = uuid5(NAMESPACE, content_hash)`; `ensure_collection`; if `document_chunk_count(document_id) > 0` return `IngestResult(document_id, chunks_stored=<that count>, already_indexed=True)` without OCR or embedding. Otherwise OCR (no hint, detection on) -> `clean_page_text` per page -> `chunk_document` -> `embed_passages` in batches -> `upsert(document_id, ..., page_texts, content_hash)` -> `IngestResult(document_id, chunks_stored, already_indexed=False)`. Embedding happens before any write, so an embedding failure stores nothing. Identity by content (not one UUID per upload) was chosen with the user so that re-uploads never duplicate chunks; different files still cannot collide because their hashes differ, and two concurrent uploads of the same file write the same ids, so the result is still a single copy. A document with no text stores zero chunks and is therefore not remembered as indexed (re-ingesting it repeats the OCR, which is harmless).
- `AskQuestion(embedder, store, llm, threshold=0.808, top_k=5, max_chunks=3, max_context_chars=6000)`: `embed_query` -> `search(top_k)` -> if empty or best score < threshold: `Answer(REFUSAL, sources=())` without calling the LLM -> `select_context` (no page fits -> refusal without calling the LLM) -> `build_messages` -> `llm.complete` -> refusal sentinel -> `Answer(REFUSAL, ())`, otherwise `Answer(reply, sources=shown_sources(kept hits, pages))`.

### D7. Test environment with Compose

`docker-compose.test.yml`: `qdrant` (official image pinned by digest, healthcheck on `/readyz`, tmpfs storage) and `tests` (the `docintel-test` image, `QDRANT_URL=http://qdrant:6333`, depends on qdrant healthy). Tests marked `qdrant_server` auto-skip when `QDRANT_URL` is unset; they run the `VectorStore` contract and an end-to-end ingest -> ask on a sample PDF with a scripted fake LLM. `docker compose -f docker-compose.test.yml run --rm tests` runs the whole suite. Ollama is not part of the test compose (block 6); an optional `llm_server` marker runs a real completion when `LLM_BASE_URL` is set.

### D8. Calibration

Retrieval only (no LLM): ingest the 60-document sample of the synthetic corpus through the real chain into Qdrant in the container, ask its answerable and unanswerable questions, and compare answer recall@k, page recall@k and the refusal rates at 0.808 with BENCHMARKS 7-8 for e5-base at `s800_o12`, paired where the benchmark has per-question results. Acceptance: within the benchmark's intervals.

## Deferred to the security pass

By decision of the user, security hardening of ingestion, the user's prompt and retrieval is handled in a dedicated pass after all blocks are stable, not in this change: prompt-injection resistance of the context (unforgeable passage delimiters, "document text is data" instructions, output verification against sources), and any further input-side controls. The finding and its repro are recorded in `docs/KNOWN_LIMITATIONS.md` (R3) so the pass starts from them. Secret handling in the LLM client (masked key, key validation, no pickling) stays in this change because it is correctness of configuration, not hardening.

## Risks / Trade-offs

- [Image grows by ~1.1 GB] -> listed in `docs/KNOWN_LIMITATIONS.md` with ONNX/int8 as the follow-up to measure.
- [Page text duplicated in payloads] -> small at this scale; separate page store if documents get long.
- [0.808 is corpus-specific; absent-entity questions are refused only ~38 %] -> configurable; the prompt-level refusal is the second line of defence; recalibration procedure in `DESIGN.md`.
- [Qdrant in-memory mode differs from the server in edge cases] -> the contract also runs against the server in compose.
- [`qdrant-client` and Qdrant server versions drift] -> both pinned; the server image by digest.

## Migration Plan

Additive except the `VectorStore.upsert`/`SearchHit` change, which updates the fake, the contract and their tests in the same change. Collections are created on first use; nothing to migrate.

## Open Questions

- None that change the specs or tasks.
