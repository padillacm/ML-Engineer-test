# Proposal

## Why

`POST /api/ingest` and `POST /api/ask` must index scanned PDFs and answer questions grounded in them, with `sources` that are literally what was retrieved. The core already has the ports (`Embedder`, `VectorStore`, `LLMClient`), the chunker (800 characters, 12 % overlap, word-safe) and fakes; no real adapter or use case exists. The benchmark on 600 OCR'd documents (`docs/BENCHMARKS.md` sections 6-8) fixed the retrieval defaults: `intfloat/multilingual-e5-base` (+5.1 answer recall@5 over e5-small), retrieve 800-character chunks and pass the parent page of the top 3 distinct chunks to the LLM (answer in context 81.4 % at ~4.8k characters), and refuse without calling the LLM when the best cosine is below 0.808 (keeps 95 % of answerable questions). The review requirements add: retries and timeouts on the LLM and the vector store, vector ids unique across documents, no hallucinated sources.

## What Changes

- Archive `add-name-extraction` so its specs join the baseline.
- Add an e5 embedder adapter (`transformers` + the CPU `torch` already in the image, mean pooling, L2 normalisation, `query: `/`passage: ` prefixes inside the adapter, pinned model baked into the image, offline) with its `EmbeddingIdentity`.
- Add a Qdrant vector-store adapter (`qdrant-client`): collection bound to the embedding identity (stored as collection metadata), collection name derived from the identity so a model change never mixes vectors, point ids as UUIDv5 of `(document_id, sequence)`, payload with the chunk, its page text and the document's content hash, cosine scores clamped to 0..1, timeouts and bounded retries, errors mapped to the core categories. The same adapter runs against an in-memory Qdrant in unit tests and a Qdrant server in the container.
- Add an OpenAI-compatible LLM client (`httpx`, `POST {base_url}/chat/completions`) usable with Ollama and hosted providers: configurable base URL, model, API key (never logged or rendered), timeouts, retries with exponential backoff and jitter on connection errors, timeouts, 429 and 5xx; everything else fails fast.
- Add the use cases `IngestDocument` (document id derived from the SHA-256 of the file, so an identical file is detected before any OCR and never stored twice; otherwise OCR -> chunk -> embed -> upsert; returns chunks stored and whether the document was already indexed) and `AskQuestion` (embed query -> search -> refuse below threshold without calling the LLM -> build a grounded prompt from the parent pages of the top distinct chunks -> answer; `sources` are the retrieved chunk texts).
- Add a test environment with Docker Compose (`docker-compose.test.yml`: Qdrant server + the test image) so the vector-store contract and an end-to-end ingest -> ask test run against a real Qdrant; block 6 extends it into the runtime compose.
- Dependencies: `qdrant-client`, `httpx` (runtime); the embedding model in the image's model stage.

## Capabilities

### New Capabilities
- `document-ingestion`: turning an uploaded PDF into stored, retrievable chunks with unique ids and provenance.
- `grounded-answering`: answering a question only from retrieved document content, refusing when nothing relevant is found, and returning the retrieved text as sources.
- `vector-storage`: how chunk vectors are stored, isolated by embedding identity, searched, and protected against transient failures.
- `llm-client`: calling an OpenAI-compatible chat model with bounded latency, retries and secret-safe errors.

### Modified Capabilities
<!-- None: the port signatures from add-domain-foundation are kept. -->

## Impact

- New: `app/adapters/embeddings/`, `app/adapters/vector_store/`, `app/adapters/llm/`, `app/application/ingest_document.py`, `app/application/ask_question.py`, `app/domain/prompting.py` (pure prompt and context building), `docker-compose.test.yml`; tests under `tests/adapters/`, `tests/application/`, `tests/integration/`.
- `pyproject.toml`/`uv.lock`: `qdrant-client`, `httpx`. `Dockerfile`: the e5-base model in the model stage (pinned revision, hashes, NOTICE), offline env.
- Image grows by ~1.1 GB (e5-base weights); the ONNX/size follow-up stays in `docs/KNOWN_LIMITATIONS.md`.
- Out of scope: HTTP endpoints, upload validation and environment settings (block 5); the runtime compose with Ollama and GPU (block 6); measuring LLM answer quality (needs the LLM; done in block 6 with Ollama).
