# Tasks

## 1. Baseline

- [x] 1.1 Archive `add-name-extraction`; verify `openspec list --specs` includes `name-extraction` and `openspec validate --all --strict` passes

## 2. Dependencies and image

- [x] 2.1 Add `qdrant-client` and `httpx` to `pyproject.toml`, `uv lock`, `uv sync --extra dev`; verify imports and `uv-audit`
- [x] 2.2 Add `intfloat/multilingual-e5-base` to the Dockerfile `model` stage (pinned revision, sha256 of every file, model card, NOTICE, read-only, `EMBEDDING_MODEL_PATH`); verify the image builds, loads it offline and report the size

## 3. Port change

- [x] 3.1 Add `page_texts` and `content_hash` to `VectorStore.upsert`, `page_text` to `SearchHit` and `document_chunk_count(document_id)` to `VectorStore`; update `FakeVectorStore`, the `VectorStoreContract` and existing tests; verify host suite and architecture guard

## 4. Domain

- [x] 4.1 Implement `app/domain/prompting.py` (`select_context`, `build_messages`, `REFUSAL`); verify page deduplication, rank order, budget never exceeded, cut at word boundary, labels, and the grounding instructions

## 5. Adapters

- [x] 5.1 Implement the e5 embedder (`app/adapters/embeddings/`): config, local offline load with weight check, prefixes, mean pooling, normalisation, batching, threads, lock, identity; verify unit tests with a tiny model, the `Embedder` contract, and in the container that the real model gives unit vectors and ranks a paraphrase above an unrelated sentence
- [x] 5.2 Implement the Qdrant store (`app/adapters/vector_store/`): collection naming and identity metadata, uuid5 ids, payload, delete-then-upsert with cleanup on failure, search with clamped scores, timeout and retries with error mapping; verify the `VectorStore` contract on `:memory:` on the host and on the server in compose, plus retry tests with an injected failing client
- [x] 5.3 Implement the OpenAI-compatible client (`app/adapters/llm/`): config, masked key, timeouts, retries with backoff, jitter and Retry-After, error mapping, response parsing; verify with `httpx.MockTransport` every scenario of the `llm-client` spec and an `LLMClient` contract

## 6. Use cases

- [x] 6.1 Implement `IngestDocument` and `AskQuestion`; verify with fakes every scenario of `document-ingestion` and `grounded-answering` (two documents kept, same file twice detected by hash before any OCR with the stored count reported, identical bytes under another name, one-byte difference stored separately, nothing stored on embedding failure, refusal without calling the LLM, empty store, parent page in context, sources equal retrieved texts, LLM errors propagate)

## 7. Test environment

- [x] 7.1 Add `docker-compose.test.yml` (pinned Qdrant with healthcheck, test image) and the `qdrant_server` marker with auto-skip; verify `docker compose -f docker-compose.test.yml run --rm tests` runs the full suite including the server contract, and document it in the README
- [x] 7.2 Add an end-to-end test in compose: ingest a sample PDF and a second PDF with real OCR and embeddings into the Qdrant server, ask an answerable and an off-topic question with a scripted LLM; verify sources, refusal, that both documents remain, and that re-ingesting the first PDF reports already indexed without adding points

## 8. Calibration

- [x] 8.1 Retrieval calibration on the 60-document sample in the container (real OCR, e5-base, Qdrant): answer and page recall@k, refusal rates at 0.808, paired with BENCHMARKS 7-8 where possible; record in `docs/BENCHMARKS.md` and add any unfixable finding to `docs/KNOWN_LIMITATIONS.md`

## 9. Wrap-up

- [x] 9.1 Host checks (pytest with coverage, ruff, mypy, pre-commit, uv-audit) and the compose suite with `--network`-isolated test image apart from Qdrant; verify all pass
- [x] 9.2 `openspec validate add-rag-pipeline --strict`
