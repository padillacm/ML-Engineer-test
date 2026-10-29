# Tasks

## 1. Baseline

- [x] 1.1 Archive `add-http-api`; verify `openspec validate --all --strict` passes with 13 specs

## 2. Runtime image

- [x] 2.1 Add the `runtime` stage (D1): runtime dependencies only, `app/` only, non-root, tini, Uvicorn with `--http h11` and a graceful-shutdown timeout, Python healthcheck; verify it builds, starts, answers `/health/ready`, and the hygiene checks (no pytest/pymupdf, no tests, uid 10001, offline); record its size against the test image

## 3. Compose stack

- [x] 3.1 `docker-compose.yml` (D2, D4): app, Qdrant with a named volume, profile `gpu` with pinned Ollama, GPU reservation and one-shot pull; internal and egress networks; app published on `127.0.0.1` only; verify start with and without the profile, `--wait`, no published Qdrant/Ollama port, and data surviving `down` + `up`
- [x] 3.2 `.env.example` and `.gitignore` (D3); verify every setting is listed, no secret has a value, the stack starts without `.env`, and a key placed in `.env` never shows in logs or `docker compose config` of the image

## 4. Smoke tests

- [x] 4.1 `tests/deploy/` with marker `deployed_service` and `docker-compose.smoke.yml` (D8); verify the suite is skipped on the host without `SERVICE_URL` and passes against the stack with and without the `gpu` profile (the answerable question only with a model)

## 5. Language model choice

- [x] 5.1 Pull the candidates (D5), record tags and digests; measure the prompt's largest token count and set the server context with margin (verify no truncation)
- [x] 5.2 Answer quality and latency per candidate (D6) on the stratified sample of the 600-document index, with the manual audit of 50 disagreements; optional hosted reference if the user provides a key; choose the model by the D5 rule and set `LLM_MODEL` and the Ollama options
- [x] 5.3 `llm-client` delta (D9): `LLM_REASONING_EFFORT`, `LLM_MAX_TOKENS_FIELD`, `LLM_SEND_TEMPERATURE`, provider error code in the error message, warning for an empty reply cut at the token limit; compose turns thinking off for Ollama; verify against Ollama (thinking off) and the hosted model without the harness rewrites
- [x] 5.4 `vector-storage` fix (D9): concurrent first `ensure_collection` calls serialised in-process and tolerant of a collection created by another process; test with concurrent first ingestions against a real Qdrant

## 6. Throughput and sizing

- [x] 6.1 Deployed throughput (D7): ingest and extract docs/min unlimited and with `cpus: 8`, questions per minute at `OLLAMA_NUM_PARALLEL` 1/2/4, cold start, parity of extraction/ingestion with the test image on the 60-document sample; set `MAX_CONCURRENT_QUESTIONS` and `LLM_READ_TIMEOUT_S` from the measurements; record everything in `docs/BENCHMARKS.md`
- [x] 6.2 Fixes from 6.1 (D10): embedder pool (`EMBEDDING_POOL_SIZE`, shared weights, calibrated), concurrency defaults derived from the available CPUs (`MAX_CONCURRENT_JOBS`, `NER_POOL_SIZE`, `EMBEDDING_POOL_SIZE`, `OCR_WORKERS`), healthcheck `start_interval` for Qdrant and Ollama; tests
- [x] 6.3 Re-measure the deployed stack after 6.2: ingest and extract at 16, 8 and 4 CPUs with the derived defaults, questions during ingestion, cold start; update `docs/BENCHMARKS.md` (section 19; with the FIFO hand-off of the NER and embedder slots measured in section 18.6, which removed `KNOWN_LIMITATIONS.md` P11)

## 7. Documentation

- [x] 7.1 README: run the stack (with and without GPU, hosted model through `.env`, Linux NVIDIA toolkit note), smoke tests, sizing table; record unfixable findings in `docs/KNOWN_LIMITATIONS.md`

## 8. Review and wrap-up

- [x] 8.1 Independent review (image contents, secrets in layers and logs, network exposure, start order and restarts, GPU absent, model download failure, disk full, answer quality claims); fix every fixable finding with tests
- [x] 8.2 Final verification: host `pytest --cov`, ruff, mypy, all pre-commit hooks, `uv-audit`, `openspec validate --all --strict`, full compose test suite, smoke suite against the stack with the `gpu` profile
