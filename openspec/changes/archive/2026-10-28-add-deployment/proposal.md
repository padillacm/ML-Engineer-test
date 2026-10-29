# Proposal

## Why

The service runs only as a test image today: there is no image meant for production, no compose file that starts the app with a persistent Qdrant and a language model, and the language model was a scripted fake in every measurement. The take-home asks for Docker/compose (bonus), and the design document needs measured answer quality, answer latency and end-to-end throughput of the deployed stack, not of test harnesses.

## What Changes

- **Runtime image** (`runtime` stage on top of `base`): project dependencies only (no dev extras, no tests), non-root user `app` (uid 10001), `tini` as PID 1, read-only models at `/models`, Hugging Face offline, `MALLOC_ARENA_MAX=2`, Uvicorn with `--http h11` (a pipelined request cannot hide a client disconnect, block 5 P5), one worker process, a graceful-shutdown timeout, a healthcheck on `/health/ready` written in Python (the image has no curl).
- **`docker-compose.yml`** for the whole stack:
  - `app` (runtime image), published on `127.0.0.1:${APP_PORT:-8000}` only.
  - `qdrant` (same pinned v1.20.0 digest as the tests) with a named volume, not published.
  - Profile `gpu`: `ollama` (pinned version and digest, NVIDIA GPU reservation, named volume for models, server-side context length) and a one-shot `ollama-pull` that downloads the configured model and exits; the app waits for both.
  - Without the profile the app starts without a language model (questions that need one answer 503 `llm_unavailable`), or uses any OpenAI-compatible endpoint set in `.env`.
  - Two networks: an internal one for app, Qdrant and Ollama; the app (outbound to a hosted model, published port) and Ollama (model download) also on an external one. Start order gated by healthchecks.
- **`.env.example`** with every variable the stack reads, placeholders only (the user's own key goes into `.env`, which is git- and docker-ignored; `.gitignore` gets `!.env.example`).
- **Deployment smoke tests** (`tests/deploy/`, marker `deployed_service`, skipped unless `SERVICE_URL` is set) run from the test image against the running stack through `docker-compose.smoke.yml`: readiness, extraction on the sample PDFs, ingest twice, an answerable and an off-topic question, runtime image hygiene (no dev packages, non-root, offline).
- **Measurements** (container, real models, real GPU), recorded in `docs/BENCHMARKS.md`:
  - Language model choice on this GPU (RTX 4070 Ti, 12 GB): Gemma 4 candidates that fit with the needed context (`gemma4:e4b-it-qat`, `gemma4:12b-it-qat`, `gemma4:12b-it-q4_K_M`), and optionally a hosted OpenAI model if the user provides a key. Answer accuracy against the corpus gold answers, refusals on answerable and unanswerable questions, answer language, latency p50/p95, tokens per second, VRAM, concurrent questions (`OLLAMA_NUM_PARALLEL`).
  - Context length: largest prompt in tokens, so the server context is set with margin and nothing is truncated.
  - Deployed throughput: parallel ingestion and extraction on the runtime image under CPU limits, the defaults re-checked there, cold start (first start after build, model load), image sizes.
- Defaults that change from the measurements (for example `LLM_MODEL`, `LLM_READ_TIMEOUT_S`, `MAX_CONCURRENT_QUESTIONS`) are applied with their evidence.

## Capabilities

### New Capabilities

- `deployment`: the runtime image, the compose stack and its profiles, configuration through `.env`, network exposure, start order and health, and the smoke tests that check a running stack.

### Modified Capabilities

The measurements (design D9 and D10) led to these deltas:

- `llm-client`: the request adapts to the model's parameters (reasoning effort, token-limit field, temperature), and rejected requests and replies cut at the token limit are diagnosable.
- `vector-storage`: concurrent first use binds the collection once, tolerating Qdrant's transient error right after a collection is created.
- `document-ingestion`: embeddings are computed concurrently on one shared model, with a slot reserved for questions and slots handed out in arrival order.
- `name-extraction`: the recognition pool hands out its pipelines in arrival order.
- `service-configuration`: the concurrency defaults (and `TORCH_THREADS`) follow the CPUs available to the process, cgroup v2 or v1; the language model and Qdrant keys can be read from files (`LLM_API_KEY_FILE`, `QDRANT_API_KEY_FILE`, design D11).
- `vector-storage`: a full store disk is named in the error (review, task 8.1).
- `observability`: successful health probes are access-logged at DEBUG (design D12).

## Impact

- New: `docker-compose.yml`, `docker-compose.smoke.yml`, `.env.example`, `deploy/empty-secret` (the empty placeholder secret, D11), `app/llm_warmup.py` (the start-up check of the local model, D2 and D13), `tests/deploy/`, a `runtime` stage in `Dockerfile`.
- Changed: `Dockerfile` header, `.gitignore`, `pyproject.toml` (new pytest marker), README (how to run the stack), `docs/BENCHMARKS.md`, `docs/KNOWN_LIMITATIONS.md`, `app/settings.py` and `app/bootstrap.py` (new settings and CPU-derived defaults), the LLM, vector-store, embedder and NER adapters, the new `app/adapters/slots.py`, and the access-line level of health probes (`app/api/middleware.py`, `app/api/health.py`, `app/main.py`, `app/observability.py`).
- Downloads (on approval): the Ollama image (about 2 GB) and one model per candidate (6.1 GB, 7.2 GB and 8.0 GB); the user's OpenAI key is used only if the user adds it to `.env`, and only for the optional comparison.
- `app/models/schemas.py` and the HTTP contract are unchanged.
