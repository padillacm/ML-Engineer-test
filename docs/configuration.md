# Configuration

In the compose stack the settings go in `.env` (copy `.env.example`, which lists every one with its default); [Running the stack](operations.md) says which ones compose sets itself.

## Settings

All configuration comes from environment variables, read once at start-up by `app/settings.py`
(names are case-insensitive; an empty value counts as unset). A `.env` file in the working
directory is read too; it is git- and docker-ignored, and API keys belong there or in the
environment, never in a committed file. An invalid or missing value stops the service before
it accepts requests, with a message naming the variable and the rule (never its value).
Required variables are marked **R**; every default is the benchmarked value ([benchmark record](BENCHMARKS.md)).

`derived` means the value is computed at start-up from the CPUs when unset ([Derived concurrency](#derived-concurrency)).

### Models and device

| Variable | Default | Meaning |
|---|---|---|
| `NER_MODEL_PATH` | **R** | Directory of the NER model (the images set it). |
| `EMBEDDING_MODEL_PATH` | **R** | Directory of the embedding model (the images set it). |
| `EMBEDDING_MODEL_REVISION` | **R** | Commit of the embedding model's files; part of the embedding identity, so a new revision opens a new vector collection. |
| `TORCH_THREADS` | 2, at most the CPUs | Torch intra-op threads, process-wide, shared by the recogniser, the embedder and the detector. The former `NER_THREADS` and `EMBEDDING_THREADS` stop start-up. |
| `NER_POOL_SIZE` | derived | Texts recognised at once, sharing one model. |
| `EMBEDDING_POOL_SIZE` | derived | Passage batches embedded at once, sharing one model; one more slot is reserved for questions. |
| `EMBEDDING_BATCH_SIZE` | 16 | Texts per batch of the embedder and of the detector. |
| `DEVICE` | `auto` | Where the three models run: `auto` (the GPU when torch sees one, else the CPU), `cpu`, or `cuda` (start-up fails without a GPU). On the GPU the embedder and the detector run in fp16, the recogniser in fp32; the image must be the CUDA build ([Running the stack](operations.md#torch-builds-and-gpu-tests)); when it pays off: [CPU or GPU](operations.md#cpu-or-gpu). |

On the CPU, start-up warns when a pool times `TORCH_THREADS` exceeds the CPUs available.

### OCR

| Variable | Default | Meaning |
|---|---|---|
| `OCR_DPI` | 200 | Render resolution. |
| `OCR_WORKERS` | derived | Tesseract processes per document. |
| `OCR_TIMEOUT_S` | 60 | Longest Tesseract run for one page. |
| `OCR_FLATTEN` | true | Illumination flattening before OCR. |
| `MAX_PAGES` | 50 | Longest document accepted (`too_many_pages`). |
| `OCR_RENDER_TIMEOUT_S` | 10 | Time budget of one page render, then `pdf_unreadable`. |
| `OCR_MAX_IMAGE_PIXELS` | 80,000,000 | Decoded image pixels a page may declare, checked before rendering, then `too_large`. |
| `MAX_PAGE_CHARS` | 30,000 | Longest recognised text of one page, then `too_large`; the densest page of the benchmark corpus has about 6,000 characters. |

### Matching, chunking and retrieval

| Variable | Default | Meaning |
|---|---|---|
| `MATCH_THRESHOLD` | 0.90 | Lowest similarity of a fuzzy match. |
| `CHUNK_SIZE` | 800 | Longest chunk, in characters. |
| `CHUNK_OVERLAP` | 0.12 | Text a chunk repeats from the previous one, as a share of `CHUNK_SIZE`. |
| `RAG_THRESHOLD` | 0.808 | Lowest cosine of the best hit; below it the question is refused without a model call. |
| `RAG_TOP_K` | 5 | Chunks retrieved per question. |
| `RAG_MAX_CHUNKS` | 3 | Chunks whose parent pages go into the context. |
| `RAG_MAX_CONTEXT_CHARS` | 6000 | Budget of the whole rendered context. |
| `RAG_MAX_CHUNKS_PER_DOCUMENT` | 2 | Context chunks one document may give when chunks of other documents also matched. |

### Guardrails

| Variable | Default | Meaning |
|---|---|---|
| `INJECTION_MODEL_PATH` | unset | Directory of Prompt Guard 2; unset means no model (every model score 0). The images set `/models/prompt-guard` when built with it. |
| `INJECTION_THRESHOLD` | 0.5 | Score at or above which a page's chunks are stored flagged and never retrieved, or a question is refused. |
| `INJECTION_POOL_SIZE` | 2 on the CPU, 1 on the GPU | Page screenings at once on the one loaded model, plus one slot reserved for questions. |
| `INJECTION_PATTERNS` | true | The instruction patterns in five languages, combined with the model's score by the higher one. |
| `RAG_SUPPORT_MIN_SHARE` | 0.6 | Share of an answer's content words that must appear in the context, else the fixed refusal. |

### Vector store

| Variable | Default | Meaning |
|---|---|---|
| `QDRANT_URL` or `QDRANT_LOCATION` | **R**, exactly one | A Qdrant server, or `:memory:` for the embedded, non-persistent mode. |
| `QDRANT_API_KEY` or `QDRANT_API_KEY_FILE` | unset | The key, or a file holding it (as for the language model). |
| `QDRANT_COLLECTION_PREFIX` | `chunks` | Prefix of the collection names. |
| `QDRANT_TIMEOUT_S` | 10 | Timeout of one request. |
| `QDRANT_ATTEMPTS` | 3 | Tries of one call on connection errors, timeouts and 502/503/504. |
| `QDRANT_DEADLINE_S` | 20 | Overall time of one call with its retries. |

### Language model

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | unset | OpenAI-compatible endpoint; unset means no model (questions that need one answer 503). |
| `LLM_MODEL` | required with a base URL | Model name. |
| `LLM_API_KEY` or `LLM_API_KEY_FILE` | unset | The key, or a file holding it, read once at start-up (an empty file means no key); not both. Compose sets the file to its secret. |
| `LLM_CONNECT_TIMEOUT_S`, `LLM_READ_TIMEOUT_S` | 5, 60 | Per-operation timeouts. |
| `LLM_TOTAL_TIMEOUT_S` | 90 | Whole call, retries included. |
| `LLM_ATTEMPTS` | 3 | Tries on connection errors, timeouts, 429 and 5xx. |
| `LLM_MAX_TOKENS` | 512 | Longest reply; a thinking model's reasoning counts against it. |
| `LLM_REASONING_EFFORT` | unset | Sent as `reasoning_effort` only when set (`none` turns thinking off on Ollama). |
| `LLM_MAX_TOKENS_FIELD` | `max_tokens` | Name of the reply limit; reasoning models need `max_completion_tokens`. |
| `LLM_SEND_TEMPERATURE` | true | Send temperature 0; false for models that accept only their default. |

### Limits

| Variable | Default | Meaning |
|---|---|---|
| `MAX_UPLOAD_MB` | 20 | Largest PDF. |
| `MAX_NAMES` | 1000 | Most entries in `names`. |
| `MAX_CONCURRENT_JOBS` | derived | Extractions and ingestions at once. |
| `MAX_CONCURRENT_QUESTIONS` | 16 | Questions at once. |
| `BUSY_WAIT_S` | 10, at most 300 | Longest wait for a slot, then 503 `service_busy`. |
| `READINESS_TIMEOUT_S` | 2, at most 30 | Longest readiness check. |
| `RATE_LIMIT_UPLOADS_PER_MINUTE`, `RATE_LIMIT_UPLOADS_BURST` | 60, 20 | Per client address, for `/api/extract` and `/api/ingest` together. |
| `RATE_LIMIT_QUESTIONS_PER_MINUTE`, `RATE_LIMIT_QUESTIONS_BURST` | 240, 60 | Per client address, for `/api/ask`. |
| `MAX_JOBS_PER_CLIENT` | 4 | Extraction and ingestion jobs one address may hold at once. |

A rate or cap of 0 turns that limit off; over a limit the client gets 429 `rate_limited` with `Retry-After`.

### API docs, audit viewer and logging

| Variable | Default | Meaning |
|---|---|---|
| `EXPOSE_DOCS` | false | Serve `/docs`, `/redoc` and `/openapi.json` (otherwise 404). |
| `AUDIT_VIEWER_ENABLED` | true | Serve the page at `/audit` and its files; false makes them 404, the API is unaffected. |
| `PDFJS_DIR` | unset | Directory of the pinned pdf.js files; unset means `vendor/pdfjs`, where `uv run python scripts/fetch_pdfjs.py` puts them (the images set `/app/vendor/pdfjs`). |
| `LOG_LEVEL` | INFO | Lowest level logged. |
| `LOG_FORMAT` | json | `json` or `text`. |
| `LOG_EXCEPTION_MESSAGES` | false | Local debugging only: logs exception messages, which may contain document text. |

## Derived concurrency

Values marked `derived` are computed at start-up when unset, from the CPUs available to the process (affinity and a
container's CPU quota, cgroup v2 or v1), and logged in one `concurrency in use` line; a set value
always wins.
The rule (`app/bootstrap.py` `concurrency_for`, with `TORCH_THREADS` T): jobs = 3/4 of the CPUs
rounded up, at most 12; NER pool = CPUs / T, at most 8; embedder pool = CPUs / (2 T), at most 4;
OCR workers = CPUs, at most 4; each at least 1. On the GPU the NER, embedder and detector pools
are 1 each instead (every size tried came within 3 %, [benchmark record](BENCHMARKS.md) 23.5); jobs and OCR
workers keep the CPU rule, because OCR stays on the CPUs. With T = 2:

| CPUs | 1 | 2 | 4 | 6 | 8 | 12 | 16 or more |
|---|---|---|---|---|---|---|---|
| `MAX_CONCURRENT_JOBS` | 1 | 2 | 3 | 5 | 6 | 9 | 12 |
| `NER_POOL_SIZE` | 1 | 1 | 2 | 3 | 4 | 6 | 8 |
| `EMBEDDING_POOL_SIZE` | 1 | 1 | 1 | 1 | 2 | 3 | 4 |
| `OCR_WORKERS` | 1 | 2 | 4 | 4 | 4 | 4 | 4 |

The 4-, 8- and 16-CPU columns are measured ([benchmark record](BENCHMARKS.md) 17.4 and 18); the others follow
the rule.

The image also sets `MALLOC_ARENA_MAX=2` (glibc), which keeps the service's memory flat under
load ([benchmark record](BENCHMARKS.md) section 15); set it too when running the service outside Docker on Linux.

## Running the service outside Docker

The service needs the model directories, Tesseract and a vector store. For example:

```bash
# .env (example values; never commit this file)
NER_MODEL_PATH=/models/xlm-roberta-base-ner-hrl
EMBEDDING_MODEL_PATH=/models/multilingual-e5-base
EMBEDDING_MODEL_REVISION=<commit of the e5 files>
QDRANT_URL=http://localhost:6333
LLM_BASE_URL=http://localhost:11434/v1     # any OpenAI-compatible server, e.g. Ollama
LLM_MODEL=gemma4:e4b-it-qat
LLM_REASONING_EFFORT=none                  # thinking off, as benchmarked
# LLM_API_KEY=...                          # only if the server needs one

uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Models and clients are built once, when the application starts (not on the first request),
and closed when it stops; one process holds one copy of the models, so scale with processes or
replicas, not threads. Without `LLM_BASE_URL` the service still starts: `/api/ask` refuses
questions that match nothing and answers 503 `llm_unavailable` when it would need the model.
If Qdrant is down the service starts too (readiness reports it) and recovers when Qdrant is
back. The test image already contains Tesseract and both models; to run the service from the
runtime image with Qdrant and, optionally, a local model, see [Running the stack](operations.md).
