# Running the stack

How to run the service with Docker Compose in each of its configurations, size it, and what happens when something fails. For a first run, [Getting started](getting-started.md) is shorter.

The service runs from the `runtime` image (no tests or development tools, non-root, models baked in)
next to a Qdrant server whose data lives in a named volume. Only the service publishes a port, on
`127.0.0.1:8000` by default: it has no authentication, so put a reverse proxy with TLS and
authentication in front of it before exposing it beyond the host.

Configuration is optional: `cp .env.example .env` and uncomment what you change. Every setting is
listed there with its default; `.env` is git- and docker-ignored and never enters an image. Compose
reads `.env` and passes the service only the settings it reads (and its keys as secret files), so
other variables in `.env` never reach the container. Put a value that contains `$` in single
quotes (`LLM_API_KEY='abc$def'`): compose reads `$def` in an unquoted or double-quoted value as a
variable and silently replaces it.

## Without a GPU

```bash
docker compose up -d --wait
```

No Hugging Face token is needed: without `HF_TOKEN` the image is built without the injection
detector (see [Images and models](#images-and-models)). The first run builds the image; the command returns when the service is healthy
(`curl http://127.0.0.1:8000/health/ready`). Extraction and ingestion work. With no language model
configured, readiness reports `"llm": "not_configured"` and questions that need a model answer 503
`llm_unavailable` (questions the documents cannot answer are still refused with 200).

## With a hosted OpenAI-compatible model

Uncomment the hosted block in `.env` and fill it in, and start the stack without the `gpu` profile
(`LLM_MODEL` names the model of both, so with the profile `ollama-pull` stops with an error
instead of trying to pull the hosted model's name):

```dotenv
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=<model name>
LLM_API_KEY='<your key>'
```

then `docker compose up -d --wait`. The key stays in `.env` only: compose hands it to the service as
a secret file (`/run/secrets/llm_api_key`, read through `LLM_API_KEY_FILE`), so it is not in the
container's environment (`docker inspect` shows `LLM_API_KEY=` empty), not in `docker compose config`
and never in the logs. Without a key the service gets an empty file and runs without one. Anyone who
can use the Docker socket can still read the file inside the container. Outside compose,
`LLM_API_KEY` works as before. `QDRANT_API_KEY` in `.env` takes the same path
(`/run/secrets/qdrant_api_key`), and the stack's Qdrant then requires it.

Reasoning models (for example `gpt-6-luna`) reject the request's `max_tokens` and its temperature 0
with HTTP 400; for them also set

```dotenv
LLM_MAX_TOKENS_FIELD=max_completion_tokens
LLM_SEND_TEMPERATURE=false
```

(the model then runs at its default temperature, so answers are not deterministic).
`LLM_REASONING_EFFORT` (`none` ... `xhigh`) is sent only when set. A rejected request answers 503
`llm_unavailable`, and the service logs `LLM request failed (status 400, unsupported_parameter)`:
the status and the provider's error code, never its message. A reply cut at the token limit with
no content (a thinking model that spent `LLM_MAX_TOKENS` reasoning) is logged as a warning with the
`model` and `max_tokens` fields.

## With a local model on an NVIDIA GPU (Ollama)

```bash
COMPOSE_PROFILES=gpu docker compose up -d --wait
```

(PowerShell: `$env:COMPOSE_PROFILES = "gpu"; docker compose up -d --wait`; or put
`COMPOSE_PROFILES=gpu` in `.env`, after which `docker compose up -d --wait` is enough.) This also
starts Ollama with one NVIDIA GPU, downloads `LLM_MODEL` into the `ollama_models` volume unless it
is already there (a warm volume starts offline), runs one short prompt to load it and compile its
GPU kernels (cached in the same volume), and only then starts the service, pointed at Ollama. The
first start therefore takes a while (download plus about a minute of warm-up); later starts reuse
the volume. If the download fails (wrong model name, full volume, no network) or the model is
missing, `up` fails and the service is not started; `docker compose logs ollama-pull llm-warmup`
says why. `--profile gpu` on its own is refused with a message: compose does not show the command
line's profiles to the file, so it could not point the service at Ollama. Context length, parallel
requests and keep-alive are `OLLAMA_CONTEXT_LENGTH`, `OLLAMA_NUM_PARALLEL` and `OLLAMA_KEEP_ALIVE`.

The service reaches Ollama at a fixed address of the internal network (`10.213.47.250`), never by
name, so no question or document text can be sent to a host that answers a DNS lookup for
`ollama` while Ollama is down: questions then answer 503 `llm_unavailable` until it is back.
Ollama itself has no route out; the download runs in the separate `ollama-pull` container. If
compose reports that the network `networks have overlapping IPv4` (another network uses
`10.213.47.0/24`, or a second copy of the stack runs), set `BACKEND_NETWORK_PREFIX` (for example
`10.213.48`) in `.env`.

Requirements: Docker Desktop with WSL2 GPU support on Windows, or on Linux the NVIDIA driver plus the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
configured for Docker (`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`).
Check with `docker run --rm --gpus all ubuntu nvidia-smi`. Without a usable GPU, run the stack without
the profile.

## Sizing

Measured on the deployed stack ([benchmark record](BENCHMARKS.md) sections 17-19): one host (Ryzen 7 7700X, 16
threads, 32 GB, RTX 4070 Ti 12 GB); 8 and 4 CPUs are CPU quotas on that host (`cpus:`), not smaller
machines; short synthetic documents (1.8 pages per extraction, 1.7 per ingestion); every setting at
its default, so the concurrency is derived from the CPUs.

> **Before the hardening.** This table was measured before the injection detector existed. With the
> detector on the CPU build, ingestion drops to about 35 docs/min on 16 CPUs (12 clients); on the
> CUDA build it is 133. Extraction does not run the detector. See [Models on the GPU](#models-on-the-gpu).

| CPUs available to the service | 16 | 8 | 4 |
|---|---|---|---|
| Derived jobs / NER pool / embedder pool / OCR workers | 12 / 8 / 4 / 4 | 6 / 4 / 2 / 4 | 3 / 2 / 1 / 4 |
| `/api/extract`, 16 clients: docs/min, p95 | 84.0, 13.3-14.0 s | 62.0, 16.9 s | 37.3, 26.9-27.0 s |
| `/api/ingest`, 16 clients: docs/min, p95 | 90.7, 12.9-13.2 s | 65.9, 16.4-16.7 s | 38.1, 27.9 s |
| Question, `gpu` profile (e4b): p50 / p95 alone | 0.60 / 1.84 s | 0.56 / 1.86 s | not measured |
| ... while 16 clients ingest | 0.84 / 2.31 s | 0.55 / 1.99 s | not measured |
| Service memory peak (extract / ingest) | 2.6 / 3.0 GB | 2.3 / 2.7 GB | 2.2 / 2.5 GB |

With the `gpu` profile, 16 concurrent questions give 128-136 questions/min at p95 9.6-9.7 s
(16 CPUs); `gemma4:e4b-it-qat` with an 8,192-token context and 2 parallel requests uses 5.9-6.5 GB of
VRAM (device total, including about 1.2 GB taken by the desktop). Start (`up -d --wait`): 13.4-13.6 s
without the profile, 15.8-19.4 s with it once the model and its compiled GPU kernels are in the
volume (17.7-17.9 s in the current stack), about 85 s on the very first
start (79.5 s in the current stack; warm-up) plus the model download (about 6 GB). The runtime image is
5.76 GB without the injection detector and 7.89 GB with it (the audit viewer's page and pdf.js add
3.2 MB; the browser tests' image is separate). Above 16
CPUs the derived values stay at the 16-CPU ones: scale with replicas.

## Models on the GPU

With `COMPOSE_PROFILES=gpu` the service is the CUDA build and `DEVICE=auto` puts the recogniser,
the embedder and the detector on the GPU beside Ollama. Measured on the same host with every
defence on ([benchmark record](BENCHMARKS.md) section 23, 60-document sample, 16 CPUs):

| | CPU build | CUDA build, models on the GPU |
|---|---|---|
| `/api/ingest`: docs/min sequential / 12 clients | 9.7 / 35.3 | 20.0 / 132.9 |
| `/api/extract`: docs/min sequential / 12 clients | 15.4 / 86.0 | 19.3 / 127.9 |
| Detector per page / per question | 1,541 / 128 ms | 94 / 19 ms |
| Embedder per page / per query | 551 / 50 ms | 16 / 5.6 ms |
| Recogniser per page (fp32 on both) | 563 ms | 102 ms |
| Question p50 / p95 under 12-client ingestion (Gemma on the GPU) | 1.0 / 2.0 s | 0.9 / 1.6 s (0.6 / 1.1 s alone) |
| VRAM: the three models in service / with Gemma, worst case | none | 2.7-2.9 GB / 9.4 of 12.3 GB |
| Image / service RAM / start to ready | 7.89 GB / 1.2 GB / 8.0-9.8 s | 17.2 GB / 2.2 GB / 9.7-12.0 s |

OCR, which stays on the CPUs, is then 94 % of the ingestion time and the GPU averages 7-16 %
busy, which is why the pool sizes no longer matter there and turning the detector or the
patterns off moves throughput by 0-5 %. The worst case leaves 2.9 GB of the 12 GB free with a desktop on
the GPU. The results equal the CPU build's (the recogniser's spans on all 1,030 corpus pages,
extraction and ingestion bodies on the sample, the detector's flags; e5 vectors within a cosine
of 0.99997). `DEVICE=cpu` makes the CUDA build behave like the
CPU one.

### CPU or GPU

The device changes speed and cost, never the results, so the choice is about the workload:

| The workload | Choose | Why |
|---|---|---|
| Mostly `/api/extract` | CPU | Extraction runs only the recogniser; OCR, which stays on the CPUs, bounds it, so the GPU adds 25 % sequentially and 49 % with 12 clients. |
| Ingestion with the injection detector on, at volume | GPU | The detector costs about 1.5 s per page on the CPU; on the GPU 94 ms. Ingestion goes from 35 to 133 docs/min with 12 clients (x3.8). |
| Ingestion without the detector (no `HF_TOKEN`) | CPU, unless volume is high | About 88 docs/min with 12 clients on 16 CPUs ([benchmark record](BENCHMARKS.md) 22.5). The GPU would still speed up the embedder, but this case was not measured on it. |
| About 1,000 documents per hour or fewer | CPU | 16 CPUs make about 5,200 extractions or 2,100 ingestions with the detector per hour. |
| Answers from a local model | GPU | Gemma needs the GPU anyway; the three models fit beside it (9.4 of 12.3 GB at worst) and slow ingestion by 7 % while questions run. |
| No NVIDIA GPU, or image size and RAM matter | CPU | The CUDA build adds 9.3 GB of image, about 1 GB of RAM and 1.3-2.2 s of start-up. |

The two can be mixed: `COMPOSE_PROFILES=gpu` with `TORCH_VARIANT=cpu` runs Gemma on the GPU and the
service on the CPU build. On the GPU, throughput then grows with CPUs and `OCR_WORKERS`, not with a
larger GPU: it averaged 7-16 % busy. Not measured: long documents, more than 12 concurrent jobs and
several replicas sharing one GPU ([benchmark record](BENCHMARKS.md) 23.10, caveats).

## Storage and failure modes

- `qdrant_data` holds the index: each chunk is a 768-dimension float vector (3 KB) plus its text
  (up to 800 characters) and payload, roughly 5-10 KB per chunk (an estimate, not measured; the
  short test documents give about 5 chunks each), plus Qdrant's write-ahead log, which needs room
  of its own: on a 40 MB volume the first write already failed ("WAL buffer size exceeds available
  disk space"). When the volume is full, ingestion answers 503 `vector_store_unavailable` with "the
  vector store has no disk space left" and the service logs that line; readiness stays 200, because
  Qdrant itself is up.
- `ollama_models` holds the model (`gemma4:e4b-it-qat`: 6.1 GB) and the compiled GPU kernels (at
  most 4 GiB, `CUDA_CACHE_MAXSIZE`): plan about 11 GB, more for other models. When it is full the
  download fails and the stack does not start (above).
- Qdrant down: the service starts, readiness answers 503 naming `vector_store`, requests that need
  the store answer 503 and recover when it is back. Ollama down: questions answer 503
  `llm_unavailable` and recover when it is back.
- After a host reboot Docker restarts the containers in any order and without the start-up
  check; questions answer 503 until Ollama is up.

## Stopping

```bash
docker compose --profile gpu down
```

This keeps indexed documents and downloaded models; add `-v` to delete them too (volumes
`docintel_qdrant_data`, `docintel_ollama_models`). A stop gives running requests up to 30 s to finish.

## Images and models

The images build from one `Dockerfile` (its header comment lists the stages: `model`, `prompt-guard`, `pdfjs`, `base`, `test`, `e2e`, `runtime`). Nothing is downloaded at run time.

The NER model (`Davlan/xlm-roberta-base-ner-hrl`, about 1.1 GB) is baked into the image:
a separate `model` build stage installs a pinned `huggingface_hub`, downloads only the
files `transformers` loads (config, tokenizer files and `model.safetensors`) plus the model
card (`README.md`) at a pinned commit, checks every file's sha256, writes a `NOTICE` with the
licences (model AFL-3.0, base XLM-RoBERTa MIT) and is copied read-only to
`/models/xlm-roberta-base-ner-hrl` (`NER_MODEL_PATH`). The image sets `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`, so nothing is fetched at run time; to check that the whole suite
needs no network at all, run it with networking disabled:

```
docker run --rm --network none docintel-test
```

Tests marked `ner_model` need the model directory and are skipped on a host without it.
Loading refuses weights that do not fit the model's configuration (missing, unexpected or
mismatched tensors) instead of running with random weights. One text sent to the model is
capped at `max_chars` (default 50,000 characters): inference costs about 0.25 s per
1,000 characters at 2 threads and holds the recogniser's lock. To bump the model, set
`NER_MODEL_REVISION` and every `*_SHA256` build arg in the `Dockerfile` to the new commit and
its file hashes (the comment above the `model` stage says where to get them), rebuild and
rerun the suite.

The embedding model (`intfloat/multilingual-e5-base`, MIT, 768 dimensions, about 1.1 GB) is
baked in the same way: the `model` stage downloads `config.json`, `model.safetensors`,
`tokenizer.json`, `sentencepiece.bpe.model`, the two tokenizer configs and the model card
at a pinned commit (never `pytorch_model.bin` or the ONNX/OpenVINO variants), checks every
file's sha256, writes a `NOTICE` (model MIT, base XLM-RoBERTa MIT) and copies it read-only
to `/models/multilingual-e5-base`. The image sets `EMBEDDING_MODEL_PATH` to that directory
and `EMBEDDING_MODEL_REVISION` to the commit (part of the embedding identity, so a new
revision opens a new vector collection). It loads offline in about 5 s; one 800-character
passage (~180 tokens) takes about 0.16 s at 2 threads. To bump it, set the global
`E5_MODEL_REVISION` build arg and every `E5_MODEL_*_SHA256` arg in the `Dockerfile` (same
sources as the NER model), rebuild and rerun the suite.

The injection detector (`meta-llama/Llama-Prompt-Guard-2-86M`, about 1.1 GB) is baked in the
same way by the `prompt-guard` stage (config, `model.safetensors`, the tokenizer files, the
model card, `LICENSE` and `USE_POLICY.md` at a pinned commit, every sha256 checked, a `NOTICE`
with the licence) and copied read-only to `/models/prompt-guard` (`INJECTION_MODEL_PATH`). The
model is gated: the account that builds the image must have accepted the Llama 4 Community
License on its Hugging Face page, and its read token must be in `.env` as `HF_TOKEN` (never
committed). Compose hands it to the build as a BuildKit secret, so it is in no layer and in no
container (`docker history` and `docker inspect` show none). Without `HF_TOKEN` the compose
files build every image without the model (the `INJECTION_MODEL` build argument is then empty,
which the `Dockerfile` treats as 0), and the service runs with the instruction patterns alone;
`INJECTION_MODEL=1` or `0` in `.env` forces either. A plain `docker build` defaults to 1 and
needs the secret, or `--build-arg INJECTION_MODEL=0`. To bump it, run `uv run python scripts/resolve_model_hashes.py
meta-llama/Llama-Prompt-Guard-2-86M <commit>` with `HF_TOKEN` in the environment, paste the
printed `ARG` lines over the `PROMPT_GUARD_*` ones in the `Dockerfile`, rebuild and rerun the
suite. Tests marked `injection_model` need the model directory and are skipped without it. The
`NOTICE` next to the model carries the licence attribution (see the root `README.md`; the licence is
not OSI-approved, [Known limitations](KNOWN_LIMITATIONS.md) S5).

## Torch builds and GPU tests

`torch` comes in two builds, chosen by exactly one of two extras that uv declares conflicting:
`cpu` from the PyTorch CPU wheel index (no NVIDIA libraries) and `cuda` from the CUDA 12.6 index.
A sync needs one of them (`uv sync --extra dev --extra cpu` on the host; without either, no torch is
installed). The images take it from the `TORCH_VARIANT` build argument (`cpu` by default); in
compose, `TORCH_VARIANT` empty means `cuda` with `COMPOSE_PROFILES=gpu` and `cpu` otherwise, and
the CUDA build also reserves the GPU for the service (`docker-compose.torch.yml`). The test image
stays `cpu` unless `TORCH_VARIANT=cuda`. To run the GPU tests on the host, keep a second
environment beside `.venv` so the two builds never replace each other:

```bash
UV_PROJECT_ENVIRONMENT=.venv-cuda uv sync --extra dev --extra cuda
```

```bash
.venv-cuda/Scripts/python.exe -m pytest -m cuda
```

(`.venv-cuda/bin/python` on Linux.) Tests marked `cuda` compare the three adapters on the GPU with
the CPU and are skipped without a visible GPU; with `NER_MODEL_PATH`, `EMBEDDING_MODEL_PATH` and
`INJECTION_MODEL_PATH` set they also compare the real models. In the CUDA test image:
`TORCH_VARIANT=cuda docker compose -f docker-compose.test.yml run --rm tests pytest -m cuda`.
