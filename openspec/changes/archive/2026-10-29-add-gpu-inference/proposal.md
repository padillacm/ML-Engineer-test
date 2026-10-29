# Proposal

## Why

The in-process models run on the CPU: the NER model is the dominant cost of extraction and ingestion (BENCHMARKS 4.4, 15, 19), and the Prompt Guard 2 detector added by `add-defensive-hardening` cuts ingestion throughput on CPU from 86.8 to about 38 documents per minute with 12 clients (about 1.3 s of screening per page, BENCHMARKS 22). The host has an RTX 4070 Ti (12 GB) that only Ollama's Gemma uses (4.8 GB). The user decided to run every in-process model on the GPU and to measure, component by component, what each one costs there, with CUDA optional so the CPU deployment keeps working.

## What Changes

- **Device setting**: `DEVICE` (`cpu`, `cuda`, `auto`; default `auto`) chooses where the recogniser, the embedder and the injection detector run; `auto` uses the GPU when one is visible and falls back to the CPU, logging the device in use. `cuda` without a visible GPU stops the service at start-up with a message naming the variable.
- **Models on the GPU**: the three adapters load their model on the chosen device, in fp16 on the GPU (the e5 fp16 equivalence is already measured: 98.99 % top-1 agreement, BENCHMARKS 5), keep the CPU path unchanged, and keep the contract of each port (same outputs within a stated tolerance, same errors). A CUDA out-of-memory error maps to the adapter's processing-failure code and the next call still works.
- **Concurrency on the GPU**: when the device is the GPU, the pool sizes are derived for it (small pools, larger batches) instead of from the CPUs; `TORCH_THREADS` and the CPU oversubscription warning apply only to the CPU path. Set values still win.
- **CUDA image as an option**: a build argument selects the torch wheel index (`TORCH_VARIANT=cpu`, the default, or `cuda`); the CPU image is unchanged. The `gpu` compose profile builds the CUDA variant and reserves the GPU for the service as well as for Ollama.
- **Measurements on the GPU** (the point of the change): VRAM of each model and of the whole stack next to Gemma; extraction and ingestion throughput and per-stage time (OCR, NER, screening, embedding, store) on the GPU against the CPU baseline of BENCHMARKS 19 and 22, with each component switched in turn (detector model on/off, patterns on/off, the answer guardrails on/off) so the impact of each is isolated; question latency with the GPU shared by Gemma and the in-process models, alone and during ingestion; result parity of extraction and ingestion between CPU and GPU on the 60-document sample.

Out of scope: moving OCR to the GPU (Tesseract has no GPU path), multi-GPU, and changing any model.

## Capabilities

### New Capabilities

- `inference-device`: where the in-process models run, how the device is chosen and reported, what happens without a GPU, and the numerical parity required between devices.

### Modified Capabilities

- `service-configuration`: the `DEVICE` setting, and concurrency defaults derived for the GPU when it is the device.
- `deployment`: the CUDA image variant selected by build argument and the GPU reservation of the service under the `gpu` profile.
- `name-extraction`: recognition runs on the configured device with results equal to the CPU path.
- `document-ingestion`: embedding runs on the configured device within a stated tolerance of the CPU vectors.
- `injection-detection`: screening runs on the configured device with the same flags as the CPU path.

## Impact

- Code: `app/settings.py`, `app/bootstrap.py` (device resolution, GPU pool rule, start-up line), the three adapter configs and adapters (`app/adapters/ner`, `app/adapters/embeddings`, `app/adapters/injection`), a small shared device helper in `app/adapters/`.
- Packaging: `pyproject.toml` (two torch variants as mutually exclusive extras, each from its own wheel index), `uv.lock`, `Dockerfile` (`TORCH_VARIANT` build argument; the CUDA image is larger, recorded), `docker-compose.yml` (`gpu` profile: CUDA build and GPU reservation for the service).
- Tests: device selection and fallback, CUDA-marked parity tests (new `cuda` marker, skipped without a GPU), OOM mapping, GPU pool rule; the existing suites stay green on the CPU image.
- Docs: README (device setting, GPU sizing), `docs/BENCHMARKS.md` new section 23 with the per-component GPU measurements, `docs/KNOWN_LIMITATIONS.md`.
- Builds on the uncommitted `add-defensive-hardening` work in the same tree (the user chose to chain both and commit at the end); its section 22 is the CPU baseline.
