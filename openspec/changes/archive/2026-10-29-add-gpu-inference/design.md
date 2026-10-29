# Design

## Context

See `proposal.md`. The three in-process models are transformers models on CPU torch from the PyTorch CPU wheel index (`pyproject.toml` `[tool.uv.sources]`): the NER recogniser builds token-classification pipelines with `device="cpu"` and a pool of pipelines over one model (`app/adapters/ner/recognizer.py`); the e5 embedder runs `AutoModel` with mean pooling and a pool sharing weights (`app/adapters/embeddings/embedder.py`); the Prompt Guard 2 detector runs `AutoModelForSequenceClassification` with windows and a `FifoSlots` pool (`app/adapters/injection/detector.py`). The composition root derives pool sizes from the CPUs (`concurrency_for`) and passes one `TORCH_THREADS`. Compose's `gpu` profile reserves the RTX 4070 Ti (12 GB) for Ollama only; Gemma e4b takes about 4.8 GB at an 8,192 context, the desktop about 0.9 GB. The CPU baselines are BENCHMARKS 19 (deployed throughput, 16/8/4 CPUs) and 22 (hardening: screening about 1.3 s per page, ingestion 38 against 87 docs/min with 12 clients). This change is applied on top of the uncommitted `add-defensive-hardening` tree; that change is archived first so its `injection-detection` spec exists.

## Goals / Non-Goals

**Goals:** every in-process model on the GPU when present, the CPU path intact and default for the CPU image; per-component impact on the GPU measured and recorded; parity with the CPU results.

**Non-Goals:** OCR on the GPU; multi-GPU or MIG; quantisation below fp16; replacing any model; a GPU-only deployment (the CPU image stays first-class).

## Decisions

### D1. One device setting, resolved once

`DEVICE: Literal["cpu","cuda","auto"] = "auto"`. The composition root resolves it once (`torch.cuda.is_available()`, device 0) and passes a `device: str` ("cpu" or "cuda:0") to the three adapter configs; adapters never probe the hardware themselves. `cuda` without a GPU raises `ConfigurationError("DEVICE: cuda requested but no GPU is visible")`. The start-up line adds `device`, GPU name and total memory. Alternatives: one setting per model (more knobs than the measurement can justify; the per-component measurements toggle components, not devices).

### D2. Half precision on the GPU, adapters otherwise unchanged

On "cuda": models are moved with `.to(device, dtype=torch.float16)`, inputs moved to the device, outputs brought back with `.float().cpu()`; the embedder normalises in fp32; the NER pipeline gets `device=0` and `torch_dtype=float16` (the pipeline's aggregation runs on CPU numpy as before). Softmax scores of the detector are computed in fp32. A `torch.cuda.OutOfMemoryError` inside a call is caught by each adapter's existing failure mapping (its code), followed by `torch.cuda.empty_cache()`. A small shared helper `app/adapters/device.py` holds `model_dtype(device)` and the OOM cleanup so the three adapters agree. Alternatives: fp32 on GPU (2x VRAM for no measured gain), bf16 (the 4070 Ti supports it, but fp16 is what e5 was validated with).

**Outcome (task 4.1):** in fp16 the recogniser's spans equalled the CPU's on the 60-document sample but differed at name boundaries on 6 of the 1,030 raw corpus pages, so by the rule in Risks the recogniser stays in float32 on the GPU (`place(..., half=False)`); the embedder and the detector stay in fp16, their parity held (BENCHMARKS 23).

### D2b. A reserved question slot in the detector

The GPU measurement found that on the CPU path a question waited about 11.7 s for a detector slot behind 12-client ingestion, because the detector, unlike the embedder, had no slot for questions. The port gains `screen_query(text)`; the Prompt Guard adapter runs `FifoSlots(pool + 1, (pool + 1, pool))` with a question lane first, as the embedder does for queries; the pattern, composite and empty detectors score a question as a text. Alternative: a separate detector instance for questions (a second copy of the weights for the same effect).

### D3. GPU pool rule

On the GPU the derived sizes are fixed small numbers to be calibrated by the measurement (start: NER 2, embedder 2, detector 1; `MAX_CONCURRENT_JOBS` keeps the CPU rule because OCR stays on the CPU and bounds jobs), and `TORCH_THREADS` stays for the CPU-side tokenisation. The oversubscription warning is computed only on the CPU path. The calibration follows BENCHMARKS 18's method (smallest pool within the best throughput band).

### D4. Two torch variants by build argument

`pyproject.toml`: `torch` moves to two mutually exclusive extras, `cpu` and `cuda`, each sourced from its own index (`pytorch-cpu`, `pytorch-cu12x` matching the torch 2.14 wheels), declared in `[tool.uv] conflicts`, so `uv.lock` holds both and `uv sync --extra cpu|cuda` picks one. The Dockerfile takes `ARG TORCH_VARIANT=cpu` and syncs with that extra in the `test` and `runtime` stages; the host development environment uses `cpu` by default. The CUDA runtime comes with the torch wheels (NVIDIA libraries as pip dependencies), so the base image stays `python:3.12-slim`. Compose: under the `gpu` profile, the app build passes `TORCH_VARIANT=cuda` and the app gets the same `deploy.resources.reservations.devices` block as Ollama. Alternatives: an `nvidia/cuda` base image (duplicates the CUDA libraries already in the wheels, larger), a separate Dockerfile (drift).

### D5. Measurement plan (the core deliverable)

All on the host's GPU with the stack's Ollama loaded, recorded in a new BENCHMARKS section 23 against the CPU baselines, reusing the harnesses of sections 15-19 and 22 (`benchmarks/deploy_perf`, `benchmarks/adversarial`):

1. **VRAM**: each model alone (after load and peak under its pool), the three together, the three plus Gemma at `OLLAMA_NUM_PARALLEL` 2 during ingestion; margin left on the 12 GB device.
2. **Per-stage time on the GPU** from the outcome lines (OCR, NER, screen, embed, store) on the 60-document sample, sequential and with 12 clients, against the CPU figures.
3. **Per-component impact**, one switch at a time on the GPU image, each cell sequential and 12 clients: everything on; detector model off; patterns off; both detector parts off; guardrails of answering off (support check and output filter, through a benchmark-only setting the harness passes, or measured as the time of `support_share` plus `output_filter` per answer from the outcome lines); NER on CPU with the rest on GPU and the reverse (via a benchmark-only override of the device per adapter, if the adapters take a device each, which D1 provides internally).
4. **Questions**: p50/p95 alone and during 12-client ingestion, with the GPU shared by Gemma and the in-process models; the share of question time spent in embed and screen.
5. **Parity**: extraction and ingestion results on the 60-document sample, CPU against GPU; vector cosine and top-1 agreement on the 750 sample questions; the adversarial suite's flags on both devices.
6. **Images**: sizes of the CPU and CUDA images; cold start on the GPU.

The pool calibration of D3 uses the cells of item 2.

## Risks / Trade-offs

- [The GPU is shared by Gemma and three models; ingestion can slow questions] -> measured in item 4; pool sizes keep in-process load bounded; documented in the sizing table.
- [fp16 changes NER spans or boxes] -> parity test on the sample; if spans differ, NER stays fp32 on the GPU (VRAM allows it) and the reason is recorded.
- [The CUDA image is several GB larger] -> it is optional; sizes recorded.
- [uv's conflicting extras and the lock] -> verified by `uv lock --check` and both builds in the gates.
- [Docker Desktop on WSL2 GPU passthrough differs from Linux] -> the existing `gpu` profile already works on this host; Linux needs the NVIDIA Container Toolkit as the README says.

## Migration Plan

Additive: default `DEVICE=auto` keeps CPU behaviour on CPU-only hosts and the CPU image; the `gpu` profile switches the service to the CUDA image. Rollback: `DEVICE=cpu` or the CPU image.

## Open Questions

None that change the specs or tasks; the GPU pool sizes and the CUDA wheel index name are set during implementation by measurement and by the torch 2.14 wheel listing.
