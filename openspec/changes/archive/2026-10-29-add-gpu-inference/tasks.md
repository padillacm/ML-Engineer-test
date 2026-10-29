# Tasks

## 1. Baseline

- [x] 1.1 Archive `add-defensive-hardening` once its tasks are done (its `injection-detection` spec must exist for this change's delta); verify `openspec validate --all --strict`

## 2. Device and adapters

- [x] 2.1 `DEVICE` setting, resolution in the composition root (D1), start-up line with device and GPU, `cuda` without a GPU refused; `app/adapters/device.py` (D2); tests for auto/cpu/cuda resolution with a patched `torch.cuda`, the configuration error, settings defaults, compose and `.env.example` listings
- [x] 2.2 The three adapters on the configured device in fp16 on the GPU (D2), OOM mapped to each adapter's code with cache cleanup; CPU tests unchanged and green; new `cuda` marker (skipped without a GPU) with parity tests: NER spans on sample pages, e5 vectors cosine >= 0.999, detector flags on the adversarial suite, OOM simulation
- [x] 2.3 GPU pool rule (D3) in `concurrency_for`, the oversubscription warning on the CPU path only; tests for both rules

## 3. Packaging

- [x] 3.1 Two torch extras with their indexes and `[tool.uv] conflicts` (D4), `uv.lock`; Dockerfile `TORCH_VARIANT`; compose `gpu` profile builds the CUDA variant and reserves the GPU for the app; verify `uv lock --check`, both image builds, torch's CUDA flag in each, the CPU image without NVIDIA libraries, the deploy wiring tests and the hygiene checks

## 4. Measurements on the GPU

- [x] 4.1 VRAM, per-stage times, per-component impact cells, questions under load, parity, image sizes (D5 items 1-6); calibrate the GPU pool defaults (D3) and apply them with their evidence; `docs/BENCHMARKS.md` section 23 against sections 19 and 22

- [x] 4.2 NER stays fp32 on the GPU after the fp16 span differences (D2 outcome); the detector's reserved question slot (D2b) with its tests; re-measure NER parity, extraction and VRAM, and question latency on the CPU and GPU under load; update section 23

## 5. Documentation

- [x] 5.1 README (device setting, the CUDA build, GPU sizing table with the measured VRAM and throughput), `.env.example`, `docs/KNOWN_LIMITATIONS.md` (what the GPU does not change: OCR bound, shared device)

## 6. Verification

- [x] 6.1 Host gates (ruff, format, mypy, pytest), the container suite on the CPU image, the `cuda`-marked suite in the CUDA image with the GPU, the smoke suite against the `gpu` stack, `openspec validate --all --strict`

## Workflow follow-up

- Commit both changes per block at the end, with the user's approval, then archive this change.
