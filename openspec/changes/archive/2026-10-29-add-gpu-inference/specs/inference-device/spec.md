# Spec Delta

## Purpose

Defines where the service's in-process models (name recogniser, embedder, injection detector) run, how the device is chosen and reported, what happens without a GPU, and the parity required between the CPU and the GPU.

## ADDED Requirements

### Requirement: The device is chosen by configuration
The in-process models SHALL run on the device named by the `DEVICE` setting: `cpu`, `cuda`, or `auto` (the default), which uses the GPU when one is visible to the process and the CPU otherwise. The device in use SHALL be logged once at start-up with the GPU's name and memory when it is the GPU.

#### Scenario: Auto with a GPU
- **WHEN** the service starts with the default setting on a host whose GPU is visible
- **THEN** the three models load on the GPU and the start-up line names the device and the GPU

#### Scenario: Auto without a GPU
- **WHEN** the service starts with the default setting where no GPU is visible
- **THEN** the models load on the CPU, the service works as before, and the start-up line says the CPU is in use

#### Scenario: CUDA requested without a GPU
- **WHEN** `DEVICE=cuda` and no GPU is visible
- **THEN** start-up fails with a configuration error naming `DEVICE`

### Requirement: Results do not depend on the device
Name extraction SHALL return the same names, boxes and matches on the GPU as on the CPU for the sample documents; passage and query vectors SHALL agree with the CPU vectors to a cosine of at least 0.999 and give the same top-1 retrieval on the sample questions; the injection detector SHALL flag the same pages and questions of the adversarial suite on both devices.

#### Scenario: Extraction parity
- **WHEN** the 60-document sample is extracted on the CPU and on the GPU
- **THEN** every response is equal, or every difference is listed and explained in the benchmarks

#### Scenario: Detector parity
- **WHEN** the adversarial suite is screened on both devices
- **THEN** the same cases are flagged

### Requirement: GPU memory errors are processing failures
A GPU out-of-memory error during a model call SHALL fail that request with the adapter's processing-failure code (`ner_failed`, `embedding_failed`, `detector_failed`), SHALL release the call's GPU memory, and SHALL NOT affect later calls.

#### Scenario: Out of memory
- **WHEN** a model call raises a CUDA out-of-memory error
- **THEN** the request fails with the adapter's code and the next call succeeds
