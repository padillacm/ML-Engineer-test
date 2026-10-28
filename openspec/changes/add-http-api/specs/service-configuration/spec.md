# Spec Delta

## Purpose

Defines how the service is configured: one settings object read from the environment, its benchmarked defaults, validation at start-up, secret handling, and single construction of models and clients.

## ADDED Requirements

### Requirement: All tunables come from one settings object
Hosts, ports, model paths and revisions, thresholds, DPI, timeouts, retry counts, limits, concurrency and logging SHALL be read from environment variables (optionally a `.env` file) into one settings object, and no module other than the settings module SHALL read the environment for them. Every value SHALL have the benchmarked default except those that identify a deployment (model paths, vector store location), which SHALL be required.

#### Scenario: Override by environment
- **WHEN** `RAG_THRESHOLD=0.75` and `OCR_DPI=300` are set
- **THEN** the question use case refuses below 0.75 and pages are rendered at 300 DPI

#### Scenario: Defaults
- **WHEN** only the required variables are set
- **THEN** the effective values equal the defaults recorded in `docs/BENCHMARKS.md`

### Requirement: Invalid configuration stops the service at start-up
An invalid or missing required value SHALL stop the service before it accepts requests, with a message that names the variable and the rule it breaks.

#### Scenario: Bad threshold
- **WHEN** `MATCH_THRESHOLD=1.5` is set
- **THEN** start-up fails naming `MATCH_THRESHOLD`

#### Scenario: Missing model
- **WHEN** `NER_MODEL_PATH` points to a directory without a model
- **THEN** start-up fails naming `NER_MODEL_PATH`

### Requirement: Secrets are never shown
The language-model and vector-store API keys SHALL be held as secrets: they SHALL not appear in the settings' representation, in logs, in error messages or in responses.

#### Scenario: Settings printed
- **WHEN** the settings object is logged or rendered with `repr`
- **THEN** the keys are masked

### Requirement: Models and clients are built once per process
The service SHALL build every model, engine and client once when it starts, share them between requests, and close them when it stops; the language model SHALL be optional, and without it the service SHALL still start.

#### Scenario: Two requests
- **WHEN** two extraction requests are served
- **THEN** the NER model was loaded once, at start-up

#### Scenario: No language model
- **WHEN** no language-model base URL is set
- **THEN** the service starts and readiness reports the language model as not configured
