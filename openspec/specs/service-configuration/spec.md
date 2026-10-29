# service-configuration Specification

## Purpose
Defines how the service is configured: one settings object read from the environment, its benchmarked defaults, validation at start-up, secret handling, and single construction of models and clients.

## Requirements

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

### Requirement: Concurrency defaults follow the CPUs available
When `MAX_CONCURRENT_JOBS`, `NER_POOL_SIZE`, `EMBEDDING_POOL_SIZE` or `OCR_WORKERS` is not set, the service SHALL derive it at start-up from the CPUs available to the process (affinity and the container's CPU quota, cgroup v2 or v1) by one rule, capped at the values measured best on 16 CPUs; a value that is set SHALL be used as given. When `TORCH_THREADS` is not set, the service SHALL use 2 threads, at most the CPUs available. Every consumer SHALL use the resolved values.

#### Scenario: Container limited to 8 CPUs
- **WHEN** the service starts with a quota of 8 CPUs and none of these settings
- **THEN** it uses the 8-CPU row of the sizing table and logs no oversubscription warning

#### Scenario: cgroup v1 quota
- **WHEN** the service starts on a cgroup v1 host with a CPU quota of 2 CPUs
- **THEN** the derived values are those of 2 CPUs

#### Scenario: Single CPU
- **WHEN** the service starts with one CPU available and `TORCH_THREADS` unset
- **THEN** it uses one torch thread and logs no oversubscription warning

#### Scenario: Explicit value
- **WHEN** `MAX_CONCURRENT_JOBS` is set
- **THEN** that value is used whatever the CPU count

### Requirement: The concurrency in use is logged at start-up
The service SHALL log the resolved concurrency values and the CPUs available in one line at start-up, and one warning for each pool whose size times `TORCH_THREADS` exceeds the CPUs available.

#### Scenario: Oversubscribed embedder pool
- **WHEN** `EMBEDDING_POOL_SIZE` x `TORCH_THREADS` exceeds the CPUs available
- **THEN** start-up logs one warning naming `EMBEDDING_POOL_SIZE`, and the service starts with the value as set

### Requirement: The keys can be read from files
The language model's key SHALL be readable from the file named by `LLM_API_KEY_FILE` instead of `LLM_API_KEY`, and the vector store's from `QDRANT_API_KEY_FILE` instead of `QDRANT_API_KEY`. Each file SHALL be read once at start-up, without a UTF-8 byte order mark or surrounding whitespace; an empty file SHALL mean no key. A key set together with its file, or a file that cannot be read, is too large or is not UTF-8, SHALL stop start-up naming the variable and the rule, never the path or contents.

#### Scenario: Key mounted as a file
- **WHEN** `LLM_API_KEY_FILE` names a file holding the key and `LLM_API_KEY` is unset or empty
- **THEN** the client sends that key, and the settings' representation masks it

#### Scenario: Empty key file
- **WHEN** `LLM_API_KEY_FILE` names an empty file
- **THEN** the service starts without a key

#### Scenario: Both set
- **WHEN** `LLM_API_KEY` and `LLM_API_KEY_FILE` are both set
- **THEN** start-up fails naming both variables

#### Scenario: File written with a byte order mark
- **WHEN** the key file starts with a UTF-8 byte order mark
- **THEN** the key is read without it

### Requirement: A rejected key names where it came from
A key read from a file that a later check rejects (for example for its characters) SHALL be reported under the file's variable, and a key from its variable under that variable.

#### Scenario: Invalid key from a file
- **WHEN** the key read from `LLM_API_KEY_FILE` contains a space
- **THEN** start-up fails naming `LLM_API_KEY_FILE`, without the key
