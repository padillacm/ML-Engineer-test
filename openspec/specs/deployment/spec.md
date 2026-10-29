# deployment Specification

## Purpose
Defines how the service is packaged and run in production-like conditions: the runtime image, the compose stack with its optional local language model, configuration through `.env`, network exposure, start order and health, and the smoke tests that check a running stack.

## Requirements

### Requirement: The runtime image contains only what the service needs
The runtime image SHALL contain the service code, its runtime dependencies, Tesseract with its language data and the pinned models, and SHALL NOT contain tests, development dependencies, build tools (including `uv`) or `.env` files. It SHALL run as a non-root user under an init process, with the models read-only and model downloads disabled. No `.env` file of any directory of the build context SHALL enter an image.

#### Scenario: Image hygiene
- **WHEN** the runtime image is inspected
- **THEN** `pytest` and `pymupdf` cannot be imported, `uv` and `uvx` are absent, there is no `tests` directory and no `.env` file under `/app`, the process user id is 10001 and Hugging Face offline mode is set

#### Scenario: Orphaned subprocesses
- **WHEN** a Tesseract subprocess outlives its worker
- **THEN** the init process reaps it, so no zombie process remains

### Requirement: The stack starts with one command
`docker compose up` SHALL start the service and a persistent vector store; with `COMPOSE_PROFILES=gpu` (in `.env` or the environment) the same command SHALL also start a local language model on the GPU, download the configured model unless its volume already holds it, and point the service at it. Without the profile and without a configured model the service SHALL start and answer questions that need a model with 503 `llm_unavailable`.

#### Scenario: Stack without a model
- **WHEN** the stack is started without the `gpu` profile and without `LLM_BASE_URL`
- **THEN** readiness reports the language model as not configured, extraction and ingestion work, and an answerable question gets 503 `llm_unavailable`

#### Scenario: Stack with the local model
- **WHEN** the stack is started with `COMPOSE_PROFILES=gpu`
- **THEN** the model is downloaded once, kept in a named volume, and an answerable question is answered from the indexed documents

#### Scenario: Switching the profile on
- **WHEN** a stack started without the profile is started again with `COMPOSE_PROFILES=gpu`
- **THEN** the service is recreated pointed at the local model and readiness reports it configured

#### Scenario: Warm volume offline
- **WHEN** the models volume already holds the configured model and the download step has no network
- **THEN** the download step succeeds without contacting the registry

### Requirement: The profile on the command line alone is refused
Selecting the `gpu` profile only with `--profile gpu`, which cannot point the service at the model, SHALL fail with a message naming `COMPOSE_PROFILES`.

#### Scenario: Profile given only on the command line
- **WHEN** the stack is started with `--profile gpu` and without `COMPOSE_PROFILES`
- **THEN** the start fails with a message naming `COMPOSE_PROFILES`, and no service runs that claims a model

### Requirement: The local model is usable before the service starts
With the local model, the service SHALL start only after a check has found the configured model on the model server and the model has answered one short prompt (so it is loaded before the first question). When the download fails, the model is missing or the server does not answer within a bounded time, the service SHALL NOT be started and the failure SHALL name its cause.

#### Scenario: Failed download
- **WHEN** the model download fails (a wrong model name, a full volume, no network)
- **THEN** `docker compose up` fails, the service container is not started, and the download step's log names the error

#### Scenario: Model missing from the server
- **WHEN** the download step succeeded but the server does not have the configured model
- **THEN** the check exits with a message naming the model and the download step, and the service is not started

### Requirement: The service reaches the local model without a name lookup
The service SHALL reach the stack's language model server at a fixed address of the internal network, so that no request for the model is resolved through DNS. While the model server is down, questions SHALL fail with 503 `llm_unavailable` and SHALL be answered again once it is back, without restarting the service. The serving model server SHALL have no route out of the stack; downloads SHALL run in a separate step.

#### Scenario: Model server down
- **WHEN** the model server is stopped, the service is restarted, and a host answering to the name `ollama` exists on another network of the service
- **THEN** a question gets 503 `llm_unavailable`, that host receives no request, and after the model server's start the next question is answered

### Requirement: Thinking is off by default for the local model
When the service uses the stack's local model, the reasoning effort SHALL be `none` unless `LLM_REASONING_EFFORT` is set, whichever way the service was pointed at it.

#### Scenario: Explicit local URL
- **WHEN** `LLM_BASE_URL` is set to the stack's model address and `LLM_REASONING_EFFORT` is unset
- **THEN** the model receives `reasoning_effort: none`

### Requirement: Indexed documents survive restarts
The vector store SHALL keep its data in a named volume, so documents indexed before a restart or a recreation of the containers are still found after it.

#### Scenario: Restart
- **WHEN** a document is ingested, the stack is stopped and started again, and the same document is ingested
- **THEN** the second ingestion reports `already_indexed`

### Requirement: Configuration comes from an environment file
Every setting of the stack SHALL be readable from a `.env` file, with a committed `.env.example` that lists each variable of the settings and of the compose files with a placeholder or its default, says that values containing `$` are single-quoted, and contains no secret. The stack SHALL start without a `.env` file, and secrets from it SHALL never appear in the image, in compose output of the service or in logs.

#### Scenario: Example file
- **WHEN** `.env.example` is checked
- **THEN** it lists every setting the service reads and every variable of the compose files, and every secret is empty

### Requirement: Only the service's own settings reach its container
The service container SHALL receive only the settings the service reads, an empty value meaning the setting's default, and never other variables of `.env`.

#### Scenario: Unrelated variable in .env
- **WHEN** `.env` holds a variable the service does not read
- **THEN** it is not in the service container's environment, in `docker inspect` or in `docker compose config`

### Requirement: The keys reach the services only as secret files
A language model key or a vector store key placed in `.env` (or the shell) as `LLM_API_KEY` or `QDRANT_API_KEY` SHALL reach the service as a compose secret mounted under `/run/secrets`, read through `LLM_API_KEY_FILE` or `QDRANT_API_KEY_FILE`, and SHALL NOT be in any container's environment as `docker inspect` shows it. With `QDRANT_API_KEY` set, the stack's vector store SHALL require it. Without a key the stack SHALL start, and the service SHALL have no key.

#### Scenario: Key in .env
- **WHEN** `.env` sets `LLM_API_KEY` and the stack is started
- **THEN** `docker inspect` of the service container and `docker compose config` show no key value, the logs do not contain it, and the service sends it to the language model

#### Scenario: Vector store key
- **WHEN** `.env` sets `QDRANT_API_KEY`
- **THEN** the vector store refuses requests without it, the service works with it, and the key is in no `docker inspect`, compose output or log

#### Scenario: Key with a dollar sign
- **WHEN** a key containing `$` is written in single quotes in `.env`
- **THEN** the service sends exactly that key

#### Scenario: No key
- **WHEN** the stack is started without `.env`, or with the key unset or empty
- **THEN** the stack starts and the service runs without a key

### Requirement: Only the service is exposed, on the local host by default
Only the service SHALL publish a port, bound to the loopback interface unless configured otherwise; the vector store and the language model SHALL be reachable only from the stack's internal network.

#### Scenario: Ports
- **WHEN** the stack runs with the defaults
- **THEN** the service answers on `127.0.0.1` and no port of the vector store or the language model is published

### Requirement: Start order and health are gated
The service SHALL start only after the vector store is ready and, with the local model, after the model check has succeeded. The service's healthcheck SHALL report healthy only when readiness succeeds, and a stop SHALL let running requests finish within the graceful-shutdown timeout.

#### Scenario: Waiting for dependencies
- **WHEN** `docker compose up --wait` is run
- **THEN** it returns only when the service is healthy, and the first request after it succeeds

### Requirement: A running stack can be smoke-tested
A smoke test suite SHALL check a running stack over HTTP: readiness, extraction on the sample PDFs, ingestion twice, an answerable and an off-topic question, an invalid upload and the request id. It SHALL be skipped unless the address of a running service is given.

#### Scenario: Smoke run
- **WHEN** the smoke suite runs against the stack started with the `gpu` profile
- **THEN** every check passes

### Requirement: The images carry the pinned viewer library
The test and runtime images SHALL contain the pinned pdf.js library, worker and licence, fetched at build time with a sha256 check and read-only for the service user; nothing SHALL be downloaded at run time.

#### Scenario: Image hygiene with the viewer
- **WHEN** the runtime image is inspected
- **THEN** the pdf.js files are present with the pinned hash, owned by root and not writable by uid 10001

### Requirement: The browser of the e2e image is pinned by hash
The e2e test image SHALL contain the Chrome Headless Shell that its Playwright version pins, fetched at build time with a sha256 check; the runtime and test images SHALL contain no browser.

#### Scenario: Tampered browser archive
- **WHEN** the e2e stage is built with a browser archive whose sha256 differs from the pin
- **THEN** the build fails

### Requirement: Containers run with least privilege
The service and vector-store containers SHALL run with a read-only root filesystem (writable scratch on tmpfs and the data volume only), every Linux capability dropped, no privilege escalation, a process-count limit and a memory limit, and the service SHALL work fully under them.

#### Scenario: Hardened stack
- **WHEN** the stack runs with the defaults
- **THEN** the smoke suite passes, the service container reports a read-only root, no capabilities and `no-new-privileges`, and a write outside tmpfs and the volumes fails

### Requirement: The detector model is built in with a secret
The image SHALL include the detector model at a pinned revision with sha256-checked files and its licence notice, downloaded at build time with a build secret that never enters a layer; a build without the secret or with the model opted out SHALL succeed and produce an image whose service runs with injection detection off.

#### Scenario: Build with the token
- **WHEN** the image is built with the Hugging Face token secret
- **THEN** the model files are present read-only under the models directory, their hashes match, the NOTICE carries the licence attribution, and no layer or environment variable contains the token

#### Scenario: Build without the token
- **WHEN** the image is built with the model opted out
- **THEN** the build succeeds and the service starts with injection detection off

### Requirement: Dependencies are audited before a push
The known-vulnerability audit of the locked dependencies SHALL run as part of the pre-push checks, failing the push on a known vulnerability without an accepted exception.

#### Scenario: Audit on push
- **WHEN** the pre-push hooks run
- **THEN** the audit runs against the lock file and its result decides the push

### Requirement: The CUDA image is a build option
The image SHALL be built with the CPU torch build by default and with the CUDA build when a build argument selects it; both SHALL contain the same service code and models, and the CPU image SHALL NOT contain CUDA libraries.

#### Scenario: Default build
- **WHEN** the image is built without the argument
- **THEN** torch reports no CUDA support and no NVIDIA library is installed

#### Scenario: CUDA build
- **WHEN** the image is built with the CUDA variant and run with a GPU
- **THEN** torch sees the GPU and the service starts on it

### Requirement: The service shares the GPU under the gpu profile
With the `gpu` profile the service SHALL be built with the CUDA variant and SHALL reserve the GPU next to the language model server, and the whole stack's GPU memory SHALL stay within the device with the measured margin recorded.

#### Scenario: Stack on the GPU
- **WHEN** the stack starts with `COMPOSE_PROFILES=gpu`
- **THEN** readiness passes, the service reports the GPU as its device, and the smoke suite passes
