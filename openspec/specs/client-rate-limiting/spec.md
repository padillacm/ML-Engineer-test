# client-rate-limiting Specification

## Purpose
Defines how the service, without authentication, limits what one client address may ask of it over time and at once, and how a client that exceeds a limit is answered so it can retry.

## Requirements

### Requirement: Requests are rate-limited per client address
The service SHALL limit the requests of each client address with a sustained rate and a burst, separately for the upload routes (extraction and ingestion) and for questions, both configurable; health probes SHALL NOT be limited. A request over the limit SHALL be answered 429 with code `rate_limited`, the uniform error body and a `Retry-After` header, before its body is read and before any job slot is taken.

#### Scenario: Burst exhausted
- **WHEN** a client sends more upload requests in a minute than its burst and rate allow
- **THEN** the excess requests are answered 429 `rate_limited` with `Retry-After`, no OCR runs for them, and the earlier requests complete normally

#### Scenario: Route classes are independent
- **WHEN** a client has exhausted its question allowance
- **THEN** its extraction requests are still accepted, and the other way round

#### Scenario: Other clients unaffected
- **WHEN** one client address is being limited
- **THEN** requests from another address are served as usual

#### Scenario: Probes are never limited
- **WHEN** `/health` and `/health/ready` are called at any rate
- **THEN** none is answered 429

### Requirement: Concurrent jobs are capped per client address
At most a configured number of extraction and ingestion jobs of one client address SHALL run or wait at once; a further one SHALL be answered 429 `rate_limited` with `Retry-After` without waiting for a job slot, so one client cannot hold every slot of the service.

#### Scenario: One client floods the jobs
- **WHEN** the per-client cap is 2 and a client has two jobs running and sends a third
- **THEN** the third is answered 429 at once, the two jobs complete, and a job from another address still gets a slot

### Requirement: Limits are disabled by configuration, not by code
Setting a rate or cap to zero SHALL turn that limit off; the defaults SHALL allow the measured single-client throughput of the service with margin, so a well-behaved client is never limited by default.

#### Scenario: Limit off
- **WHEN** the upload rate is configured as zero
- **THEN** no upload request is answered 429, whatever its rate

#### Scenario: Defaults and the benchmark client
- **WHEN** the benchmark harness drives the service from one address at the measured throughput
- **THEN** no request is answered 429 with the default limits
