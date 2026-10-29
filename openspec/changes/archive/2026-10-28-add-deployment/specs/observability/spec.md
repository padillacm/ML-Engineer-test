# Spec Delta

## MODIFIED Requirements

### Requirement: Every request and use case leaves one outcome line
Each request SHALL produce one access line with method, route template, status and duration in milliseconds, and each use case SHALL produce one outcome line with counts and durations: pages and names found for extraction; document id, chunks stored and whether it was already indexed for ingestion; hits kept, best score, whether it refused and model latency for questions. The access line of a successful liveness or readiness probe SHALL be written at DEBUG, so periodic healthchecks do not flood the log at the default level; every other request, a failing probe included, SHALL be logged at INFO.

#### Scenario: Successful extraction
- **WHEN** a document is extracted
- **THEN** one access line records the route template and status 200, and one outcome line records pages, names found and stage durations

#### Scenario: Healthcheck probes
- **WHEN** `/health` and `/health/ready` answer 200 and the log level is INFO
- **THEN** no access line is written for them, while a client request in between is logged

#### Scenario: Failing readiness probe
- **WHEN** `/health/ready` answers 503
- **THEN** its access line is written at INFO with status 503
