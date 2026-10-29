# Spec Delta

## Purpose

Defines what the service logs, in which format, with which request context, and what it must never log given that documents contain personal data.

## ADDED Requirements

### Requirement: Logs are structured and carry the request id
Logs SHALL be written to standard output, one JSON object per line by default (plain text when configured), each with timestamp, level, logger, message and, inside a request, its request id. The request id SHALL be taken from a well-formed `X-Request-ID` header (1 to 64 characters of letters, digits, `.`, `_`, `-`) or generated, and SHALL be returned in the `X-Request-ID` response header and in error bodies.

#### Scenario: Incoming request id
- **WHEN** a request carries `X-Request-ID: abc-123`
- **THEN** every log line of that request and the response header carry `abc-123`

#### Scenario: Malformed request id
- **WHEN** a request carries an `X-Request-ID` with a line break or 200 characters
- **THEN** it is ignored and a new id is generated

### Requirement: Every request and use case leaves one outcome line
Each request SHALL produce one access line with method, route template, status and duration in milliseconds, and each use case SHALL produce one outcome line with counts and durations: pages and names found for extraction; document id, chunks stored and whether it was already indexed for ingestion; hits kept, best score, whether it refused and model latency for questions.

#### Scenario: Successful extraction
- **WHEN** a document is extracted
- **THEN** one access line records the route template and status 200, and one outcome line records pages, names found and stage durations

### Requirement: Errors are logged with their code
Errors SHALL be logged with their code and category; unexpected exceptions with their stack trace, exception messages redacted.

#### Scenario: Failed request
- **WHEN** an upload is rejected as `not_a_pdf`
- **THEN** one line records the code and category and the access line records 415

### Requirement: Personal data and secrets are not logged
Logs SHALL NOT contain document text, extracted or requested names, questions, answers, source texts, file names or API keys, including inside exception messages; sizes, counts, hashes, document ids and scores MAY be logged.

#### Scenario: Extraction log content
- **WHEN** a document mentioning "María González" is extracted with that name requested
- **THEN** no log line of the request contains "María", "Gonzalez" or "González"

#### Scenario: Exception carrying document text
- **WHEN** an adapter raises an unexpected exception whose message contains document text
- **THEN** no log line contains that text, and the logged trace still shows the exception type and frames

### Requirement: Exception messages are logged only on request
Exception messages SHALL be left out of logs unless `LOG_EXCEPTION_MESSAGES` is enabled, which is meant for local debugging only.

#### Scenario: Debug opt-in
- **WHEN** `LOG_EXCEPTION_MESSAGES=true` and an unexpected exception occurs
- **THEN** its message appears in the logged trace; with the default it does not
