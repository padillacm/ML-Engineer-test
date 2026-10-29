# Spec Delta

## ADDED Requirements

### Requirement: API documentation is served only when enabled
The interactive documentation and the OpenAPI document SHALL be served only when the corresponding setting is enabled; by default they SHALL answer 404 like any unknown path, and the audit viewer SHALL be unaffected.

#### Scenario: Default
- **WHEN** `/docs`, `/redoc` or `/openapi.json` is requested with the default configuration
- **THEN** the response is 404 `not_found`, while the API routes and the audit viewer answer as usual

#### Scenario: Enabled
- **WHEN** the documentation setting is enabled
- **THEN** `/docs` and `/openapi.json` answer 200

### Requirement: API responses carry protective headers
Every response of an API route (`/api/*` and the health routes) SHALL carry `Cache-Control: no-store`, `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer`; the audit viewer keeps its own headers.

#### Scenario: Extraction response headers
- **WHEN** `/api/extract` answers, successfully or with an error
- **THEN** the three headers are present with those values

### Requirement: Rate limits are reported like other errors
A 429 from the rate limiter SHALL use the uniform error body with code `rate_limited` and the request id, SHALL be logged once with its code like any failed request, and SHALL be counted in the access line.

#### Scenario: Limited request
- **WHEN** a request is answered 429
- **THEN** the body is `{"detail": ..., "code": "rate_limited", "request_id": ...}`, `Retry-After` is a positive number of seconds, and the access line records 429

### Requirement: Connections are bounded by the server
The server SHALL limit the number of open connections it accepts at once and SHALL close idle keep-alive connections after a short timeout, so a client holding connections open cannot exhaust it.

#### Scenario: Idle connections
- **WHEN** a client opens connections and sends nothing
- **THEN** each is closed by the server after the keep-alive timeout
