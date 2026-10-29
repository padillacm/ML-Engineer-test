# llm-client Specification

## Purpose
Defines how the service calls an OpenAI-compatible chat model, local or hosted, with bounded latency, retries for transient failures and errors that never expose secrets.

## Requirements

### Requirement: Any OpenAI-compatible endpoint can be used
The client SHALL send chat completions to a configurable base URL with a configurable model name and an optional API key sent as a bearer token, so that a local Ollama server and hosted providers are interchangeable by configuration only.

#### Scenario: Local model without key
- **WHEN** the client is configured with an Ollama base URL and no API key
- **THEN** requests are sent without an authorization header

#### Scenario: Hosted model with key
- **WHEN** the client is configured with an API key
- **THEN** requests carry it as a bearer token

### Requirement: Calls are bounded in time and retried when transient
Each request SHALL have configurable connect and read timeouts. Connection errors, timeouts, HTTP 429 and HTTP 5xx SHALL be retried a configurable number of times with exponential backoff and jitter, honouring a Retry-After header when present; other HTTP errors SHALL fail immediately.

#### Scenario: Rate limited once
- **WHEN** the first response is 429 and the second is 200
- **THEN** the reply of the second response is returned

#### Scenario: Overall deadline
- **WHEN** the server keeps answering slowly or asking to retry later
- **THEN** the call fails with a dependency-timeout error once the configured overall deadline has passed, whatever the remaining attempts

#### Scenario: Bad request
- **WHEN** the response is 400
- **THEN** the call fails without retrying

#### Scenario: Always timing out
- **WHEN** every attempt times out
- **THEN** the call fails with a dependency-timeout error after the configured attempts

### Requirement: Errors never expose secrets or prompts
Errors raised by the client SHALL be dependency-unavailable or dependency-timeout errors whose messages contain neither the API key nor the prompt text, and the configured key SHALL NOT appear in the client's text representation.

#### Scenario: Key not leaked
- **WHEN** a request configured with an API key fails
- **THEN** neither the error text nor the client's representation contains the key

#### Scenario: Unusable key rejected at configuration
- **WHEN** the API key contains whitespace, control characters or non-ASCII characters
- **THEN** configuration fails with an invalid-configuration error that does not contain the key

#### Scenario: Malformed response
- **WHEN** the server returns 200 with a body that has no message content
- **THEN** the call fails with a dependency-unavailable error

### Requirement: The request adapts to the model's parameters
The client SHALL send `reasoning_effort` only when a reasoning effort is configured, SHALL name the reply limit `max_tokens` or `max_completion_tokens` as configured, and SHALL leave `temperature` out when it is configured not to send it. With the defaults the request is unchanged.

#### Scenario: Thinking turned off on a local server
- **WHEN** the reasoning effort is set to `none`
- **THEN** the request carries `reasoning_effort: "none"`

#### Scenario: Hosted reasoning model
- **WHEN** the limit field is `max_completion_tokens` and temperature is not sent
- **THEN** the request has `max_completion_tokens` and neither `max_tokens` nor `temperature`

### Requirement: Rejected requests and cut-off replies are diagnosable
A rejected request SHALL fail with `llm_unavailable` and an error message that names the HTTP status and the provider's error code when the body has one, validated as a short identifier, never the provider's message or the body. A reply that stopped at the token limit with empty content SHALL be logged as its own outcome with the configured model name and token limit as fields, without its content.

#### Scenario: Unsupported parameter
- **WHEN** the server answers 400 with error code `unsupported_parameter`
- **THEN** the error message contains the status and `unsupported_parameter`, and nothing else from the body

#### Scenario: Empty reply at the token limit
- **WHEN** the reply has `finish_reason` `length` and empty content
- **THEN** a warning names that outcome and carries the model name and the token limit as fields, and the caller still receives the empty content
