# Spec Delta

## ADDED Requirements

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
