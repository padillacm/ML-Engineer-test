# Spec Delta

## ADDED Requirements

### Requirement: The guardrail tunables have safe defaults
The detector model path and threshold, the rate limits and per-client cap, the render time budget, the page text limit, the context share for support, the per-document chunk cap and the documentation switch SHALL be settings read like every other tunable, with defaults equal to the code defaults, and the detector path SHALL be optional (unset means no detector).

#### Scenario: Defaults
- **WHEN** the service starts with only the required variables
- **THEN** the limiter, the render budget, the page limit and the support check are active with their defaults, documentation is off and no detector is loaded

#### Scenario: Invalid value
- **WHEN** the render budget is set to zero or the support share above one
- **THEN** start-up fails naming the variable and the rule, not the value
