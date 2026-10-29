# Spec Delta

## ADDED Requirements

### Requirement: Recognition runs in parallel without changing results
The recogniser SHALL serve up to its configured pool size of calls at once, sharing one copy of the model weights, and SHALL return the same spans for a text whatever the pool size or the number of concurrent calls.

#### Scenario: Concurrent calls
- **WHEN** 16 threads recognise the 60 sample documents with a pool of 4
- **THEN** every result equals the result with a pool of 1 and no call fails
