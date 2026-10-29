# Spec Delta

## MODIFIED Requirements

### Requirement: Recognition runs in parallel without changing results
The recogniser SHALL serve up to its configured pool size of calls at once, sharing one copy of the model weights, SHALL hand free pipelines to waiting calls in the order they started waiting (a caller that has just released a pipeline and calls again queues behind them), and SHALL return the same spans for a text whatever the pool size or the number of concurrent calls.

#### Scenario: Concurrent calls
- **WHEN** 16 threads recognise the 60 sample documents with a pool of 4
- **THEN** every result equals the result with a pool of 1 and no call fails

#### Scenario: Waiting calls in arrival order
- **WHEN** every pipeline is busy, calls wait in a known order, and a job that recognises page after page gives its pipeline back between two pages
- **THEN** the waiting calls run in the order they started waiting, before that job's next page
