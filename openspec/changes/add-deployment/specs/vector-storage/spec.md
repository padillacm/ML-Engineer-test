# Spec Delta

## ADDED Requirements

### Requirement: Concurrent first use binds the collection once
Concurrent first calls that bind the store to a collection SHALL all succeed: within one process they SHALL be serialised, and a create that finds the collection already made by another process SHALL verify it instead of failing, waiting a bounded time while the new collection does not yet serve reads.

#### Scenario: Simultaneous first ingestions
- **WHEN** twelve ingestions start at the same time against an empty store
- **THEN** every one succeeds and exactly one collection exists

#### Scenario: Replicas binding at the same time
- **WHEN** several processes bind to the same collection of an empty store at the same time
- **THEN** each one verifies the collection that one of them created, and exactly one collection exists

### Requirement: A full store disk is named in the error
When the vector store rejects a write because its storage has no space left, the service SHALL fail the call with `vector_store_unavailable` and a message saying that the vector store has no disk space left, SHALL NOT retry it, and SHALL log one error line saying so, without the store's own message.

#### Scenario: Qdrant volume full
- **WHEN** an ingestion's upsert is answered with Qdrant's "No space left on device" error
- **THEN** the request gets 503 `vector_store_unavailable` with the message "the vector store has no disk space left", and the log holds that line
