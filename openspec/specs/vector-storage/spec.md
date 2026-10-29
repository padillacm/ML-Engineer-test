# vector-storage Specification

## Purpose
Defines how chunk vectors are stored and searched in the vector database: isolation by embedding model, unique ids, similarity scores, and behaviour under transient failures.

## Requirements

### Requirement: Collections are isolated by embedding identity
The store SHALL keep vectors of different embedding identities in different collections, named from the identity, and SHALL record the identity with the collection so that opening it with another identity fails with an incompatible-embedding error and changes nothing.

#### Scenario: Model change
- **WHEN** the embedding model is changed and the service starts again
- **THEN** a new collection is used and vectors of the old model are never compared with new queries

#### Scenario: Identity mismatch on an existing collection
- **WHEN** a collection's recorded identity differs from the configured one
- **THEN** opening it fails with an incompatible-embedding error

### Requirement: A lost collection is recreated
When the bound collection no longer exists (for example after the database lost its storage), the store SHALL recreate it with the same identity once and retry the operation, instead of failing every later call.

#### Scenario: Database restarted empty
- **WHEN** the database restarts without its data while the service is running
- **THEN** the next store operation recreates the collection and succeeds

### Requirement: Re-storing a document is idempotent
Storing a document SHALL overwrite its chunks by id without first deleting the document, and SHALL then remove only chunks of that document beyond the new chunk count; each chunk SHALL record the document's total chunk count so that an incompletely stored document can be recognised.

#### Scenario: Concurrent identical writes
- **WHEN** two writers store the same document at the same time
- **THEN** readers never see the document disappear and the result is one copy of each chunk

### Requirement: Chunk ids are unique across documents
Each stored chunk's id SHALL be derived deterministically from its document identifier and sequence, so that chunks of different documents never share an id and re-storing a document replaces exactly its own chunks.

#### Scenario: Same sequence in two documents
- **WHEN** chunk 0 of document A and chunk 0 of document B are stored
- **THEN** both are kept under different ids

### Requirement: Search returns cosine similarity between 0 and 1
Search SHALL return the best-matching chunks, best first, with their document, page, sequence, text and a similarity score between 0 and 1.

#### Scenario: Ranked results
- **WHEN** a query vector equal to a stored chunk's vector is searched
- **THEN** that chunk is first with a score of 1.0 (within rounding)

### Requirement: Transient failures are retried within a time limit
Calls to the vector database SHALL have a configurable per-request timeout and a configurable overall deadline for each database request including its retries (a multi-batch write is several requests, so a long but healthy write is not cut off), and SHALL be retried a bounded number of times with backoff on connection errors, timeouts and HTTP 502, 503 and 504; when retries or the deadline are exhausted the store SHALL fail with a dependency-unavailable or dependency-timeout error.

#### Scenario: Deadline reached
- **WHEN** the database does not answer and the overall deadline elapses before the attempts are used up
- **THEN** the operation fails with a dependency-timeout error no later than the deadline plus one request timeout

#### Scenario: Database briefly unavailable
- **WHEN** the first call fails with a connection error and the second succeeds
- **THEN** the operation succeeds

#### Scenario: Database down
- **WHEN** every attempt fails with a connection error
- **THEN** the operation fails with a dependency-unavailable error after the configured number of attempts

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
