# Spec Delta

## ADDED Requirements

### Requirement: Embeddings are computed concurrently
The embedder SHALL serve up to its configured pool size of batches at once, sharing one copy of the model weights and one tokenizer, SHALL give a waiting question's query the next free slot before any waiting passage batch, and SHALL return the same vectors for a text whatever the pool size or the number of concurrent calls, within 1e-6 per component of the single-call result.

#### Scenario: Concurrent embedding
- **WHEN** 16 threads embed the chunks of the 60 sample documents with a pool larger than 1
- **THEN** every vector equals the pool-of-1 vector within tolerance and no call fails

#### Scenario: Question during ingestion
- **WHEN** every slot is busy with passage batches and both passage batches and a query are waiting
- **THEN** the query runs in the next slot that frees up, before the waiting passage batches

#### Scenario: Shutdown
- **WHEN** the embedder is closed while calls run and others wait for a slot
- **THEN** the running calls finish, the waiting ones and every later call fail with `embedding_failed`, and the model is released

### Requirement: Embedding slots are handed out in arrival order
The embedder SHALL hand free slots to waiting queries, and to waiting passage batches, in the order they started waiting, so that a caller that has just released a slot and calls again queues behind the callers already waiting; queries keep their priority over passage batches.

#### Scenario: Waiting calls in arrival order
- **WHEN** every passage slot is busy, passage batches wait in a known order, and a running call with more batches gives its slot back between two of them
- **THEN** the waiting batches run in the order they started waiting, before that call's next batch
