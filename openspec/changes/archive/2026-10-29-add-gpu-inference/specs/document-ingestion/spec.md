# Spec Delta

## ADDED Requirements

### Requirement: Embedding runs on the configured device
The embedder SHALL compute passage and query vectors on the configured device, in half precision on the GPU with normalisation in full precision, keeping the embedding identity unchanged, so a collection written on one device is searched from the other.

#### Scenario: Index built on the CPU, searched on the GPU
- **WHEN** documents ingested by a CPU service are searched by a GPU service
- **THEN** the collection opens without an identity mismatch and the sample questions get the same top-1 chunk
