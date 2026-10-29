# Spec Delta

## ADDED Requirements

### Requirement: Recognition runs on the configured device
The name recogniser SHALL run its model on the configured device, in full precision (float32) on the GPU as on the CPU, with the pool of pipelines sharing one copy of the weights on that device, and SHALL return the same spans as the CPU path on the sample documents.

#### Scenario: Spans on the GPU
- **WHEN** the sample pages are recognised on the GPU
- **THEN** the spans equal those of the CPU path
