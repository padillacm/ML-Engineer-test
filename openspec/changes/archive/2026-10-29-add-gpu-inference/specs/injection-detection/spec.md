# Spec Delta

## ADDED Requirements

### Requirement: Screening runs on the configured device
The injection detector SHALL score on the configured device, in half precision on the GPU, with the same flags as the CPU path on the adversarial suite, and its cost per page and per question on the GPU SHALL be measured and recorded.

#### Scenario: Screening cost on the GPU
- **WHEN** the 60-document sample is ingested on the GPU with the detector on
- **THEN** the screening time per page is recorded next to the CPU figure

### Requirement: Questions are screened ahead of pages
The detector SHALL keep one slot beyond its pool for questions, SHALL give a waiting question the next free slot before any waiting page batch, and SHALL hand slots to waiting questions, and to waiting page batches, in arrival order; a question's score SHALL equal the score the same text gets as a page.

#### Scenario: Question during ingestion
- **WHEN** every pool slot is busy with page batches and more page batches wait
- **THEN** a question is scored in the reserved slot without waiting for them

#### Scenario: Question waiting with pages
- **WHEN** the reserved slot is also busy and both a question and page batches wait
- **THEN** the question runs in the next slot that frees up, before the waiting page batches
