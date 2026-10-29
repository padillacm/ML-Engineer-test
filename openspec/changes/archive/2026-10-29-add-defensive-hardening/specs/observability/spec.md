# Spec Delta

## ADDED Requirements

### Requirement: Guardrail outcomes are logged by count and reason
Ingestion outcome lines SHALL carry the number of flagged chunks; question outcome lines SHALL carry the refusal reason (`question_flagged`, `unsupported`, `output_filtered` among them) and the support share when computed; rate-limited requests SHALL be logged with code `rate_limited`. No flagged text, question or answer SHALL be logged.

#### Scenario: Flagged ingest
- **WHEN** a document with one flagged chunk is ingested
- **THEN** its outcome line has `chunks_flagged` 1 and contains no document text

#### Scenario: Unsupported answer
- **WHEN** an answer is replaced by the refusal as unsupported
- **THEN** the outcome line has the reason `unsupported` and the computed share, and neither the answer nor the context
