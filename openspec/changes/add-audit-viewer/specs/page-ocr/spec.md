# Spec Delta

## ADDED Requirements

### Requirement: Words carry their recognition confidence
Each recognised word SHALL carry the confidence the recognition engine reports for it, from 0 to 100. An engine output with a word confidence that is not a number, not finite or above 100 SHALL fail as an OCR failure; words with a negative confidence SHALL be dropped as before.

#### Scenario: Confidence kept
- **WHEN** Tesseract reports the word "Ramírez" with confidence 96.5
- **THEN** the recognised word "Ramírez" carries confidence 96.5

#### Scenario: Confidence above the scale
- **WHEN** the engine output gives a word the confidence 100.5
- **THEN** recognition of the page fails with an OCR failure
