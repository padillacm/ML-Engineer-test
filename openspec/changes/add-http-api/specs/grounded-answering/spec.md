# Spec Delta

## ADDED Requirements

### Requirement: Blank questions are invalid input
A question that is empty or only whitespace SHALL be rejected with an invalid-input error coded `invalid_question` before anything is embedded, searched or sent to the model.

#### Scenario: Whitespace question
- **WHEN** the question is a tab and two spaces
- **THEN** an invalid-input error with code `invalid_question` is raised and neither the embedder nor the store is called
