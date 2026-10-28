# Spec Delta

## ADDED Requirements

### Requirement: Words carry their paragraph
Each recognised word SHALL carry the 0-based index of the paragraph it belongs to on its page, as segmented by the recognition engine, numbered in reading order. Words of the same printed paragraph SHALL share an index and words of different paragraphs SHALL NOT.

#### Scenario: Two paragraphs
- **WHEN** a page shows a heading line, a blank gap, and then a two-line paragraph
- **THEN** the heading's words have one paragraph index and the two lines below share a different, larger one

#### Scenario: Paragraph indices restart per page
- **WHEN** a two-page document is recognised
- **THEN** the first paragraph of each page has index 0
