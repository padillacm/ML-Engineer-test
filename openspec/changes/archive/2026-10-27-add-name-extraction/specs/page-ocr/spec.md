# Spec Delta

## ADDED Requirements

### Requirement: Words carry their text line
Each recognised word SHALL carry the 0-based index of the text line it belongs to on its page, as segmented by the recognition engine, with lines numbered in reading order. Words on the same printed line SHALL share an index and words on different lines SHALL NOT.

#### Scenario: Two lines
- **WHEN** a page shows "Maria Gonzalez" on one line and "Director" on the next
- **THEN** "Maria" and "Gonzalez" have the same line index and "Director" has a different, larger one

#### Scenario: Line indices restart per page
- **WHEN** a two-page document is recognised
- **THEN** the first line of each page has index 0

### Requirement: Results report the language used
Each page result SHALL carry the two-letter code of the language the page was read with: the hint when one was given, the detected language otherwise, or no language when the combined model was used because detection was inconclusive.

#### Scenario: Detected language reported
- **WHEN** a French document is recognised without a hint
- **THEN** every page result reports "fr"

#### Scenario: Inconclusive detection
- **WHEN** detection is inconclusive and the combined model is used
- **THEN** the page results report no language
