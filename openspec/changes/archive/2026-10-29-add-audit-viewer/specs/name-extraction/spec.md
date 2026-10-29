# Spec Delta

## ADDED Requirements

### Requirement: Names read with low confidence throughout are not reported
The service SHALL NOT report a candidate name whose every word was recognised with a confidence below 20 (on the engine's 0-100 scale). A candidate with at least one word at or above 20, or with a word whose engine reported no confidence, SHALL be judged by the other rules only. The text given to the recogniser SHALL NOT change: the rule applies to candidates, not to the words of the page.

#### Scenario: Signature read as letters
- **WHEN** a signature scribble is read as "Pivo Wviy" with word confidences 0 and 12 and the recogniser tags it as a person
- **THEN** no name is reported for it, and a name on the same page read with confidence 95 is reported

#### Scenario: One readable word keeps the name
- **WHEN** a name is read as two words with confidences 0 and 63
- **THEN** it is reported if the other rules accept it
