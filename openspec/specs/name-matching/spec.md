# name-matching Specification

## Purpose
Defines how person names found in documents are compared with the name pairs a caller asks for, tolerating accents, letter case, word order and small recognition errors without accepting different people who merely share a surname.

## Requirements

### Requirement: Names are normalised before comparison
Before comparison the core SHALL remove accents and diacritical marks, fold the letters ø, ł, đ, æ, œ and ı to their plain Latin equivalents, ignore letter case, treat every apostrophe variant as the same apostrophe and every hyphen or dash variant as a space, and collapse any run of whitespace into a single space with leading and trailing whitespace removed. Normalisation SHALL be applied identically to extracted names and requested names.

#### Scenario: Accents and case are ignored
- **WHEN** "JOSÉ GARCÍA" is compared with the requested pair first="Jose", last="Garcia"
- **THEN** both normalise to the same text and the score is 1.0

#### Scenario: Accented name against its unaccented request
- **WHEN** "María González" is compared with the requested pair first="Maria", last="Gonzalez"
- **THEN** the score is 1.0

#### Scenario: German sharp s
- **WHEN** "Weiß" and "WEISS" are normalised
- **THEN** both give the same text

#### Scenario: Letters without a decomposition
- **WHEN** "Søren Łukasz" is compared with the requested pair first="Soren", last="Lukasz"
- **THEN** the score is at least 0.90

#### Scenario: Typographic apostrophe
- **WHEN** "Sean O’Brien" (typographic apostrophe) is compared with first="Sean", last="O'Brien"
- **THEN** the score is 1.0

#### Scenario: Hyphenated given name
- **WHEN** "Jean-Luc Picard" is compared with first="Jean Luc", last="Picard"
- **THEN** the score is 1.0

#### Scenario: Irregular whitespace
- **WHEN** "  Maria\n  Gonzalez " is normalised
- **THEN** the result is "maria gonzalez"

### Requirement: Score is a full-string similarity between 0 and 1
The core SHALL score an extracted name against a requested pair by comparing the whole normalised extracted name with the normalised text "first last", ignoring the order of the words, and SHALL report the score as a number from 0.0 to 1.0. A name that only contains the requested name as a part SHALL NOT receive a score inflated by that partial overlap.

#### Scenario: Word order differs
- **WHEN** "Garcia, Jose" is compared with the pair first="Jose", last="Garcia"
- **THEN** the score is at least 0.90

#### Scenario: Extra given name is not a full match
- **WHEN** "Karl-Heinz Müller" is compared with the pair first="Heinz", last="Müller"
- **THEN** the score is below 0.90

#### Scenario: Same surname, different given name
- **WHEN** "Pedro Martinez" is compared with the pair first="Maria", last="Martinez"
- **THEN** the score is below 0.90

### Requirement: Matches are reported only at or above a threshold
The core SHALL report a match only when its score is greater than or equal to a configurable threshold, with a default of 0.90. A threshold outside 0.0 to 1.0 SHALL be rejected as invalid configuration.

#### Scenario: Recognition error within tolerance
- **WHEN** the text read from a scan is "Maria Gonza1ez" and the requested pair is first="Maria", last="Gonzalez"
- **THEN** a match is reported with a score of at least 0.90

#### Scenario: Score exactly at the threshold
- **WHEN** a candidate scores exactly the configured threshold
- **THEN** it is reported

#### Scenario: Stricter threshold
- **WHEN** the threshold is configured to 0.98 and a candidate scores 0.93
- **THEN** no match is reported for it

#### Scenario: Invalid threshold
- **WHEN** the threshold is configured as 1.5
- **THEN** configuration is rejected with an error

### Requirement: Each extracted name yields at most its best match
For every extracted name the core SHALL report the requested pair with the highest score among those at or above the threshold, as the matched name in "first last" form together with the extracted name and the score. When several pairs tie, the one that appears first in the request SHALL be chosen. An extracted name with no pair at or above the threshold SHALL produce no entry.

#### Scenario: Two close candidates
- **WHEN** "Ana Lopez" is matched against the pairs ("Ana", "Lopes") and ("Ana", "Lopez")
- **THEN** one match is reported, for ("Ana", "Lopez") with score 1.0

#### Scenario: Tie
- **WHEN** two identical requested pairs score the same
- **THEN** the first one in the request is reported

#### Scenario: No candidate reaches the threshold
- **WHEN** "Carlos Ruiz" is matched against ("Ana", "Lopez")
- **THEN** the result contains no entry for "Carlos Ruiz"

### Requirement: Degenerate inputs are handled without failure
The core SHALL return an empty result when there are no extracted names or no requested pairs, and SHALL ignore extracted names that are empty after normalisation.

#### Scenario: Nothing extracted
- **WHEN** the list of extracted names is empty
- **THEN** the result is empty

#### Scenario: Blank extracted name
- **WHEN** an extracted name is only whitespace
- **THEN** it produces no entry and does not raise an error
