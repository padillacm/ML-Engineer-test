# name-extraction Specification

## Purpose
Defines how person names are found on OCR'd pages, turned into clean name mentions with a box on the page where each is printed, and matched against the names a caller asks for, so a reviewer can verify every reported name in place.

## Requirements

### Requirement: Only person names are extracted
The service SHALL report names of people only. Organisations, places, dates, products, e-mail addresses and other entities SHALL NOT be reported as names.

#### Scenario: Mixed entities
- **WHEN** a page reads "Maria Gonzalez of Acme Corporation met in Madrid on 3 March"
- **THEN** "Maria Gonzalez" is reported and "Acme Corporation", "Madrid" and the date are not

### Requirement: Every mention is reported with its own box
The service SHALL report each printed occurrence of a name separately, with its 0-based page number and one box in PDF points covering exactly the words of that occurrence. Names SHALL be reported in page order and, within a page, in reading order.

#### Scenario: Name repeated on two pages
- **WHEN** "Maria Gonzalez" is printed once on page 0 and once on page 1
- **THEN** two mentions are reported, one per page, each with a box around its own occurrence

#### Scenario: Box covers the name only
- **WHEN** "Dr. Maria Gonzalez, CEO" is printed on a line
- **THEN** the reported name is "Maria Gonzalez" and its box covers those two words, not the title or the role

### Requirement: Upper-case names are recognised
The service SHALL recognise names printed in capital letters as well as names in mixed case.

#### Scenario: Name in capitals
- **WHEN** a page reads "SIGNED BY MARIA GONZALEZ"
- **THEN** a name is reported whose box covers "MARIA GONZALEZ"

### Requirement: Mentions are normalised into clean names
The service SHALL strip leading titles and honorifics, keep hyphenated and apostrophised names whole, join "Last, First" into one name, extend a name to a preceding initial and split a span that crosses a line break. It SHALL NOT report a one-word name unless a title preceded it, nor a candidate that is all lowercase, contains digits or "@", or has fewer than three letters.

#### Scenario: Title stripped
- **WHEN** a page reads "Sra. Ana López"
- **THEN** the reported name is "Ana López"

#### Scenario: Last, First
- **WHEN** a page reads "Gonzalez Ruiz, Maria"
- **THEN** one name covering "Gonzalez Ruiz, Maria" is reported

#### Scenario: Titled surname kept, bare surname dropped
- **WHEN** a page reads "Dr. Weber" in one sentence and "Weber" alone in another
- **THEN** "Weber" is reported for the titled occurrence only

#### Scenario: List of full names
- **WHEN** a page reads "Researchers: Dr. Priya Sharma, Lucas Zimmermann"
- **THEN** two names are reported, "Priya Sharma" and "Lucas Zimmermann", each with its own box

#### Scenario: Longer comma-separated list
- **WHEN** a page reads "Authors: Weber, Hans, Smith, John, Rossi, Marco"
- **THEN** three names are reported, one per "Last, First" pair

#### Scenario: Capitalised surnames before the given names
- **WHEN** a Spanish page reads "Titular: GONZÁLEZ RUIZ, María José, con domicilio en Madrid"
- **THEN** one name covering "GONZÁLEZ RUIZ, María José" is reported

#### Scenario: Title from the detected language
- **WHEN** a French document without a language hint reads "M. Guyon a signé"
- **THEN** the reported name is "Guyon", because the detected language makes "M." a title

#### Scenario: Span across a line break
- **WHEN** the model returns one span covering "Maria Gonzalez" at the end of a line and "Director" at the start of the next
- **THEN** only "Maria Gonzalez" is reported

### Requirement: Extracted names are matched against the requested list
The service SHALL compare each distinct extracted name with the requested name pairs using the name-matching rules and report at most one match per distinct extracted name, in the order the names were first extracted. A request with no pairs SHALL produce no matches and still report the extracted names.

#### Scenario: Repeated name matched once
- **WHEN** "Maria Gonzalez" is extracted three times and the request contains ("María", "González")
- **THEN** one match is reported for "Maria Gonzalez" with score 1.0

#### Scenario: No requested names
- **WHEN** the request contains an empty list of names
- **THEN** the extracted names are reported and the match list is empty

### Requirement: Recognition runs offline with a pinned model
The name recogniser SHALL load its model once, from a configured local path, and SHALL NOT access the network at run time. A missing or unreadable model SHALL fail when the recogniser is created with an invalid-configuration error, not on the first request.

#### Scenario: Model does not match its configuration
- **WHEN** the model directory holds weights that do not fit the declared architecture
- **THEN** creation fails with an invalid-configuration error instead of running with random weights

#### Scenario: Model missing
- **WHEN** the recogniser is created with a model path that does not exist
- **THEN** creation fails with an invalid-configuration error

#### Scenario: Offline
- **WHEN** names are extracted with network access disabled
- **THEN** extraction succeeds

### Requirement: Long pages are fully processed
The service SHALL find names anywhere on a page, including pages whose text exceeds the model's maximum input length.

#### Scenario: Text beyond the configured limit
- **WHEN** a page's text is longer than the configured maximum number of characters
- **THEN** the request fails with a processing failure with code "ner_failed" without running the model

#### Scenario: Name at the end of a long page
- **WHEN** a page has more text than fits in one model input and a name is printed in its last paragraph
- **THEN** that name is reported

### Requirement: Recognition runs in parallel without changing results
The recogniser SHALL serve up to its configured pool size of calls at once, sharing one copy of the model weights, and SHALL return the same spans for a text whatever the pool size or the number of concurrent calls.

#### Scenario: Concurrent calls
- **WHEN** 16 threads recognise the 60 sample documents with a pool of 4
- **THEN** every result equals the result with a pool of 1 and no call fails
