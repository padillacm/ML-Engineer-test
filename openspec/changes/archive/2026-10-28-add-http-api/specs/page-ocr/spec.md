# Spec Delta

## ADDED Requirements

### Requirement: Page count is limited
The OCR engine SHALL reject a document with more pages than its configured maximum (default 50) with an invalid-input error coded `too_many_pages`, right after opening it and before any page is rendered.

#### Scenario: One page over the limit
- **WHEN** a 51-page PDF is recognised with the default maximum
- **THEN** `too_many_pages` is raised and no page is rendered or sent to Tesseract

#### Scenario: At the limit
- **WHEN** a PDF with exactly the maximum number of pages is recognised
- **THEN** every page is recognised

### Requirement: Declared page count is verified
The engine SHALL verify the declared page count without loading or parsing any page, checking only page index `max_pages` (and, when it is missing, the last declared page) beyond the limit. A declared count larger than the pages the document holds SHALL not cause a refusal, and a damaged page followed by readable ones SHALL be `pdf_unreadable`.

#### Scenario: Overstated page count
- **WHEN** a one-page PDF whose page tree declares a million pages is recognised
- **THEN** its one page is recognised and the document is not refused

#### Scenario: Damaged page at the limit
- **WHEN** a 60-page PDF whose page index 50 cannot be loaded while its last page can is recognised with the default maximum
- **THEN** `pdf_unreadable` is raised and nothing is recognised

#### Scenario: Damaged middle page
- **WHEN** page 2 of a 3-page PDF cannot be loaded while page 3 can
- **THEN** `pdf_unreadable` is raised and nothing is recognised
