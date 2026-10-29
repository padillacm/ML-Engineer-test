# Spec Delta

## ADDED Requirements

### Requirement: A page render has a time budget
Rendering one page SHALL be stopped once a configurable time budget (default 10 s) is spent, and the document SHALL then fail with `pdf_unreadable` naming the page, so no page can hold the renderer for longer than the budget; a page that renders within the budget SHALL render exactly as before.

#### Scenario: Hostile page
- **WHEN** a page whose content takes longer than the budget to draw is rendered
- **THEN** rendering stops within the budget plus one progress step, the error is `pdf_unreadable`, and another document's rendering proceeds afterwards

#### Scenario: Ordinary page
- **WHEN** a sample page is rendered with the default budget
- **THEN** the pixels equal those of an unbounded render

### Requirement: Page text is limited
The recognised text of one page SHALL be limited to a configurable number of characters (default 30,000); a page with more SHALL fail the document as invalid input coded `too_large`, naming the page, before any name recognition, chunking or embedding of the document runs.

#### Scenario: Page over the limit
- **WHEN** a page's recognised text has one character more than the limit
- **THEN** the request fails 413 `too_large` and neither the recogniser nor the embedder is called

#### Scenario: Dense but legitimate page
- **WHEN** the densest page of the sample corpus is recognised with the default limit
- **THEN** it is accepted
