# page-ocr Specification

## Purpose
Defines how a scanned PDF is turned into recognised words with positions in PDF page space, page by page, so that names can later be located and verified on the page where they appear.

## Requirements

### Requirement: Every page is recognised in order
The OCR engine SHALL return exactly one result per page of the PDF, in page order, with 0-based page numbers, including the first page. A page with no recognisable text SHALL yield a result with no words rather than being skipped.

#### Scenario: Multi-page document
- **WHEN** a three-page scanned PDF is recognised
- **THEN** three results are returned with page numbers 0, 1 and 2 in that order

#### Scenario: First page has text
- **WHEN** a PDF whose first page contains text is recognised
- **THEN** the result for page 0 contains words from that page

#### Scenario: Blank page
- **WHEN** one page of the PDF is blank
- **THEN** its result is present, has the page's size and contains no words

### Requirement: Word boxes are in PDF space
Each recognised word SHALL carry a box in PDF points with origin at the top-left of the page as displayed (after applying the page rotation and relative to its crop box), computed from the actual size of the rendered image and the page rectangle, and clipped to the page. Each result SHALL carry the displayed page width and height in points.

#### Scenario: Known word position
- **WHEN** a PDF page shows the word "Ramírez" drawn at a known rectangle and is recognised
- **THEN** the returned box for "Ramírez" overlaps that rectangle with intersection-over-union of at least 0.5

#### Scenario: Box inside the page
- **WHEN** any page is recognised
- **THEN** every word box lies within that page's width and height

#### Scenario: Resolution does not change positions
- **WHEN** the same page is recognised at 150 DPI and at 300 DPI
- **THEN** the boxes of the same word overlap with intersection-over-union of at least 0.5

### Requirement: Rendering and preprocessing are configurable
The engine SHALL render pages at a configurable resolution, with a default of 200 DPI, and SHALL by default flatten uneven illumination before recognition. Resolution, flattening, page segmentation mode and engine timeout SHALL be configuration values validated when the engine is created.

#### Scenario: Defaults
- **WHEN** the engine is created without options
- **THEN** it renders at 200 DPI with illumination flattening enabled

#### Scenario: Invalid configuration
- **WHEN** the engine is created with a resolution of 0 or a timeout of 0
- **THEN** creation fails with an invalid-configuration error

### Requirement: Language is chosen per document
The engine SHALL support English, Spanish, Portuguese, French and German, identified by two-letter codes. With a language hint, it SHALL read every page with that language only. Without a hint, it SHALL detect the document language from a first pass that uses all supported languages, then read the whole document with the detected language; when detection is inconclusive it SHALL use all supported languages together.

#### Scenario: Explicit language
- **WHEN** a PDF is recognised with the hint "es"
- **THEN** every page is read with the Spanish model only

#### Scenario: Detected language
- **WHEN** a Spanish document is recognised without a hint
- **THEN** the document language is detected as Spanish and every page is read with the Spanish model

#### Scenario: First page does not decide the language
- **WHEN** a document without a hint starts with a page of names and figures followed by Spanish prose
- **THEN** detection keeps reading further pages and the document is read with the Spanish model

#### Scenario: Inconclusive detection
- **WHEN** a document without a hint has too little text to decide its language
- **THEN** every page is read with all supported languages together

#### Scenario: Unsupported language
- **WHEN** a PDF is recognised with the hint "it"
- **THEN** the request fails with an invalid-input error with code "unsupported_language"

### Requirement: Pages are processed concurrently without changing results
The engine SHALL recognise pages concurrently with a configurable upper bound on simultaneous engine processes, each limited to a single thread, and SHALL return the same results in the same order as sequential processing.

#### Scenario: Concurrency preserves order
- **WHEN** a ten-page PDF is recognised with four workers
- **THEN** the results are in page order and equal to those obtained with one worker

### Requirement: Failures are reported with the core error categories
The engine SHALL report a file that cannot be opened as a PDF, or is encrypted, as invalid input with code "pdf_unreadable"; a page whose render would exceed the configured pixel budget as invalid input with code "too_large"; and a missing, failing or timed-out recognition engine as a processing failure with code "ocr_failed". No other exception type SHALL escape the engine.

#### Scenario: Corrupt file
- **WHEN** bytes that start like a PDF but are corrupt are recognised
- **THEN** the request fails with code "pdf_unreadable"

#### Scenario: Truncated file
- **WHEN** a valid PDF cut short so that page objects are missing is recognised
- **THEN** the request fails with code "pdf_unreadable" instead of returning invented blank pages

#### Scenario: No pages
- **WHEN** a PDF with zero pages is recognised
- **THEN** the request fails with code "pdf_unreadable"

#### Scenario: Wrapper process outlives the timeout
- **WHEN** the configured engine command starts child processes and exceeds the timeout
- **THEN** the request fails with code "ocr_failed" and none of those processes keeps running

#### Scenario: Encrypted file
- **WHEN** a password-protected PDF is recognised
- **THEN** the request fails with code "pdf_unreadable"

#### Scenario: Engine times out
- **WHEN** recognising a page exceeds the configured timeout
- **THEN** the request fails with code "ocr_failed" and the engine process is terminated

#### Scenario: Engine not installed
- **WHEN** the recognition engine executable cannot be found
- **THEN** the request fails with code "ocr_failed"

#### Scenario: Huge page
- **WHEN** a page would render to more pixels than the configured budget
- **THEN** the request fails with code "too_large" before rendering it

### Requirement: No resources are leaked
The engine SHALL read the PDF from memory, SHALL NOT write temporary files, and SHALL release the document and rendered images whether recognition succeeds or fails.

#### Scenario: Failure mid-document
- **WHEN** the recognition engine fails on the second page
- **THEN** the document is closed and no engine process remains running

#### Scenario: No temporary files
- **WHEN** a PDF is recognised
- **THEN** no file is created in the system temporary directory or the working directory

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

### Requirement: Words carry their paragraph
Each recognised word SHALL carry the 0-based index of the paragraph it belongs to on its page, as segmented by the recognition engine, numbered in reading order. Words of the same printed paragraph SHALL share an index and words of different paragraphs SHALL NOT.

#### Scenario: Two paragraphs
- **WHEN** a page shows a heading line, a blank gap, and then a two-line paragraph
- **THEN** the heading's words have one paragraph index and the two lines below share a different, larger one

#### Scenario: Paragraph indices restart per page
- **WHEN** a two-page document is recognised
- **THEN** the first paragraph of each page has index 0

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

### Requirement: Words carry their recognition confidence
Each recognised word SHALL carry the confidence the recognition engine reports for it, from 0 to 100. An engine output with a word confidence that is not a number, not finite or above 100 SHALL fail as an OCR failure; words with a negative confidence SHALL be dropped as before.

#### Scenario: Confidence kept
- **WHEN** Tesseract reports the word "Ramírez" with confidence 96.5
- **THEN** the recognised word "Ramírez" carries confidence 96.5

#### Scenario: Confidence above the scale
- **WHEN** the engine output gives a word the confidence 100.5
- **THEN** recognition of the page fails with an OCR failure

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
