# Spec Delta

## Purpose

Defines how a scanned PDF is turned into recognised words with positions in PDF page space, page by page, so that names can later be located and verified on the page where they appear.

## ADDED Requirements

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
