# Spec Delta

## Purpose

Defines how the text recognised from scanned pages is cleaned and split into overlapping, word-safe chunks that keep a reference to their page, so passages can be embedded, retrieved and traced back to where they came from.

## ADDED Requirements

### Requirement: Recognised text is cleaned before splitting
The core SHALL join a word that was hyphenated across a line break when the hyphen ends a line and the next line starts with a lowercase letter, SHALL turn a single line break inside a paragraph into a space, and SHALL treat one or more blank lines as a paragraph separator. Other hyphens SHALL be kept.

#### Scenario: Word split across lines
- **WHEN** the text is "the docu-\nment was signed"
- **THEN** the cleaned text is "the document was signed"

#### Scenario: Hyphenated name is preserved
- **WHEN** the text is "Karl-\nHeinz Müller" where the next line starts with an uppercase letter
- **THEN** the hyphen is kept and the line break becomes a space

#### Scenario: Lines of one paragraph
- **WHEN** the text is "first line\nsecond line\n\nnext paragraph"
- **THEN** the cleaned text has two paragraphs, "first line second line" and "next paragraph"

### Requirement: Chunks respect a configurable maximum size
The core SHALL split each page into chunks that do not exceed a configurable maximum number of characters, with a default of 800. A page whose cleaned text fits in the maximum SHALL produce exactly one chunk. A page that is empty or only whitespace SHALL produce no chunk.

#### Scenario: Short page
- **WHEN** a page has 500 characters of text and the maximum is 800
- **THEN** one chunk with the whole cleaned text is produced

#### Scenario: Long page
- **WHEN** a page has 3000 characters of text and the maximum is 800
- **THEN** every chunk has at most 800 characters and together they cover the whole cleaned text

#### Scenario: Blank page
- **WHEN** a page contains only whitespace
- **THEN** no chunk is produced for it

### Requirement: Words are never split
The core SHALL start and end every chunk at a word boundary. A single word longer than the maximum size SHALL be emitted whole in its own chunk instead of being cut.

#### Scenario: Boundary falls inside a word
- **WHEN** the 800th character of a page lies in the middle of a word
- **THEN** that word goes entirely to the next chunk or entirely stays in the current one

#### Scenario: Oversized token
- **WHEN** a page contains a 1200-character run without spaces and the maximum is 800
- **THEN** that run appears unbroken in one chunk

### Requirement: Chunks never span pages
The core SHALL create chunks for each page independently, and every chunk SHALL record exactly one page number.

#### Scenario: Two short pages
- **WHEN** a document has two pages of 100 characters each
- **THEN** two chunks are produced, one for page 0 and one for page 1, and none contains text from both

### Requirement: Splitting prefers natural boundaries
When a page must be split, the core SHALL prefer to end a chunk at a paragraph boundary, then at a sentence boundary, and only then at a word boundary, while still honouring the maximum size.

#### Scenario: Paragraph boundary available
- **WHEN** a page has three paragraphs of 300 characters each and the maximum is 800
- **THEN** the first chunk contains the first two paragraphs and no chunk ends in the middle of a sentence

#### Scenario: Long paragraph
- **WHEN** one paragraph is longer than the maximum and contains several sentences
- **THEN** chunks end at sentence boundaries wherever a sentence fits within the maximum

### Requirement: Consecutive chunks overlap by a configurable amount
Within a page, each chunk after the first SHALL begin with text repeated from the end of the previous chunk. The repeated text SHALL NOT exceed a configurable fraction of the maximum size, with a default of 12 %, and SHALL start at a word boundary. An overlap of 0 SHALL produce no repeated text. An overlap fraction below 0 or at or above 0.5 SHALL be rejected as invalid configuration.

#### Scenario: Default overlap
- **WHEN** a long page is split with the maximum 800 and the default overlap
- **THEN** the repeated text between consecutive chunks is at most 96 characters

#### Scenario: No overlap
- **WHEN** overlap is configured as 0
- **THEN** concatenating the chunks of a page, ignoring the whitespace between them, reproduces the cleaned page text

#### Scenario: Invalid overlap
- **WHEN** overlap is configured as 0.6
- **THEN** configuration is rejected with an error

### Requirement: Chunks carry provenance
Every chunk SHALL record its page number (0-based), its position in the sequence of chunks of the document (0-based, consecutive), and the start and end character offsets of its text within the cleaned text of its page, so that the text of the page can be recovered from the offsets.

#### Scenario: Offsets locate the text
- **WHEN** a chunk is produced for a page
- **THEN** the page's cleaned text between the chunk's start and end offsets equals the chunk's text

#### Scenario: Sequence numbers
- **WHEN** a document of two pages yields five chunks
- **THEN** their sequence numbers are 0, 1, 2, 3, 4 in page order

### Requirement: Chunking is deterministic
Given the same text and configuration, the core SHALL always produce identical chunks in the same order.

#### Scenario: Repeated run
- **WHEN** the same document is chunked twice with the same configuration
- **THEN** both results are equal
