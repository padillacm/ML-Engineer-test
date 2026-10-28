# document-ingestion Specification

## Purpose
Defines how an uploaded PDF becomes stored, retrievable pieces of text that keep track of the document and page they came from, without ever overwriting another document.

## Requirements

### Requirement: Every page of a document is indexed
Ingesting a PDF SHALL recognise every page, split each page into chunks with the text-chunking rules, embed every chunk and store it, and SHALL report the number of chunks stored.

#### Scenario: Two-page document
- **WHEN** a two-page scanned PDF with text on both pages is ingested
- **THEN** chunks from page 0 and page 1 are stored and the reported count equals the number of chunks stored

#### Scenario: Document with no text
- **WHEN** a PDF whose pages contain no recognisable text is ingested
- **THEN** zero chunks are stored and zero is reported

### Requirement: Documents are identified by their content
The document identifier SHALL be derived from the SHA-256 hash of the uploaded file's bytes, so that different files never share an identifier and never replace each other's chunks. Each stored chunk SHALL record its document identifier, page, sequence, the content hash and the full text of its page.

#### Scenario: Two different documents
- **WHEN** document A and then document B are ingested
- **THEN** chunks of both A and B are stored and retrievable

### Requirement: The same file is never stored twice
When a file whose content hash is already stored is ingested again, the service SHALL NOT recognise, embed or store it again; it SHALL report that the document was already indexed together with the number of chunks stored for it.

#### Scenario: Same file twice
- **WHEN** the same PDF is ingested twice
- **THEN** the store holds one copy of its chunks, the second ingest reports "already indexed" with that chunk count, and no recognition or embedding runs for it

#### Scenario: Same content, different file name
- **WHEN** a PDF is ingested and then the identical bytes are uploaded under another file name
- **THEN** it is treated as the same document

#### Scenario: One byte different
- **WHEN** a PDF and a copy differing in one byte are ingested
- **THEN** both are stored as separate documents

### Requirement: Page text keeps the document's paragraphs
The page text that is chunked and stored SHALL separate the paragraphs recognised on the page with a blank line and the lines within a paragraph with a line break, so that chunk boundaries can follow paragraphs.

#### Scenario: Paragraph boundary used
- **WHEN** a page has two recognised paragraphs that together exceed the chunk size
- **THEN** a chunk boundary falls between the two paragraphs

### Requirement: Failures never lose or fake an indexed document
If recognition, embedding or storing fails, ingesting SHALL fail with the corresponding core error. A failed ingest SHALL NOT remove chunks written by another, completed ingest of the same file. A document whose chunks were only partly stored SHALL NOT be reported as already indexed, and the next ingest of the same file SHALL complete it.

#### Scenario: Embedding fails mid-document
- **WHEN** the embedder fails while a document is being ingested
- **THEN** the ingest fails and no chunk of that document is stored

#### Scenario: Concurrent ingests of the same file, one fails
- **WHEN** two ingests of the same file run at the same time, the first completes and the second fails while writing
- **THEN** the first ingest's chunks are all still stored and searchable

#### Scenario: Interrupted ingest is repaired
- **WHEN** an ingest stopped after writing only some of a document's chunks and the same file is ingested again
- **THEN** the second ingest is not short-circuited as already indexed and afterwards all chunks of the document are stored once
