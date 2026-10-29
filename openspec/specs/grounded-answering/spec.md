# grounded-answering Specification

## Purpose
Defines how a question is answered only from the content of ingested documents, how the service refuses when nothing relevant was found, and what it returns as sources so the answer can be checked.

## Requirements

### Requirement: Questions are answered from retrieved content only
The service SHALL embed the question, retrieve the most similar stored chunks and ask the language model to answer using only the provided context, instructing it to say that the documents do not contain the answer when that is the case.

#### Scenario: Answerable question
- **WHEN** a document stating "The contract was signed by Maria Gonzalez on 3 March 2024" has been ingested and the question is "Who signed the contract?"
- **THEN** the language model receives that text in its context and the service returns its answer

#### Scenario: Prompt forbids outside knowledge
- **WHEN** any question is sent to the language model
- **THEN** the instructions tell it to answer only from the context and to say the answer is not in the documents otherwise

### Requirement: Context uses the parent pages of the best chunks
The context given to the language model SHALL be the full text of the pages that contain the best-ranked distinct chunks, up to a configurable number of chunks (default 3) and a configurable character budget, each page labelled with its document and page number.

#### Scenario: Answer outside the matching chunk
- **WHEN** the best chunk is the first half of a page and the answer is in the second half of the same page
- **THEN** the context contains the whole page, including the answer

#### Scenario: Budget respected
- **WHEN** the selected pages exceed the character budget
- **THEN** pages are added in rank order until the budget is reached and the context, labels and separators included, never exceeds it

### Requirement: Low relevance is refused without calling the model
When the best retrieved chunk scores below a configurable similarity threshold (default 0.808), or nothing is stored, the service SHALL return a fixed refusal answer with no sources and SHALL NOT call the language model.

#### Scenario: Off-topic question
- **WHEN** the best chunk's similarity is below the threshold
- **THEN** the answer is the fixed refusal, the sources are empty and the language model is not called

#### Scenario: Empty store
- **WHEN** no document has been ingested
- **THEN** the answer is the fixed refusal and the sources are empty

### Requirement: Sources are the retrieved text the model saw
The sources returned with an answer SHALL be the texts of the retrieved chunks that passed the threshold and whose text was included in the context given to the language model, exactly as stored, in rank order. The service SHALL NOT return text it did not retrieve or did not show to the model. When the model reports that the documents do not contain the answer, the service SHALL return the fixed refusal with no sources.

#### Scenario: Sources match retrieval
- **WHEN** a question is answered
- **THEN** every returned source is the exact text of a chunk returned by the search for that question

#### Scenario: Retrieved but not shown
- **WHEN** a chunk passes the threshold but its page was left out of the context because of the chunk or character limit
- **THEN** that chunk is not returned as a source

#### Scenario: Model refuses
- **WHEN** the model answers that the documents do not contain the information
- **THEN** the service returns the fixed refusal and no sources

### Requirement: Model failures are reported, not hidden
When the language model is unavailable or times out after its retries, answering SHALL fail with the corresponding dependency error rather than returning a made-up answer.

#### Scenario: Model down
- **WHEN** the language model cannot be reached
- **THEN** the request fails with a dependency-unavailable error

### Requirement: Blank questions are invalid input
A question that is empty or only whitespace SHALL be rejected with an invalid-input error coded `invalid_question` before anything is embedded, searched or sent to the model.

#### Scenario: Whitespace question
- **WHEN** the question is a tab and two spaces
- **THEN** an invalid-input error with code `invalid_question` is raised and neither the embedder nor the store is called
