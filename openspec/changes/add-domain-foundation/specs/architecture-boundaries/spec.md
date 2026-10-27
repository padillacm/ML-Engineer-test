# Spec Delta

## Purpose

Defines the contracts through which the service core talks to external technologies and the rules that keep the core independent of them, so a model, engine or database can be replaced without changing business logic.

## ADDED Requirements

### Requirement: Each external capability has a contract
The core SHALL define one contract for each external capability it needs: recognising text and word positions from a page image (OCR), recognising person names in text, turning text into embedding vectors, storing and searching vectors, and completing a chat conversation with a language model. Each contract SHALL be expressed only in terms of core data types.

#### Scenario: Test double replaces a real technology
- **WHEN** a use case is exercised with an in-memory implementation of each contract
- **THEN** it runs without any external model, engine or service installed

#### Scenario: Contract uses core types
- **WHEN** a contract method receives or returns data
- **THEN** that data is built from core types or standard Python types, not from types of a third-party library

### Requirement: Embeddings carry an identity
An embedding provider SHALL expose an identity made of its model identifier, vector dimension, text-prefix scheme, whether vectors are normalised, and distance metric. Two providers with the same identity SHALL be interchangeable for stored vectors; any other pair SHALL NOT.

#### Scenario: Identity is comparable
- **WHEN** two providers report the same model, dimension, prefix scheme, normalisation and metric
- **THEN** their identities compare equal

#### Scenario: Different prefix scheme
- **WHEN** two providers differ only in their prefix scheme
- **THEN** their identities compare unequal

### Requirement: A vector collection is bound to one embedding identity
A vector store SHALL record the embedding identity when a collection is created and SHALL refuse to open that collection, or to add to it, with a different identity, raising an incompatible-embedding error. Search scores SHALL be cosine similarity from 0.0 to 1.0, higher meaning more similar.

#### Scenario: Reopening with the same identity
- **WHEN** a collection is opened with the identity it was created with
- **THEN** it opens and previously stored chunks remain searchable

#### Scenario: Reopening with another identity
- **WHEN** a collection created with a 384-dimension model is opened with a 768-dimension identity
- **THEN** the operation fails with an incompatible-embedding error and nothing is modified

#### Scenario: Re-ingesting the same document
- **WHEN** a document is added twice under the same document identifier
- **THEN** the store holds the chunks of the latest addition only

### Requirement: Failures use a closed set of categories
The core SHALL raise errors from one hierarchy in which each error has a stable machine-readable code, a human-readable message and one of these categories: invalid input, processing failure, dependency unavailable, dependency timeout, and incompatible embedding. Error messages SHALL NOT contain secrets such as API keys.

#### Scenario: Error carries a code
- **WHEN** an invalid-input error is raised for a file that is not a PDF
- **THEN** it exposes a stable code, a message and the category "invalid input"

#### Scenario: Wrapped vendor errors do not leak into the message
- **WHEN** a core error is raised from a vendor exception whose text contains an API key
- **THEN** the core error's text, code and message contain only what its raiser supplied, not the vendor text

#### Scenario: Errors survive process boundaries
- **WHEN** an error is copied or sent between processes
- **THEN** the copy has the same type, code, message and category

#### Scenario: Catching by category
- **WHEN** code catches the base error type
- **THEN** every error raised by the core is caught

### Requirement: The core imports no vendor library
Modules of the domain and application layers SHALL import only the Python standard library, each other (the domain imports nothing from the application layer), and the single string-similarity dependency allowed in the domain. They SHALL NOT import web frameworks, validation frameworks, PDF, OCR, NLP, machine-learning, vector-database or HTTP-client libraries, nor the adapter layer. An automated test SHALL fail the build when this rule is broken.

#### Scenario: A forbidden import is added
- **WHEN** a domain module imports a web framework
- **THEN** the architecture test fails and names the module and the import

#### Scenario: Clean core
- **WHEN** the architecture test runs on the unmodified core
- **THEN** it passes

### Requirement: The HTTP contract is not part of the core
The request and response models that define the HTTP API SHALL live only at the HTTP edge. Core modules SHALL NOT import them, and adding the core SHALL NOT change any field of those models.

#### Scenario: Contract unchanged
- **WHEN** the existing API schema module is compared before and after this change
- **THEN** it is byte-for-byte identical
