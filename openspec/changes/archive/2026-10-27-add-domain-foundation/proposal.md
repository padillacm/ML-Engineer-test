# Proposal

## Why

The repository is a skeleton: every service raises `NotImplementedError` and nothing is tested. Before any adapter (OCR, NER, embeddings, vector store, LLM) is written, the project needs a stable core that does not depend on any of them. The benchmarks run on a 600-PDF synthetic corpus (en/es/pt/fr/de, real Tesseract output) fixed the numeric defaults for the pure-logic pieces: coordinate conversion, accent/case-insensitive fuzzy matching at 0.90, and 800-character chunks with 12 % overlap. Building this core first keeps the later blocks small and lets the swap-ability promise (change model or vector DB without touching business logic) be enforced by a test instead of by convention.

## What Changes

- Introduce the package layout `app/domain` (pure logic), `app/application` (outbound port definitions; use cases arrive in later blocks) and `app/adapters` (empty marker; real adapters arrive in later blocks). `app/models/schemas.py` stays the HTTP contract and is not imported by the new packages.
- Add domain value objects (frozen dataclasses): page-space box, OCR word, page text with character offsets, chunk, name pair, fuzzy match.
- Add page geometry: pixel-to-point conversion from the real raster size and page rectangle (never an assumed DPI), union of word boxes, clamping to the page, IoU, and localisation of a character span on a page into one box.
- Add name matching: NFKD accent-strip + casefold normalisation and `rapidfuzz` `token_sort_ratio` scoring on `"first last"`, score in 0..1, configurable threshold (default 0.90), one best match per extracted name.
- Add text chunking: OCR text cleanup (de-hyphenation, line joining), hierarchical split page -> paragraph -> sentence -> word that never splits a word nor crosses a page, configurable size (default 800 characters) and overlap (default 12 %), with offsets metadata.
- Add the five outbound ports as `typing.Protocol` classes (`OcrEngine`, `NameRecognizer`, `Embedder`, `VectorStore`, `LLMClient`) and the domain error hierarchy that later blocks map to HTTP.
- Add the tooling needed to run tests with `uv`: `pytest` configuration, a `tests/` tree, `rapidfuzz` as a runtime dependency, and removal of `thefuzz` and `python-Levenshtein` from `pyproject.toml`.
- Add an architecture test that fails if `app/domain` or `app/application` import a vendor library (FastAPI, Pydantic, PyMuPDF, pytesseract, spaCy, transformers, sentence-transformers, Qdrant, httpx, etc.).
- No HTTP endpoint changes and no behaviour visible to API clients in this block. The existing `app/services/*.py` stubs stay untouched; each one is removed in the block that replaces it.

## Capabilities

### New Capabilities
- `page-geometry`: coordinate conversion from raster pixels to PDF points, box union/clamping/IoU, and mapping a text span back to a page box.
- `name-matching`: accent- and case-insensitive normalisation of person names and thresholded fuzzy matching of extracted names against requested name pairs.
- `text-chunking`: cleanup of OCR text and splitting of pages into overlapping, word-safe chunks with provenance metadata.
- `architecture-boundaries`: the outbound ports, the domain error hierarchy and the enforced rule that core packages import no vendor library.

### Modified Capabilities
<!-- None: openspec/specs/ is empty. -->

## Impact

- New code under `app/domain/`, `app/application/`, `app/adapters/` and `tests/`; no existing module is modified except `pyproject.toml`.
- Dependencies: add `rapidfuzz`; drop `thefuzz` and `python-Levenshtein` (4 -> 1 fuzzy-related packages; justified by dependency count, not speed, to be recorded in `DECISIONS.md`). Add a `uv.lock`.
- `rapidfuzz` is the only third-party library allowed inside `app/domain`; all other third-party imports are forbidden there and in `app/application`.
- Out of scope for this block: any OCR, NER, embedding, vector-store or LLM adapter; span post-processing rules for NER output (title stripping, "Last, First" merge; block 3); settings loading and the FastAPI composition root (block 5); Docker (block 6).
