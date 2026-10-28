# Design

## Context

The repository holds the take-home skeleton: `app/main.py`, two empty routers, `app/models/schemas.py` (fixed HTTP contract) and seven `app/services/*.py` stubs that raise `NotImplementedError`. `pyproject.toml` lists `thefuzz` and `python-Levenshtein`, `chromadb` and `spacy` among the baseline dependencies. There are no tests. Motivation is in `proposal.md`; the observable behaviour is in the four specs of this change.

Numeric defaults come from measurements on a 600-PDF synthetic corpus run through real Tesseract output (benchmark notes live outside the repo and are summarised in `DECISIONS.md` at the end): `token_sort_ratio >= 0.90` on `"first last"` gave 100 % precision and 87.5 % end-to-end recall with the chosen NER model; 800-character chunks with 12 % overlap were the best setting inside the tested grid; pixel boxes must be converted with the real raster size because truth boxes are in points.

## Goals / Non-Goals

**Goals:**
- A dependency-light core that can be unit-tested with `uv run pytest` without models, Docker or network.
- Ports defined once, with test fakes, so adapters in later blocks only have to satisfy the same contract tests.
- Make the "core imports no vendor" rule executable.

**Non-Goals:**
- Real OCR, NER, embedding, vector-store or LLM adapters.
- NER span post-processing (title stripping, "Last, First" merge, single-token filter, ALL-CAPS title-casing): these depend on the NER adapter output and belong to block 3.
- Settings loading, logging and the FastAPI composition root (block 5).
- Removing the `app/services/*.py` stubs: each is deleted in the block that supersedes it, so the tree stays importable.

## Decisions

### D1. Layout: `domain` (pure), `application` (ports and, later, use cases), `adapters` (later)

```
app/
  domain/        geometry.py  text.py  matching.py  chunking.py  errors.py  types.py
  application/   ports.py     (use cases arrive in blocks 2-4)
  adapters/      __init__.py  (empty; real adapters arrive in blocks 2-4)
  api/ models/   unchanged HTTP edge
tests/
  domain/  application/  architecture/  fakes/
```

Ports live in `application`, not `domain`: they describe what the use cases need from the outside, and the domain stays free of any notion of "an OCR engine". Domain types are frozen `dataclass(slots=True)`; Pydantic is reserved for the HTTP edge and settings, so the core has no validation-framework dependency. Mapping from domain types to `schemas.py` models happens in the routers in block 5.
*Alternative: put ports in `domain` (classic DDD).* Rejected: the domain has no behaviour that needs them and it would force `domain` to name infrastructure concepts.
*Alternative: reuse the Pydantic models in the core.* Rejected: the HTTP contract is fixed and unrelated to what the core needs (offsets, per-page raster size), and it would make the architecture test impossible.

### D2. Ports are `typing.Protocol`, not abstract base classes

Adapters and test fakes satisfy a port structurally, with no inheritance and no import of the application layer from an adapter. All five Protocols are marked `runtime_checkable` so tests can assert with `isinstance` that a fake or adapter provides every method (this checks method presence, not signatures). Contract tests (the same test body run against the fake now and the real adapter later) are written in this block for the vector store and the embedder identity rule; the real adapters run them in blocks 2-4.
*Alternative: ABCs.* Rejected: forces inheritance and gives errors at instantiation rather than type-check time, with no practical benefit here.

Port shapes (fixed now, so later blocks do not churn them):
- `OcrEngine.recognize(pdf: bytes, language: str | None = None) -> list[OcrPage]`, every page in order; an `OcrPage` carries the page size in points and its words as `PageWord`s (text plus a point-space `Box`). The adapter renders the PDF and converts pixels to points with `PageRaster` (D4) using the real pixmap size and `page.rect`, so callers never see raster pixels; `language=None` lets the adapter decide. (Revised during implementation: the first draft had the caller convert pixels.)
- `NameRecognizer.find_persons(text: str) -> list[TextSpan]`, character offsets into the given text.
- `Embedder`: `identity`, `embed_passages(texts)`, `embed_query(text)`; prefixes live inside the adapter.
- `VectorStore`: `ensure_collection(identity)`, `upsert(document_id, chunks, vectors)`, `search(vector, top_k)`, `delete_document(document_id)`, `count()`. Scores are cosine in 0..1 (negative cosine is clamped to 0); point ids are derived by the store from `document_id` and chunk sequence so re-ingest of a document replaces its chunks and two documents never collide. A vector whose length differs from the bound dimension raises `IncompatibleEmbedding`. Search results are `SearchHit(document_id, page, sequence, text, score)`.
- `LLMClient.complete(messages: list[ChatMessage]) -> str`, with retries and timeouts inside the adapter and `LLMUnavailable` / `LLMTimeout` as the only failures.

### D3. Name matching: `rapidfuzz.fuzz.token_sort_ratio` over own NFKD normalisation

`normalize(text)`: NFKD, drop combining marks, `casefold()` (so ß -> ss), collapse whitespace. The score is `token_sort_ratio(normalize(extracted), normalize(f"{first} {last}")) / 100`. Full-string scorers were chosen over `WRatio`/`token_set_ratio`/`partial_ratio` because on the name-list benchmark those matched "Heinz Müller" inside "Karl-Heinz Müller" and produced 75-115 false positives, while `token_sort_ratio` produced none at 0.90; plain `ratio` was order-sensitive and lost "Last, First" (84.6 % vs 94.0 % oracle recall).

`rapidfuzz` is chosen over `thefuzz` + `python-Levenshtein` to reduce the maintained dependencies from two (plus a C extension that has to be built or pinned) to one with wheels; measured behaviour was not a reason (no rounding or ranking difference was found) and `DECISIONS.md` will say so. It is the only third-party import allowed in `domain`, because it is a pure computation with no I/O or global state.

One match per extracted name: the best scoring pair at or above the threshold, ties resolved by request order. This keeps the response size bounded by the number of extracted names and makes the result deterministic.

### D4. Geometry: explicit raster descriptor, never a DPI

`PageRaster(width_px, height_px, page_width_pt, page_height_pt)` carries the measured pixmap size and `page.rect`; `to_points(box)` scales each axis separately. This removes the classic bug of assuming 72/150/200 DPI and handles rounding of the rendered size. Rotated pages are the adapter's job: it passes the rotated `page.rect` and an image rendered in the same orientation, so this module only ever sees axis-aligned boxes.

`PageText` is built from `PageWord`s already in points: words joined by single spaces, with a parallel list of `(start, end)` character offsets. `locate(span)` returns the union of all words whose character range intersects the span; an empty span or one touching only a separator yields `None`. Blank words are dropped when building. Union uses the whole word box (no proportional trimming), matching the benchmark where trimming gave no gain. `locate` does not clamp because `PageText` does not know the page size; callers apply `clamp` with the page rectangle (`OcrPage` carries it). `clamp` rejects a box that only touches the page edge, since clipping would leave it empty.

### D5. Chunking: hierarchical greedy packing with word-aligned overlap

`clean_page_text` implements the specified cleanup. Then, for each page: split into paragraphs on blank lines, each paragraph into sentences (regex on `.?!` followed by whitespace and not preceded by a short abbreviation of one or two letters), each sentence into words. Units are packed greedily up to `max_chars`; a unit larger than the remaining room descends one level. Overlap is the largest proper suffix of the previous chunk that starts at a word boundary and is at most `floor(overlap_ratio * max_chars)` characters. It is chosen before packing the next chunk, which then fills the remaining room; if the first new word does not fit beside it, the overlap shrinks, down to none. (Choosing it after greedy packing would almost never leave room.) A chunk ends at the furthest paragraph break that fits, else the furthest sentence end, else the last word that fits, so a short heading can become its own chunk. Offsets are computed on the cleaned page text so `page_text[start:end] == chunk.text`. Sequence numbers are assigned across the whole document.

The chunk record deliberately has no token count: counting tokens needs the embedder's tokenizer, which the core must not import. The adapter that needs it can compute it from `text`.

The specs set the defaults at 800 characters and 12 %. The benchmark showed recall still improving up to the whole page; the answer chosen in `DECISIONS.md` is to retrieve at 800 characters and hand the parent page to the LLM (block 4), which needs only `page` and offsets that this block already records.

### D6. Errors: one small hierarchy with stable codes

`ServiceError(code, message, category)` with subclasses `InvalidInput`, `ProcessingFailed`, `DependencyUnavailable`, `DependencyTimeout`, `IncompatibleEmbedding`, plus `GeometryError` and configuration errors under `InvalidInput`/`ValueError` as appropriate. Codes are lowercase snake strings (`not_a_pdf`, `too_large`, `llm_unavailable`) listed in one module so the HTTP handler in block 5 can map category to status (400/422, 413, 500, 503, 504, 409) from a table. Adapters wrap vendor exceptions into these; messages are built from fixed strings plus non-secret details.

### D7. Architecture test by AST scan

`tests/architecture/test_imports.py` walks `app/domain` and `app/application` with `ast`, collects top-level module names from `import` / `from ... import`, and compares them against `sys.stdlib_module_names` plus an allowlist (`rapidfuzz`, the project's own `app.domain` / `app.application`). Anything else, including `app.adapters` and `app.models`, fails the test with file and line. An allowlist (rather than a deny list of vendors) means a newly added dependency cannot slip in unnoticed. Relative imports are resolved to absolute names. A second test checks that `app/domain` does not import `app.application`.

### D8. Tooling

`uv` project: the `dev` extra holds `pytest`, `pytest-cov`, `ruff`, `mypy` and `pre-commit`; `[tool.pytest.ini_options]` sets `testpaths = ["tests"]` and `pythonpath = ["."]`. `thefuzz` and `python-Levenshtein` are removed; `rapidfuzz>=3` is added (the lockfile pins it). `uv.lock` is committed. `pyproject.toml` lists only what the code imports or runs today (`fastapi`, `uvicorn[standard]`, `pydantic`, `rapidfuzz`): the skeleton's unused baseline (`spacy`, `chromadb`, `sentence-transformers`, `pytesseract`, `pymupdf`, `pillow`, `httpx`, `pydantic-settings`, `python-multipart`) was dropped, which also cut the lock from 170 to 45 packages and removed the eight `chromadb` advisories reported by `uv audit`. Each later block adds the dependencies of its own adapter in the same change that introduces it (PyMuPDF and Tesseract bindings, transformers and torch CPU, the Qdrant client, an HTTP client, settings, multipart), so no block leaves the project half-migrated.

### D9. Test fakes live in `tests/fakes`, not in `app`

In-memory `FakeOcrEngine`, `FakeNameRecognizer`, `FakeEmbedder` (deterministic hashed bag-of-words vectors, so related texts score higher), `FakeVectorStore` (pure-Python cosine, enforces identity binding and replace-on-reingest) and `FakeLLMClient`. Keeping them out of `app` prevents production code from depending on test code, at the cost of later blocks importing them from `tests`.

### D10. Benchmark evidence is committed, not left in a scratchpad

Defaults and the capacity estimate in `DESIGN.md` must be traceable. `docs/BENCHMARKS.md` records for each measurement the data, versions, hardware, tables with intervals, caveats and reproduction commands; `benchmarks/` holds the small scripts and the seeded corpus generator (never data or weights). The documents/hour estimate is written as a formula over measured quantities, not as a bare number, and states that it must be re-measured on the target hardware.

## Commitments for later blocks

Decisions already taken with the user that this block does not implement but later blocks must honour:

- **Docker is the primary way to run the service.** The vector store is a Qdrant server in `docker-compose.yml` (block 6); tests use an in-memory instance. No embedded-store mode is added now; a one-line configuration switch could be added later if the service has to run without Docker.
- **Resilience on every external call:** timeouts and bounded retries with backoff on the LLM client and on the vector store, tested with fakes that fail (blocks 4-5). Failures surface as `DependencyUnavailable` / `DependencyTimeout`; nothing is swallowed (ruff rules in task 1.1 enforce no bare or silent `except`).
- **Resource management:** PDFs are opened from bytes and closed with context managers, pixmaps are released, no temporary files are written; a test verifies the document is closed when OCR raises (block 2).
- **Unique ids on ingest:** `document_id` is a UUID generated per ingest, so a second upload never overwrites another document; the content hash is stored as metadata for a later de-duplication feature (block 4).
- **Configuration:** one `Settings` object holds hosts, ports, model names, thresholds, DPI, chunking and upload limits, overridable by environment (block 5).

## Risks / Trade-offs

- [Token-sort scoring accepts near-identical names such as "Jon Smith" vs "John Smith" (about 0.95)] -> Documented in `DECISIONS.md`; threshold stays configurable; the benchmark's decoys were separated by a wider margin than real ones, so real-world precision may be lower than the measured 100 %.
- [Letters that NFKD does not decompose (ø, ł, đ, æ) are not folded to ASCII] -> Only affects names with these letters, in the requested list or in the document; add an explicit fold table if a test document shows it. Not covered by the benchmark, which used Western/Latin names.
- [Sentence splitting by regex mis-splits abbreviations such as "Dr. Smith"] -> Mitigated by the short-abbreviation rule; worst case a chunk ends one sentence early, which costs recall slightly but never correctness, since splitting falls back to word boundaries.
- [Chunk size tuned on template text with 1,750-character pages; real pages may prefer smaller chunks] -> Size and overlap are settings; the recalibration rule goes in `DESIGN.md`.
- [Protocol shapes fixed early may need small adjustments when the first real adapter lands] -> Contract tests make an adjustment visible in one place; changes are made through a follow-up change rather than silently.
- [Keeping `app/services` stubs next to the new packages briefly leaves two structures] -> Each stub is removed in the block that replaces it; the README layout section is updated at the end.

## Migration Plan

No data or API migration: nothing in this block is wired to an endpoint. Rollback is deleting the new packages and restoring `pyproject.toml`. `uv lock` is run once after the dependency edit; the lockfile is part of the change.

## Open Questions

- Whether `application/ports.py` should be split into one module per port once adapters arrive; deferred, it does not alter any spec.
