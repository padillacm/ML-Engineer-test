# Document Intelligence Service

A FastAPI service that reads scanned PDFs and returns every person's name with **where it sits on the page** (0-based page, box in PDF points), fuzzy-matches those names against a list, and answers questions about ingested documents with retrieval-augmented generation (RAG) that only answers from the documents.

> **For reviewers.** The two written deliverables are [Design](../DESIGN.md) and [Decisions](../DECISIONS.md). Everything else on this site is reference material they link to. A 15-minute path: [Getting started](getting-started.md), then [Architecture](architecture.md), [Design](../DESIGN.md) and [Decisions](../DECISIONS.md); [Evolving the solution](evolution.md) says what comes next.

## What is in the box

| Endpoint | What it does |
|---|---|
| `GET /health`, `GET /health/ready` | Liveness, and readiness of the vector store and the language model. |
| `POST /api/extract` | OCR, person names, boxes in PDF points, fuzzy matches. |
| `POST /api/ingest` | OCR, word-safe chunks, embeddings, stored in Qdrant under an id derived from the file's bytes. |
| `POST /api/ask` | Retrieval, a grounded answer and the exact stored chunks it was given, or a fixed refusal. |
| `GET /audit` | The audit viewer: draws the returned boxes over the PDF rendered by pdf.js. |

The stack: Tesseract 5.5 with PDFium rendering, the `Davlan/xlm-roberta-base-ner-hrl` NER model plus span rules, rapidfuzz, `intfloat/multilingual-e5-base` embeddings, Qdrant, any OpenAI-compatible model (Gemma 4 on a local Ollama by default with the `gpu` profile). Five languages: en, es, pt, fr, de.

## The assignment's checklist

Each requirement of [the assignment](assignment.md), where it is implemented and what proves it.

| Requirement | Where | Proof |
|---|---|---|
| OCR every page, including the first | `app/adapters/ocr/engine.py` | `tests/adapters/ocr/test_engine.py`, `tests/integration/test_ocr_samples.py` |
| Boxes in PDF points, 0-based pages | `app/domain/geometry.py` (scale from the real raster size, never an assumed DPI) | `tests/domain/test_geometry.py`, `tests/integration/test_ocr_positions.py`, browser tests of the viewer |
| Person names only | NER model + span rules in `app/domain/names.py` | `tests/domain/test_names.py`, `tests/integration/test_extract_names_e2e.py` |
| Case- and accent-insensitive full-string matching | `app/domain/matching.py` (`token_sort_ratio` >= 0.90) | `tests/domain/test_matching.py`, `tests/domain/test_matching_decoys.py` |
| Chunks never cut a word | `app/domain/chunking.py` | `tests/domain/test_chunking.py` |
| No hallucinated `sources` | `app/domain/prompting.py`: sources are stored chunk texts the model saw | `tests/domain/test_prompting.py`, `tests/application/test_ask_question.py` |
| Vector ids unique across documents | `uuid5(document_id:sequence)` in the Qdrant adapter | `tests/adapters/vector_store/test_qdrant_contract.py` |
| Reject non-PDFs by content, empty and oversized uploads | `app/domain/uploads.py`, `app/api/middleware.py` | `tests/domain/test_uploads.py`, `tests/api/test_body_limits.py` |
| No leaked handles or temp files | Tesseract via stdin/stdout, PDFium objects closed under a lock | `tests/api/test_resource_release.py` |
| Timeouts and retries on the LLM and the vector store | `app/adapters/llm/`, `app/adapters/vector_store/retry.py` | `tests/adapters/llm/test_client.py`, `tests/adapters/vector_store/test_retry.py` |
| Structured logs with context | `app/observability.py` (JSON, request id, allow-listed fields) | `tests/observability/` |
| Config from the environment | `app/settings.py` | `tests/settings/test_settings.py`, `tests/deploy/test_env_example.py` |
| SOLID, DI, thin routers | ports in `app/application/ports.py`, one composition root | `tests/architecture/test_imports.py` (fails on a forbidden import) |
| At least one end-to-end path | real OCR, models and Qdrant | `tests/integration/test_http_e2e.py`, `tests/integration/test_rag_e2e.py` |
| Bonus: Docker, isolated vector DB, audit viewer, multilingual, hardening | `Dockerfile`, `docker-compose.yml`, `app/web/` | `tests/deploy/`, `tests/e2e/`, `tests/adversarial/` |

## State of the code

| Check | Result |
|---|---|
| Host suite (`uv run pytest -q --cov`) | about 4,400 tests pass; coverage of `app/` 98.5 % (gate 90 %) |
| Container suite (Tesseract and the real models) | about 4,600 tests pass |
| ruff, mypy `--strict`, `uv audit` | clean |
| CI | `.github/workflows/ci.yml`: lint, types, the host suite, the dependency audit and the specs |

## How the work was done

Each block of work was planned as an [OpenSpec](https://github.com/Fission-AI/OpenSpec) change (proposal, design, spec deltas, tasks), implemented, measured and archived. The living specs are in `openspec/specs/`, the archived changes with their designs in `openspec/changes/archive/`. Defaults come from measurements recorded in the [benchmark record](BENCHMARKS.md); what could not be fixed is in [Known limitations](KNOWN_LIMITATIONS.md). The use of AI tools is disclosed in [Decisions](../DECISIONS.md).
