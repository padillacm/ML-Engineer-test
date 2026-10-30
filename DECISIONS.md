# Decisions

The notable engineering decisions, why they were taken and where the evidence is. Each block of work was an OpenSpec change whose `design.md` (in `openspec/changes/archive/`) records its decisions in full, numbered D1, D2...; the measurements are in the [benchmark record](docs/BENCHMARKS.md), summarised in [Benchmarks](docs/benchmark-summary.md). The design as a whole is in [Design](DESIGN.md); terms and metrics are explained in the [glossary](docs/benchmark-summary.md#glossary).

## How the work was run

- **Spec first, one block at a time.** Nine OpenSpec changes, in order: domain foundation, OCR engine, name extraction, RAG pipeline, HTTP API, deployment, audit viewer, defensive hardening, GPU inference. Each was proposed, designed, implemented with tests, measured, reviewed and archived before the next; its specs are merged into `openspec/specs/`. Why: every behaviour has a written requirement and scenario a reviewer can check against the code.
- **Defaults are measured, not guessed.** OCR settings, the NER model and its rules, the match threshold, the embedder, chunk size, the refusal threshold, the concurrency rule and the language model were each chosen on a seeded synthetic corpus of 600 scanned PDFs in five languages, comparing candidates on the same documents with 95 % intervals. The OCR engine, the vector store and the concurrency model were judgement calls (only their settings were measured), and so are a few values the record marks as such (for example `READINESS_TIMEOUT_S`).
- **Measurement scripts stay out of the repository.** They are research code with machine-specific paths; the repository keeps their results, each table naming the result file it came from, and the method needed to repeat them.
- **Known limits are written down.** What cannot be fixed with this stack is in [Known limitations](docs/KNOWN_LIMITATIONS.md), each with its evidence, impact and what would remove it.

## API contract

`app/models/schemas.py` is unchanged. What was added, all optional or new, so every documented request and response still works:

- **`POST /api/ingest` response:** `status` (already in the contract) takes one of two values, `indexed` or `already_indexed` (the same bytes were stored before), and an optional `document_id` is added, the id derived from the file's content. Both come from a subclass of the contract's response in `app/api/rag.py`, so the contract file itself is untouched. Why: idempotent ingestion has to tell the client which case it hit, and the id is what the answers' passages cite.
- **`POST /api/extract` form:** `names` defaults to `[]`; an optional `language` (`en`, `es`, `pt`, `fr`, `de`) skips language detection. Why: a caller that knows the language gets better OCR (the single-language model keeps diacritics).
- **`GET /health/ready`:** readiness separate from liveness. Why: an orchestrator must not restart a live process because Qdrant is down.
- **Errors:** every error body is `{"detail", "code", "request_id"}` with a stable code per failure; 503 and 504 carry `Retry-After`; every response carries `X-Request-ID`. Why: clients branch on codes, operators correlate by id, and nothing echoes the input.
- **Extras:** `/audit` (the audit viewer); `/docs` and `/openapi.json` only with `EXPOSE_DOCS=true`.

## Architecture

- **Ports as `typing.Protocol`, one composition root.** Adapters and fakes satisfy the ports structurally; only `app/bootstrap.py` imports adapters; `tests/architecture` fails on a forbidden import. Why: the use cases are tested with in-memory fakes in milliseconds, and the boundary cannot erode.
- **Pure domain.** Geometry, span rules, matching, chunking, prompt building and the answer guardrails are plain functions with no vendor library (rapidfuzz aside). Why: the logic the assignment grades is unit-tested without models.
- **One error hierarchy with fixed messages.** Adapters wrap vendor exceptions; messages never repeat document text, keys or the request. Why: errors and logs can never leak personal data.
- **Contract tests shared by fakes and adapters.** The same test suite runs against the fake and the real vector store, LLM client and OCR engine. Why: a fake that drifts from the real adapter is caught.

## Extraction

- **PDFium, not PyMuPDF.** PyMuPDF is AGPL-3.0: a service that uses it over a network must publish its source to its users. The OCR benchmark had rendered with PyMuPDF; re-measured with PDFium, name recall did not change (-0.1 points, interval -1.1 to +0.7). PyMuPDF stays a test-only tool to build fixtures.
- **Tesseract CLI per page, through stdin and stdout.** No temp files, one single-threaded process per page in a small pool, killed with its process group on timeout or cancellation. Why: parallel across pages and nothing to leak. 54 pages took 7.9 s as 16 single-threaded processes, against 15.2 s as 4 processes of 4 threads and 43.7 s as one multi-threaded process (the service runs 4 workers per document and several documents at once).
- **200 DPI with illumination flattening, `psm 3`** (Tesseract's fully automatic page segmentation). Best character error rate (CER) and name recall in the grid; flattening is what makes hard scans readable: the share of printed names still found in the OCR text rises from 42.7 % to 90.3 %. The flattening's blur is re-implemented bit-exact in numpy so that OpenCV, a large binary dependency used for this one function, and its native libraries stay out of the slim image, while the benchmark's numbers still apply exactly.
- **Boxes converted with the real raster size, never an assumed DPI**, in the page's displayed frame (after `/Rotate`, relative to the crop box), clamped to the page.
- **Language detected once per document** from the first pages, then every page read with that model; the five-language model only when detection is inconclusive. Why: the combined model drops diacritics.
- **`Davlan/xlm-roberta-base-ner-hrl` plus a rule chain on word ranges.** F1 92.0 end to end on OCR text; the rules (title stripping, "Last, First" joins, initials, dropping single words unless titled) take it from 75.4 with the base rule set. ALL-CAPS words are title-cased before NER and reported as printed.
- **Names whose every word has OCR confidence below 20 are dropped** (signatures and noise read as letters): +0.2 precision, no recall lost.
- **Fuzzy matching with `rapidfuzz.token_sort_ratio` >= 0.90** on accent-stripped, casefolded text, once per distinct name. Why: it compares whole names in any word order; partial and set ratios gave 66-115 false matches on generated decoys, this one none. 0.88 and 0.90 both gave zero false matches; 0.90 keeps a margin of 0.02 above the closest generated wrong-person name (0.878) for 0.7 points of recall (87.5 % against 88.2 %), and 0.85 would add 1.7 points of recall but 7 false matches in 400 near-miss decoys. rapidfuzz rather than the suggested `thefuzz`: thefuzz now calls rapidfuzz itself, so this is the same scorer without the wrapper. Accepted: short variants of one name still match ("Jon" and "John" score 0.947).

## Retrieval and answering

- **Document id from the content** (`uuid5` of the SHA-256), point ids from `document_id:sequence`. Why: the same file is never stored twice, and two files never collide.
- **Idempotent, repairable upsert.** Chunks are overwritten by id, stale ones deleted only after every write succeeded; a document counts as stored only when complete. Why: an interrupted ingest is completed by the next one, never half-visible.
- **Qdrant as a separate server, not embedded Chroma** (a judgement call, not benchmarked). The assignment counts an isolated vector database as an extra; a server keeps the vectors when the service restarts and lets several replicas share them. Cost: one more container, and readiness has to report it.
- **One Qdrant collection per embedding identity** (model, revision, dimension, prefixes, metric), recorded in its metadata. Why: vectors of two models can never be mixed or compared.
- **e5-base with 800-character chunks, 12 % overlap, retrieving chunks and sending their parent pages.** 81.4 % of the answers reach the context against 72.2 % with whole-page chunks, with a smaller context; the page gives the model the surrounding sentences. e5 is used through `transformers` with mean pooling of the token vectors, the model card's recipe, rather than through `sentence-transformers`, a second wrapper over the same libraries. The cross-lingual rival (`snowflake-arctic-embed-m-v2.0`) runs code shipped with the model, so it was not adopted.
- **A relevance threshold of 0.808 before any model call.** It keeps 95 % of answerable questions on the corpus and refuses about half of the unanswerable ones without spending a model call; the model's own refusal is the second line and the support check the third. The model refuses every off-topic or absent-entity question it is sent, but only 14-18 % of the cross-document ones, so the threshold and the support check carry most of the grounding; together the checks refuse 0.47 % of correct answers. Trade-off: it is corpus-calibrated (Design section 6).
- **Sources are the stored texts of the chunks the model saw**, never text the model wrote.
- **Gemma 4 e4b on Ollama, thinking off.** The rule, written before the run: among the local models within the 95 % interval of the best one, the one with the lowest p95 latency. e4b scored 72.3 % against 72.8 % for the best (not distinguishable with 600 questions, about ±4 points), at 1.8 s p95 and with the least GPU memory. The 72.3 % is end to end, with refusals counted as wrong; when retrieval had put the answer in the passages the model got 95.5 % right, so most of the loss is retrieval. Thinking on cost 10-14 points and 7 times the latency. The hosted reference scored 69.5 %, so a hosted model is a deployment choice, not a quality upgrade.

## Robustness and operations

- **Async handlers, blocking work on dedicated thread limiters, bounded slots.** Jobs and questions have separate slots and threads; a request waits at most `BUSY_WAIT_S` for one, then gets 503 with `Retry-After`. Why: overload is shed early and visibly instead of piling up threads.
- **Concurrency derived from the CPUs.** Unset sizes follow a measured rule over the CPUs the container may use (affinity and cgroup quota): on 16 CPUs 12 jobs, 8 NER pipelines, 4 embedder slots, 4 OCR workers. Why: the fixed defaults oversubscribed small containers (-12 % to -26 %).
- **Fair pools.** Model slots are handed out in arrival order with a slot reserved for questions. Why: a question waited over two minutes behind passage batches before.
- **Cancellation on disconnect** for extraction and ingestion (a question runs to its end: it is short and holds no job slot). Why: an abandoned 50-page job would otherwise hold a slot for minutes.
- **Retries only where they help.** Connection errors, timeouts, 429 and 5xx are retried with full-jitter backoff inside an overall deadline; other errors fail at once.
- **Images.** Models baked in at pinned revisions with every file's sha256 checked, offline at run time; non-root, `tini` as PID 1, a read-only root, all capabilities dropped, PID and memory limits. Keys reach the containers as secret files, not variables.
- **The injection detector is optional at build time.** It is a gated model (Llama 4 Community License): with `HF_TOKEN` it is baked in, without one the stack still builds and the instruction patterns run alone.
- **Models on the GPU when present.** `DEVICE=auto`; the embedder and the detector in fp16, NER kept in fp32 (fp16 moved name boundaries on 6 of 1,030 pages). Why: on the CPU the detector cost 59 % of ingestion throughput; on the GPU 0-5 %.

## Security without authentication

Authentication and TLS are out of scope by decision (a reverse proxy's job); the service binds to `127.0.0.1`. Within that scope: per-address rate limits and job caps before the body is read, bounded PDF rendering, a text cap per page, injection screening of pages and questions, delimited passages, a support check and an output filter on answers, API docs off, protective headers. Details and measured effect in the [Security audit](docs/SECURITY.md).

## Considered and not adopted

- **Keyword search (BM25).** Measured alone, it beat the embedders on this corpus (67.6 against 64.4 % of answers in the top 5 chunks), because the generated questions quote exact names and terms. Real, paraphrased questions are expected to favour embeddings; a hybrid of the two was not tested and is the first experiment to run on real documents.
- **`snowflake-arctic-embed-m-v2.0`:** better cross-lingual, worse same-language, runs code shipped with the model.
- **spaCy** (`xx_ent_wiki_sm`, F1 65.9 end to end), **GLiNER and larger NER models:** behind on public data, or costlier.
- **`WRatio` and partial or set ratios:** 66-115 false matches on decoys.
- **Whole-page or longer chunks, Otsu binarisation, 300-DPI rendering, per-language refusal thresholds:** worse or noisy in the measurements.
- **A job queue:** the bounded slots are enough at this scale; Design section 3 says when it is not.
- **An LLM judge for answer quality:** a second model to trust.

## Use of AI tools

The work was done with Claude Code (Anthropic's coding agent, Claude Opus models) as a pair: it drafted the OpenSpec proposals and designs, wrote code, tests and documentation, ran the measurements and the reviews, under my direction. I set the scope and the order of the blocks, took the decisions the agent raised as options (models and their licences, behaviour on flagged content, defaults, what to defer), reviewed each block, and approved every commit; each commit carries a `Co-Authored-By` trailer naming the model. The agent's project instructions are versioned in `.claude/` (the OpenSpec commands and the commit skill).
