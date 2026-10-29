# Tasks

## 1. Baseline and shared scaffolding

- [x] 1.1 Archive `add-audit-viewer`; verify `openspec validate --all --strict` passes with 15 specs
- [x] 1.2 Shared scaffolding (D13): new settings with defaults and ranges (`EXPOSE_DOCS`, `RATE_LIMIT_*`, `MAX_JOBS_PER_CLIENT`, `OCR_RENDER_TIMEOUT_S`, `OCR_MAX_IMAGE_PIXELS`, `MAX_PAGE_CHARS`, `INJECTION_MODEL_PATH`, `INJECTION_THRESHOLD`, `INJECTION_POOL_SIZE`, `RAG_SUPPORT_MIN_SHARE`, `RAG_MAX_CHUNKS_PER_DOCUMENT`), the `detector_failed` core code with its category and status, `429: rate_limited` in the HTTP status table, `Chunk.flagged`, `IngestResult.chunks_flagged`, the `InjectionDetector` port, `NoDetector`, a fake detector in `tests/fakes/`, and the compose environment list; verify the settings tests (code defaults, agreed defaults, every setting listed) and `tests/deploy/test_compose_wiring.py` pass

## 2. HTTP edge

- [x] 2.1 `RateLimitMiddleware` (D1): token buckets per address and route class, per-client job counter, 429 through the HTTP error handler with `Retry-After`, bucket pruning, zero disables; unit tests with a fake clock (burst, refill, route classes, two clients, probes exempt, cap of running jobs, disabled limit) and a served-app test with real connections
- [x] 2.2 Documentation switch, protective headers and server bounds (D2): `EXPOSE_DOCS`, the three headers on `/api/*` and `/health*` only, `--limit-concurrency` and `--timeout-keep-alive` in the runtime command; tests for 404 by default and 200 when enabled, headers on success and error responses, the viewer unaffected, and the error-body and access-line checks for 429; README section on limits and headers

## 3. PDF and OCR bounds

- [x] 3.1 Progressive rendering with a time budget and the image pre-check (D3): `OCR_RENDER_TIMEOUT_S`, `OCR_MAX_IMAGE_PIXELS`; tests that a synthetic hostile page (thousands of large images, or millions of paths) fails `pdf_unreadable` within the budget plus a step while another document renders afterwards, that an over-budget image page fails `too_large` without rendering, and that every sample PDF and the rotation and crop-box fixtures render pixel-identical to the previous code
- [x] 3.2 Page text cap (D4): `MAX_PAGE_CHARS` enforced in the OCR engine, `too_large` naming the page before NER, chunking or embedding; tests with a page one character over the limit (recogniser and embedder not called) and the densest sample page accepted; record the sample's maximum page length in the benchmarks section

## 4. Injection detector

- [x] 4.1 `app/adapters/injection/` (D5): `PromptGuardConfig`, `PromptGuardDetector` (local model, windows with stride, max score, shared model with `FifoSlots`, tokenizer lock, batches, `detector_failed` on failure, `close`), composition root wiring and the start-up line; unit tests with a tiny randomly initialised classifier in `tests/adapters/injection/` (window scoring, batching, concurrency, failure mapping, closed detector) and `injection_model`-marked tests with the real model on the delta's Spanish injection and benign contract text
- [x] 4.2 `prompt-guard` image stage (D11): pinned revision and sha256 ARGs (resolved with `scripts/resolve_model_hashes.py` and the user's `HF_TOKEN`), build secret, NOTICE with the Llama 4 Community License attribution, `INJECTION_MODEL=0` opt-out, `INJECTION_MODEL_PATH` in the images, compose `build.secrets`; verify both builds, the hashes, that the token is in no layer (`docker history`) and no container environment, and that the opt-out image starts with detection off

## 5. Ingestion, store and answering guardrails

- [x] 5.1 Flagging at ingestion and exclusion at search (D6): screening before embedding, `flagged` payload (only when true), `must_not` filter in search, the `flagged` payload index in server mode, fake store parity; tests for the three ingestion scenarios (outcome line count, same file again) and the three store scenarios (flagged best match, old collection, count includes flagged) against the embedded store, and the Qdrant-server marked test
- [x] 5.2 Question screening, delimited passages and the data prompt (D7, D8): `question_flagged` refusal before embedding, `render_context` delimiters with neutralised forged sequences, the system prompt wording; tests for the refusal without search or model, the message shape with an injected sentence inside a passage, forged-delimiter neutralisation, sources unchanged, and the contract test of `build_messages`
- [x] 5.3 Support check, output filter and document diversity (D9): `app/domain/guardrails.py`, `AskQuestion` reasons `unsupported` and `output_filtered` with the share logged, `select_context` per-document cap; tests for a fabricated answer, a grounded answer, the prompt-leak and foreign-URL filters, document and page references, the repetitive-document and single-document selections, and the outcome-line privacy

## 6. Container hardening

- [x] 6.1 Compose runtime options (D12) for `app` and `qdrant`, `TMPDIR` and cache paths on tmpfs, `uv-audit` on pre-push; verify the full compose test suite and the smoke suite pass under the options, extend the smoke suite with the `docker inspect` checks and the write-outside-tmpfs check, and record the measured peak memory that sets the limit

## 7. Adversarial suite and measurements

- [x] 7.1 `tests/adversarial/` (D10): injection cases in five languages with OCR-noisy variants (direct and indirect through generated PDFs), `hostile_pdfs.py` generators with the expected code per file, limiter tests; verify every case passes on the host and in the test image, with the real detector where marked
- [x] 7.2 `benchmarks/adversarial/` measurements (D10): detector precision and recall by threshold per language on adversarial against benign corpus chunks; cost per page and per question; the prompt change paired on the answer-quality sample with the `llm_eval` harness; the support-share distribution of correct answers and the false-refusal rate of the support check and output filter; the benchmark client at its measured rate under the default limits; set `INJECTION_THRESHOLD` and `RAG_SUPPORT_MIN_SHARE` from the results and record everything in `docs/BENCHMARKS.md`

## 7b. Instruction patterns (D5b, from the 7.1 measurement)

- [x] 7.3 `app/domain/injection_patterns.py`, `PatternDetector`, `CompositeDetector`, `INJECTION_PATTERNS` setting and wiring; tests per attack class and language (clean and OCR-noisy), zero matches on every benign control, composite max, the setting off; the adversarial suite's measured floors raised to the combined detector's rates

## 8. Documentation and limitations

- [x] 8.1 README (defences and their settings, the `HF_TOKEN` build secret and the Hugging Face access request, the opt-out build, "Built with Llama", the limits and the no-proxy caveat), `.env.example`; `docs/KNOWN_LIMITATIONS.md`: replace R3 and P5's render bound with what remains after this change (residual injection risk, pause granularity, peer-address limiting behind a NAT), and any false-refusal rate from 7.2

## 9. Review and wrap-up

- [x] 9.1 Independent review (bypasses of the limiter through pipelining or `Expect: 100-continue`, delimiter forging through Unicode look-alikes, flagged chunks reachable through counting or repair paths, detector exceptions on empty or huge texts, secrets in layers, read-only root with every endpoint, log content); fix every fixable finding with tests
- [x] 9.2 Final verification: host `pytest --cov`, ruff, mypy, all pre-commit hooks including `uv audit`, `openspec validate --all --strict`, the full compose test suite with the model, the smoke suite against the hardened stack with the `gpu` profile, and the e2e suite

## Workflow follow-up

- Archive the change after the review requirements are satisfied, then verify the archived specs.
