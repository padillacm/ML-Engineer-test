# Tasks

## 1. Baseline

- [x] 1.1 Archive `add-rag-pipeline`; verify `openspec validate --all --strict` passes with 10 specs

## 2. Dependencies

- [x] 2.1 Add `pydantic-settings` and `python-multipart` to `pyproject.toml`, `uv lock`, `uv sync --extra dev`; verify imports, `uv-audit` and that nothing else was added

## 3. Core changes

- [x] 3.1 Add codes `invalid_question` and `service_busy`; make `AskQuestion` raise `InvalidInput("invalid_question")` for a blank question; verify the `grounded-answering` scenario (embedder and store not called) and the error tests
- [x] 3.2 Add `app/domain/uploads.py` (`validate_pdf_bytes`): verify empty, exact limit, one byte over, header at 0 and at 1,023, header at 1,024 rejected, PNG/ZIP/HTML rejected
- [x] 3.3 Add `TesseractOcrConfig.max_pages` (default 50), checked after opening and before rendering; verify limit and limit + 1 with a renderer spy (nothing rendered) and the config validation

## 4. Settings and observability

- [x] 4.1 Implement `app/settings.py` (D1 table); verify defaults equal the benchmarked values, overrides, required variables, invalid values naming the variable, masked secrets in `repr`, `.env` loading
- [x] 4.2 Implement `app/observability.py` (JSON and text formatters, request-id context, `configure_logging`); verify field set, request id propagation across threads, allow-list of extra fields

## 5. HTTP edge

- [x] 5.1 Implement `app/bootstrap.py` (`build_container`, `UnconfiguredLLM`, close) and `create_app` with lifespan in `app/main.py`; verify with fake factories that models are built once and closed at shutdown, that a `ConfigurationError` names the variable, and that the service starts without an LLM
- [x] 5.2 Implement `app/api/{errors,uploads,dependencies,middleware,health,extract,rag}.py`; verify with fakes and `TestClient` every scenario of `http-api` and `observability` (validation, status mapping for every code, `Retry-After`, request id, busy limit, no personal data in logs, file closed on success and failure)
- [x] 5.3 Delete `app/services/`; remove the ruff/mypy exclusions; add the new modules to strict mypy and coverage; extend the architecture guard for `app/api`; verify ruff, mypy, guard and coverage >= 90 %

## 6. End-to-end in the container

- [x] 6.1 HTTP end-to-end tests in compose with real OCR, NER, e5 and Qdrant and a scripted LLM: `/api/extract` on the sample PDFs (names, boxes in PDF points, matches), `/api/ingest` twice (`indexed`, `already_indexed`), `/api/ask` answerable and off-topic, `/health/ready` with Qdrant up and stopped; verify the full compose suite passes

## 7. Documentation

- [x] 7.1 README: how to run the service (environment variables, `.env`, endpoints with `curl` examples, error codes table); record any unfixable finding in `docs/KNOWN_LIMITATIONS.md`

## 8. Evidence

- [x] 8.1 Parity and overhead on the 60-document sample (HTTP against direct use cases); record in `docs/BENCHMARKS.md`
- [x] 8.2 Concurrency measurement for `MAX_CONCURRENT_JOBS` and `BUSY_WAIT_S` (D7); set the defaults from it and record the table
- [x] 8.3 Resource check: file descriptors, temporary files and RSS around 200 mixed requests; record the result

## 9. Review and wrap-up

- [x] 9.1 Independent review (adversarial uploads, malformed forms, huge headers, slow bodies, dependency failures, log content); fix every fixable finding with tests
- [x] 9.2 Final verification: host `pytest --cov`, ruff, mypy, all pre-commit hooks, `uv-audit`, `openspec validate --all --strict`, full compose suite

## 10. Limitation fixes (design D10)

- [x] 10.1 NER pool (`NER_POOL_SIZE`): pipelines sharing weights; verify identical output to pool 1 on the 60-document sample, no tokenizer errors under 16 concurrent threads, and measure throughput for pool sizes 1, 2, 4 (and the concurrency default again)
- [x] 10.2 Cancellation on disconnect: token checked between pages and embedding batches; verify with a disconnecting client that OCR stops after the current page, the slot is freed and the log records 499 `request_cancelled`
- [x] 10.3 Overstated page count: verify a 1-page PDF declaring `/Count 1000000` is processed, and a real 51-page PDF is still refused before rendering
- [x] 10.4 Drain after early 413: verify with real Uvicorn that a client sending a body slightly over the limit reads the 413, and that a body far over the drain limit is cut
- [x] 10.5 Question limit (`MAX_CONCURRENT_QUESTIONS`) and readiness ping timeout: verify 503 `service_busy` within `BUSY_WAIT_S` for questions over the limit and a hung store's check thread ending within `READINESS_TIMEOUT_S`
- [x] 10.6 `LOG_EXCEPTION_MESSAGES` opt-in and `MAX_NAME_CHARS`: verify messages logged only when enabled, names over 100 characters rejected as `invalid_names`, and the envelope allowance covering the worst-case valid form
- [x] 10.7 50-page documents at the job limit: peak memory and Tesseract processes; update `docs/KNOWN_LIMITATIONS.md` with only what remains

## 11. Second review (D10 fixes)

- [x] 11.1 Independent review of tasks 10.x; fix every fixable finding with tests: one `TORCH_THREADS` (process-wide threads, the former per-model variables refused) and a CPU oversubscription warning; `Connection: close` on every early 413; cancellation on an outer cancel and on a failing `receive`, 499 for a request cancelled before any response; page checks without loading or parsing pages and no silent cut at the limit; docs and limitations updated
- [x] 11.2 Final verification: host suite with coverage, ruff, mypy, all pre-commit hooks, `uv-audit`, `openspec validate --all --strict`, rebuilt image and full compose suite
