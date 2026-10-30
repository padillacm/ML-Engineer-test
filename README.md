# Document Intelligence Service

A FastAPI service that reads scanned PDFs and returns every person's name with where it sits on the page (0-based page, box in PDF points), fuzzy-matches the names against a requested list, and answers questions about ingested documents only from those documents. Built for the ML Engineer take-home ([the assignment](docs/assignment.md)).

**Deliverables:** [DESIGN.md](DESIGN.md) and [DECISIONS.md](DECISIONS.md). Everything else is in the [documentation](docs/README.md), which also renders as a docsify site.

## Quick start

With Docker (no account or key needed):

```bash
docker compose up -d --wait
```

Then open the audit viewer at <http://127.0.0.1:8000/audit>. The [step-by-step guide](#run-it-step-by-step) at the end of this page covers the stack, the viewer, the tests and the documentation site.

## Documentation

| Page | What it covers |
|---|---|
| [Overview](docs/README.md) | What the service does and where each requirement of the assignment is met and tested |
| [Getting started](docs/getting-started.md) | Run it, try it, develop on it |
| [Architecture](docs/architecture.md) | Layers, components, the three request flows |
| [Design](DESIGN.md) | Choices and trade-offs, scaling to 1,000+ PDFs per hour, failure modes, model routing |
| [Decisions](DECISIONS.md) | Each notable decision, why, and the evidence; contract additions; use of AI tools |
| [Evolving the solution](docs/evolution.md) | What to collect, re-measure and try next on real documents, and the privacy cost of each choice |
| [HTTP API](docs/api.md), [Configuration](docs/configuration.md), [Running the stack](docs/operations.md), [Audit viewer](docs/audit-viewer.md), [Testing](docs/testing.md) | Reference |
| [Benchmarks](docs/benchmark-summary.md), [Security audit](docs/SECURITY.md), [Known limitations](docs/KNOWN_LIMITATIONS.md) | Evidence, with the full [benchmark record](docs/BENCHMARKS.md) |

It also renders as a site ([step 4](#4-browse-the-documentation-site)).

## Project layout

`app/` is a ports-and-adapters package: a pure domain, use cases behind `typing.Protocol` ports, one adapter per external technology, a thin HTTP edge and one composition root. The folder-by-folder map is in [Architecture](docs/architecture.md), section "Where the code is".

## Licences and attribution

The service code uses permissively licensed libraries; PyMuPDF (AGPL-3.0) is a test-only tool. The baked-in models: `Davlan/xlm-roberta-base-ner-hrl` (AFL-3.0), `intfloat/multilingual-e5-base` (MIT) and, when built with a Hugging Face token, `meta-llama/Llama-Prompt-Guard-2-86M`. Each model directory in the images carries a `NOTICE`.

This service is **built with Llama**: Llama 4 is licensed under the Llama 4 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. The licence is not OSI-approved ([Known limitations](docs/KNOWN_LIMITATIONS.md), S5).

## Run it step by step

You need Docker with Compose v2 (Docker Desktop on Windows or macOS). Steps 3 and 4 also need Python 3.12 and [uv](https://docs.astral.sh/uv/).

### 1. Start the stack

```bash
docker compose up -d --wait
```

The first run builds the image and downloads the pinned models (about 2.2 GB); the command returns once the service is healthy on `127.0.0.1:8000`. Check it:

```bash
curl http://127.0.0.1:8000/health/ready
```

Optional settings go in `.env` (copy `.env.example`):

- `HF_TOKEN`: a Hugging Face token of an account that accepted the Llama 4 Community License, to bake in the Prompt Guard 2 injection detector. Without it the instruction patterns run alone.
- `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`: a hosted OpenAI-compatible model for `/api/ask`. Or, on an NVIDIA GPU, start the stack with Gemma 4 on Ollama instead:

```bash
COMPOSE_PROFILES=gpu docker compose up -d --wait
```

Try the endpoints with the sample PDFs:

```bash
curl -F pdf_file=@sample_pdfs/meeting_minutes.pdf -F 'names=[{"first_name": "Elizabeth", "last_name": "Park"}]' http://127.0.0.1:8000/api/extract
```

```bash
curl -F pdf_file=@sample_pdfs/company_memo.pdf http://127.0.0.1:8000/api/ingest
```

```bash
curl -H 'Content-Type: application/json' -d '{"question": "Who was promoted to Vice President of Engineering?"}' http://127.0.0.1:8000/api/ask
```

Stop it with `docker compose --profile gpu down` (add `-v` to delete the indexed documents and the downloaded models). [Running the stack](docs/operations.md) has every option.

### 2. Open the audit viewer

With the stack running:

1. Open <http://127.0.0.1:8000/audit>.
2. Choose a PDF, for example `sample_pdfs/meeting_minutes.pdf`.
3. Optionally type the names to match, one per line (`Elizabeth Park`), and pick the language or leave auto-detect.
4. Press **Extract**. Every returned name is drawn as a box over the page: solid for a match, dashed for no match. Hover a box or pick a name in the side list to see its page, its box in points and its best match.

[Audit viewer](docs/audit-viewer.md) explains every control and warning.

### 3. Run the test suite

On the host (Tesseract and the models are not needed; the tests that need them are skipped, and each skip says why):

```bash
uv sync --extra dev --extra cpu
```

```bash
uv run pytest -q
```

The full suite, with Tesseract and the real models, runs in the test image next to a Qdrant server:

```bash
docker compose -f docker-compose.test.yml run --rm tests
```

[Testing](docs/testing.md) lists every suite, including the smoke and browser tests.

### 4. Browse the documentation site

```bash
uv run python scripts/serve_docs.py
```

Then open <http://127.0.0.1:3000/docs/> and stop the server with Ctrl+C. It serves only `docs/`, `DESIGN.md` and `DECISIONS.md`, on `127.0.0.1`; `--port` picks another port.
