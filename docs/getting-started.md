# Getting started

Two ways in: run the whole stack with Docker (the quickest way to see it work), or set up the Python environment to develop and run the tests.

## Run the stack with Docker

You need Docker with Compose v2 (Docker Desktop on Windows or macOS). No account or key is needed.

```bash
docker compose up -d --wait
```

The first run builds the image: it downloads the pinned NER and embedding models (about 2.2 GB, each file checked against its sha256) and pdf.js. The command returns once the service is healthy. The service listens on `127.0.0.1:8000` only, because it has no authentication.

> **Optional: the injection detector.** With `HF_TOKEN` in `.env` (a Hugging Face read token of an account that accepted the Llama 4 Community License), the build also bakes in Meta's Prompt Guard 2 detector. Without it the image is built without the detector and the instruction-pattern detector runs alone. See [Security audit](SECURITY.md).

Try it with the sample PDFs:

```bash
curl http://127.0.0.1:8000/health/ready
```

```bash
curl -F pdf_file=@sample_pdfs/meeting_minutes.pdf -F 'names=[{"first_name": "Elizabeth", "last_name": "Park"}]' http://127.0.0.1:8000/api/extract
```

```bash
curl -F pdf_file=@sample_pdfs/company_memo.pdf http://127.0.0.1:8000/api/ingest
```

```bash
curl -H 'Content-Type: application/json' -d '{"question": "Who was promoted to Vice President of Engineering?"}' http://127.0.0.1:8000/api/ask
```

Then open the audit viewer at <http://127.0.0.1:8000/audit>, choose a sample PDF and press **Extract**: every returned box is drawn over the page as the browser renders it.

**Questions need a language model.** Without one, `/api/ask` still refuses questions the documents cannot answer, and answers 503 `llm_unavailable` when it would need the model. Two ways to add one, both described in [Running the stack](operations.md):

- **A hosted OpenAI-compatible model:** set `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` in `.env` (copy `.env.example`; the key reaches the container as a secret file).
- **A local model on an NVIDIA GPU:** `COMPOSE_PROFILES=gpu docker compose up -d --wait` starts Ollama with Gemma 4 and points the service at it.

Stop the stack with `docker compose --profile gpu down`. Indexed documents and downloaded models stay in named volumes; add `-v` to delete them.

## Develop on the host

You need Python 3.12 and [uv](https://docs.astral.sh/uv/). torch comes from one of two extras, so name one of them: a sync without `cpu` or `cuda` installs no torch.

```bash
uv sync --extra dev --extra cpu
```

```bash
uv run pre-commit install
```

```bash
uv run pytest -q
```

The host has no Tesseract and no models, so the tests that need them are skipped (each skip says how to run it). The full suite runs in the test image, next to a real Qdrant:

```bash
docker compose -f docker-compose.test.yml run --rm tests
```

Lint and types: `uv run ruff check app tests scripts` and `uv run mypy app tests scripts`. The pre-commit hooks run these on each commit, and the tests with the coverage gate and the dependency audit on each push. [Testing](testing.md) lists every suite.

To run the service itself outside Docker, it needs the model directories, Tesseract and a vector store, all named by environment variables. [Configuration](configuration.md) lists them; the images set them for you.

## Read these docs locally

```bash
uv run python scripts/serve_docs.py
```

Then open <http://127.0.0.1:3000/docs/>. The script serves `docs/` and the two root deliverables only; docsify is loaded from a CDN, pinned by version and hash. Every page is also plain Markdown that GitHub renders.

## Where next

- [Architecture](architecture.md): layers, components and the three request flows.
- [Design](../DESIGN.md): choices, trade-offs, scaling and failure modes.
- [Decisions](../DECISIONS.md): what was decided, why, and the evidence.
- [HTTP API](api.md): every endpoint, error code and header.
