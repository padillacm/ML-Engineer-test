# Proposal

## Why

`POST /api/extract` must return every person name in a scanned PDF with the page and box where it is printed, plus fuzzy matches against the names the caller supplies. Blocks 1 and 2 produce OCR words with boxes in PDF space and a matcher; nothing yet finds the names. The end-to-end benchmark on real Tesseract output (`docs/BENCHMARKS.md` section 4) fixed the approach: `Davlan/xlm-roberta-base-ner-hrl` on the whole page with line breaks as spaces and ALL-CAPS words title-cased, then a small set of span rules, reaches mention F1 92.0 % [91.2-92.7] (precision 94.2, recall 89.9, box IoU >= 0.5) over en/es/pt/fr/de. Raw model spans score 75.4; the rules are worth 16.6 points, so they are part of the product, not polish.

## What Changes

- Archive `add-domain-foundation` and `add-ocr-engine` so their specs become the accepted baseline in `openspec/specs/` (this change modifies `page-ocr`).
- OCR words gain the index of the text line they belong to (from Tesseract's block/paragraph/line), so spans can be split at line breaks without guessing from geometry.
- Add a transformer NER adapter under `app/adapters/ner/` implementing `NameRecognizer` (Hugging Face `transformers` token classification, `aggregation_strategy="first"`, `stride=64`, CPU `torch`), with model id, revision and local model path configurable; the model is loaded once and runs offline.
- Add pure domain code for the NER input and output: building the page text with ALL-CAPS words title-cased (offsets preserved), and the recommended span rules (title stripping, hyphen/apostrophe merge, "Last, First" merge up to a 4-token surname, initial extension, drop single-token spans unless titled, drop lowercase/digit/short spans, split at line breaks).
- Add the `ExtractNames` use case in `app/application/`: OCR, per-page NER, span rules, one box per mention via `PageText.locate`, and fuzzy matching against the requested pairs with one match per distinct extracted name.
- Dependencies: `transformers` and `torch` (CPU wheels from the PyTorch CPU index); the model is baked into the container image at a pinned revision.

## Capabilities

### New Capabilities
- `name-extraction`: finding person names on OCR'd pages, turning model output into clean name mentions, locating each mention as a box, and matching extracted names against the requested list.

### Modified Capabilities
- `page-ocr`: every recognised word also carries the index of its text line on the page.

## Impact

- New: `app/adapters/ner/`, `app/domain/names.py` (text preparation and span rules), `app/application/extract_names.py`; tests under `tests/domain/`, `tests/application/`, `tests/adapters/ner/`, `tests/integration/`.
- Changed: `PageWord` gains a `line` field; the OCR adapter fills it; `FakeOcrEngine` and fixtures follow.
- `pyproject.toml`/`uv.lock`: NER dependencies with the CPU torch index; `Dockerfile`: model download at build time into the image, offline flags at run time.
- Image size grows by roughly 0.2-0.4 GB (CPU torch) plus 1.1 GB (model). Tests that need the model are marked and run in the container.
- Out of scope: the HTTP endpoint, upload validation and settings from environment (block 5); embeddings and RAG (block 4).
