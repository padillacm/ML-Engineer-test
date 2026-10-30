# Security report

What the service defends against, how, and with what measured effect. The work was done as the OpenSpec change `add-defensive-hardening` (proposal, design D1-D13 and D5b, spec deltas); the measurements are in [`BENCHMARKS.md`](BENCHMARKS.md) section 22 and the residual risks in [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md) (S1-S7, R3, P5). This page is the summary that ties them together.

## Scope and threat model

The service is a proof of concept: it has **no authentication** and no TLS, by decision (a reverse proxy's job before it leaves the host; it binds to `127.0.0.1`). Within that scope the defences assume an attacker who can:

- send any HTTP request to the four endpoints, at any rate, from one address or a few;
- upload a PDF crafted to cost CPU, memory or time (render bombs, page-tree tricks, compression bombs, deep nesting);
- write instructions into a document that will be ingested (indirect prompt injection), or into a question (direct injection), to make the language model ignore its rules, leak its instructions, cite foreign sources or give a chosen answer.

Out of scope: identity and access control, network attacks below HTTP, the host and Docker daemon, supply-chain attacks on the pinned images and models beyond hash checks, content moderation of answers (the service answers only from its own documents).

## Starting point (audit)

Already in place before the change: body limits per route, PDF validation by content, page-count limit with a verified page tree, a pixel budget per page, Tesseract as a subprocess with a timeout and process-group kill, a 2,000-character question limit, uniform error bodies that never echo input, redacted logs, secrets as files, a non-root image with hash-pinned offline models, bounded job and question slots with cancellation.

Gaps found:

| Gap | Where it was recorded |
|---|---|
| No limit on what one client may ask over time or hold at once | design `add-deployment` (deferred) |
| A hostile page could hold the PDFium lock for 33 s | `KNOWN_LIMITATIONS` P5 |
| Unbounded recognised text per page (NER refused late, with a 500) | review |
| Document text reached the model verbatim; no injection defence beyond the system prompt | `KNOWN_LIMITATIONS` R3 |
| Answers were not checked against the context they came from | `KNOWN_LIMITATIONS` R2 |
| API docs exposed by default; no protective headers | review |
| Containers with a writable root and default capabilities; dependency audit only on request | review |

## Defences by layer

### HTTP edge

- **Rate limits per client address** (token bucket per address and route class, uploads and questions) and a **per-address cap on running jobs**, decided before any body byte is read; over a limit: 429 `rate_limited` with `Retry-After`. Health probes are never limited. Defaults are above the measured single-client throughput; 0 turns a limit off.
- **Server bounds**: Uvicorn `--limit-concurrency 64`, `--timeout-keep-alive 5`, `--http h11`.
- **API documentation off** by default (`EXPOSE_DOCS`); `Cache-Control: no-store`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer` on API responses.

### PDF ingestion

- **Render time budget**: pages are drawn with PDFium's progressive API and stopped at `OCR_RENDER_TIMEOUT_S` (`pdf_unreadable`).
- **Pre-check before drawing**: declared image pixels of the page, its nested forms and its annotation appearance streams are summed against `OCR_MAX_IMAGE_PIXELS` (`too_large`); forms nested deeper than PDFium's pause-poll depth (8) are refused.
- **Page text cap**: `MAX_PAGE_CHARS` refuses a page before NER, chunking or embedding (`too_large`).

### Prompt injection and answer integrity

- **Detection at ingestion and at answering**: Llama Prompt Guard 2 (86M, five languages) plus a rule detector of instruction patterns tolerant of case, spacing and OCR substitutions; the higher score counts.
  - At ingestion the **page** is the unit (the model's context is the whole parent page of a retrieved chunk): every chunk of a flagged page is stored flagged, counted, and never returned by a search. The document stays complete.
  - A flagged question gets the fixed refusal without a search or a model call.
- **Passages as data**: each passage is enclosed in `<<<passage n: document <id>, page <p>>>>` ... `<<<end passage n>>>`; runs of `<`/`>` and their look-alikes (`‹ « ＜`) in page text are broken so a document cannot forge a boundary; the instructions say passage content is never an instruction.
- **Output checks**: an answer is returned only if at least `RAG_SUPPORT_MIN_SHARE` of its content tokens appear in the context the model saw, and if it neither quotes the instructions nor cites a URL, document or page that was not shown (`unsupported`, `output_filtered`).
- **Context diversity**: one document gives at most `RAG_MAX_CHUNKS_PER_DOCUMENT` context chunks when several documents matched.

### Containers and supply chain

- Service and Qdrant run with a read-only root, tmpfs scratch, all capabilities dropped, `no-new-privileges`, PID and memory limits.
- The detector model is downloaded at build time with a BuildKit secret (`HF_TOKEN`, in no layer and in no container), every file sha256-pinned; `INJECTION_MODEL=0` builds without it.
- `uv audit` of the lock file runs on every push.

### Observability

Every guardrail outcome is logged by count and reason, never by content: `chunks_flagged`, `injection_score`, `support_share`, `filter_reason`, `refusal_reason`, `rate_limited`.

## Measured results

All from `BENCHMARKS.md` section 22, with the before-and-after table of times and resources in 22.8 (CPU, the 600-document synthetic corpus, Gemma 4 e4b on the local GPU).

| What | Result |
|---|---|
| Hostile page that took 4.3 s to draw | stopped in about 70 ms with a 50 ms budget |
| Form nested 9 levels deep (was drawn as one operation, 2.2 s) | refused in 30-40 ms |
| Annotation stamp hiding a 400-megapixel image | refused in under 1 ms |
| Prompt Guard 2 alone, threshold 0.5 | 75 % of direct injections, 35 % of indirect ones, 32.5 % of OCR-noisy indirect ones; 0.29 % of corpus pages flagged |
| Model + patterns | 100 % of the adversarial suite (a floor check: the patterns were written against it); 0.39 % of corpus pages flagged; none of the 750 benchmark questions flagged |
| Patterns alone | 0 matches on 2,776 generator sentences and the benign controls; one OCR-garbage corpus page |
| New prompt, raw model replies on identical contexts | -0.8 points of accuracy [-2.0, +0.4] |
| Service accuracy with every guardrail | 71.0 against 72.3, paired -1.3 [-2.8, +0.0] |
| Correct answers refused by the output checks | 0.47 % (2 of 428) |
| Detector cost on CPU, ingestion | -38 % throughput sequential, -59 % with 12 clients (about 1.3 s per page in this run; 1.5 s in the CPU-against-GPU comparison of section 23) |
| Patterns cost | about 1.5 ms per page, no measurable throughput loss |
| Detector cost per question | about 130 ms |
| Detector memory, start-up and image | +230-300 MiB of RAM, +1.0-1.6 s to ready, +1.1 GB of image layer |
| Limiter, independent review | no bypass found (pipelining, `Expect`, chunked, path variants, HEAD/OPTIONS) |

The detector's CPU cost is why the models moved to the GPU in the follow-up change `add-gpu-inference` (BENCHMARKS 23): there the detector costs about 94 ms per page and 19 ms per question, ingestion with every defence on runs at 20.0 docs/min sequential and 132.9 with 12 clients, and turning the detector or the patterns off changes throughput by 0-5 %. Questions are screened in a detector slot kept for them, so page screenings no longer delay them on the CPU build either.

## Independent review

The review (task 9.1) probed every layer as an attacker would and fixed what it could, each with a regression test:

- **High:** images inside annotation appearance streams escaped the pixel pre-check. Fixed: annotations are walked too.
- **Medium:** look-alike brackets could forge a passage boundary when the pattern detector is off. Fixed: they are neutralised like ASCII ones.
- **Low:** document references in Spanish, Portuguese or German, and compact ids, escaped the output filter. Fixed.
- **Not fixed, documented:** images in tiling patterns and shadings are not counted (no API access); `INJECTION_THRESHOLD=0` flags everything; flags are not recomputed for documents already indexed; bare domains pass the URL check; a planted `Question:` line stays inside its passage but is not neutralised.

Regular-expression timing under adversarial input: the worst pattern takes 6.6 ms on a 30,000-character near-miss text; the whole set 20-35 ms.

## Residual risk

- **Authentication** is absent by decision; the limits key on peer addresses, so behind a NAT or proxy they are shared.
- **Prompt injection** is reduced, not eliminated: an instruction phrased outside the patterns and missed by the model can still steer an answer within the context's own words. The logs make such attempts visible.
- **Rendering**: one huge vector operation and content-stream parsing before the first pause poll are still unbounded by the budget (bounded by page size and upload size).
- **False positives**: a legitimate page flagged by the detector silently leaves the answers; `chunks_flagged` is the only signal.

Each item has its evidence and what would remove it in `KNOWN_LIMITATIONS.md` S1-S7.

## Operating the defences

| Setting | Default | Notes |
|---|---|---|
| `RATE_LIMIT_UPLOADS_PER_MINUTE` / `_BURST` | 60 / 20 | 0 turns the rate off |
| `RATE_LIMIT_QUESTIONS_PER_MINUTE` / `_BURST` | 240 / 60 | |
| `MAX_JOBS_PER_CLIENT` | 4 | 0 turns the cap off; set it to 0 behind a shared address |
| `EXPOSE_DOCS` | false | |
| `OCR_RENDER_TIMEOUT_S`, `OCR_MAX_IMAGE_PIXELS`, `MAX_PAGE_CHARS` | 10, 80,000,000, 30,000 | |
| `INJECTION_MODEL_PATH` | set by the image | unset: patterns only |
| `INJECTION_PATTERNS` | true | |
| `INJECTION_THRESHOLD` | 0.5 | applies to documents ingested afterwards |
| `RAG_SUPPORT_MIN_SHARE` | 0.6 | keeps 99.5 % of correct answers |
| `RAG_MAX_CHUNKS_PER_DOCUMENT` | 2 | |

Tests: `tests/adversarial/` is the regression corpus of the attacks above (injections in five languages with OCR-noisy variants, hostile PDFs, limiter floods); `injection_model`-marked tests run with the real detector.
