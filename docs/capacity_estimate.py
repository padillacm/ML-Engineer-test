"""Capacity estimate: CPU core-seconds and documents/hour/core for /api/extract and /api/ingest.

Stdlib only.  Run:  python docs/capacity_estimate.py

Every constant below is copied from a benchmark result file (path in the comment).  Nothing is
estimated except the two derivations that are marked "derived" (linear interpolation of the
embedding time to the measured chunk length, and the 4-thread -> core-seconds conversion).  The
result is an order-of-magnitude figure for the hardware the inputs were measured on (see
docs/BENCHMARKS.md, section 9) and MUST be re-measured on the target hardware.

Not included (not measured): PDF open/page count, chunking, fuzzy matching, geometry, JSON
serialisation, vector-store upsert, HTTP overhead, model loading, and the LLM call of /api/ask
(outside this estimate).
"""

# ---------------------------------------------------------------- corpus shape
DOCS = 600  # data/README.md
PAGES = 1045  # data/README.md (1,045 pages in 600 documents)
PAGES_PER_DOC = PAGES / DOCS

# ---------------------------------------------------------------- OCR
# Source: ocr/results/timing_summary.md.
# 200 DPI, flattening ("flat"), psm 3, oem 1, OMP_THREAD_LIMIT=1; 30 first pages, sequential, one
# page at a time, Linux container (python:3.12-slim, Tesseract 5.5.0 tessdata_fast); rows of
# timing_grid_flat.jsonl.
OCR_SINGLE_S = 0.87  # t_ocr mean, language argument = the document language
OCR_ALL_S = 1.65  # t_ocr mean, language argument = eng+spa+por+fra+deu
RENDER_PREP_S = 0.15  # render + flatten mean (same for both rows)
# Throughput run (timing_summary.md, thr_16x1.jsonl): 54 pages, 16 worker processes,
# OMP_THREAD_LIMIT=1, 200 DPI flat psm3 single: "done 54 units in 7.9s (wall)".  16 = logical
# processors of the benchmark machine.
LOADED_PAGES, LOADED_WALL_S, LOADED_WORKERS = 54, 7.9, 16

# ---------------------------------------------------------------- NER
# Source: ner_corpus/results/e6_cpu_speed_xlmr.json.
# Davlan/xlm-roberta-base-ner-hrl, fp32, CPU, 4 threads, whole page in one call ("page" mode),
# after 2 warm-up pages.  Bucket 1500-2200 chars (8 pages, mean 1881 chars) is the one that
# contains the median OCR page length (1765 chars).
NER_BUCKET = "1500-2200"
NER_MEAN_MS, NER_MEDIAN_MS = 416.7, 373.8
NER_THREADS = 4

# ---------------------------------------------------------------- embeddings
# Source: C:/eb/out/tables/cpu_table.md; cpu_bench.jsonl.
# multilingual-e5-base, fp32, CPU, 4 threads, batch 16, median of 5 timed runs.
E5_MS_AT_120, E5_TOK_AT_120 = 106.2, 122  # "ms/passage @~120 tok", measured tokens 122
E5_MS_AT_400, E5_TOK_AT_400 = 356.2, 402  # "ms/passage @~400 tok", measured tokens 402
E5_THREADS = 4
# emb_corpus/logs: e5-base on the 'ocrflat' s800_o12 chunks: tokens mean 159 (arctic-m-v2 on the
# 'ocr' chunks: 158).
CHUNK_TOKENS = 159
# emb_corpus/logs/chunk_ocr.log: "ocr s800_o12: 3196 chunks, 3.06/page" (REPORT_tables.md shows
# 3.10/page because it divides by the 1,032 non-empty pages instead of 1,045).
CHUNKS_800_12 = 3196
CHUNKS_PER_PAGE = CHUNKS_800_12 / PAGES
CHUNKS_PER_PAGE_ALT = CHUNKS_800_12 / 1032

TARGET_DOCS_PER_HOUR = 1000


def embed_ms_per_chunk(tokens: float) -> float:
    """Derived: linear interpolation between the two measured points (122 and 402 tokens)."""
    slope = (E5_MS_AT_400 - E5_MS_AT_120) / (E5_TOK_AT_400 - E5_TOK_AT_120)
    return E5_MS_AT_120 + slope * (tokens - E5_TOK_AT_120)


def summarise(
    label: str, prep_ocr_core_s: float, ner_ms: float, chunks_per_page: float, embed_ms: float
) -> None:
    # derived: wall seconds x threads = core-seconds (upper bound)
    ner_core_s = ner_ms / 1000 * NER_THREADS
    emb_core_s = chunks_per_page * embed_ms / 1000 * E5_THREADS  # derived: same conversion
    extract_page = prep_ocr_core_s + ner_core_s
    ingest_page = prep_ocr_core_s + emb_core_s
    print(f"  {label}")
    print(
        f"    per page : render+prep+OCR {prep_ocr_core_s:.3f} s | NER {ner_core_s:.3f} core-s | "
        f"embedding {emb_core_s:.3f} core-s "
        f"({chunks_per_page:.3f} chunks x {embed_ms:.1f} ms x {E5_THREADS})"
    )
    for name, per_page in (
        ("/api/extract (OCR+NER)", extract_page),
        ("/api/ingest (OCR+chunk+embed)", ingest_page),
    ):
        per_doc = per_page * PAGES_PER_DOC
        docs_h_core = 3600 / per_doc
        cores = TARGET_DOCS_PER_HOUR * per_doc / 3600
        print(
            f"    {name:<30} {per_page:6.3f} core-s/page  {per_doc:6.3f} core-s/doc  "
            f"{docs_h_core:7.1f} docs/h/core  {cores:5.2f} cores for {TARGET_DOCS_PER_HOUR} docs/h"
        )


def main() -> None:
    embed_ms = embed_ms_per_chunk(CHUNK_TOKENS)
    print("Capacity estimate (core-seconds; 100 % core utilisation, no headroom)")
    print(f"pages per document = {PAGES} / {DOCS} = {PAGES_PER_DOC:.4f}")
    print(
        f"embedding ms/chunk at {CHUNK_TOKENS} tokens "
        f"(interpolated between {E5_TOK_AT_120} and {E5_TOK_AT_400} tokens) = {embed_ms:.1f}"
    )
    print(
        f"NER ms/page, bucket {NER_BUCKET} chars: "
        f"mean {NER_MEAN_MS}, median {NER_MEDIAN_MS} (primary = mean)"
    )
    print()
    print("Scenarios")
    summarise(
        "A  language known, idle machine (timing_grid_flat, single)",
        RENDER_PREP_S + OCR_SINGLE_S,
        NER_MEAN_MS,
        CHUNKS_PER_PAGE,
        embed_ms,
    )
    summarise(
        "B  all five languages in one pass (timing_grid_flat, all)",
        RENDER_PREP_S + OCR_ALL_S,
        NER_MEAN_MS,
        CHUNKS_PER_PAGE,
        embed_ms,
    )
    loaded = LOADED_WORKERS * LOADED_WALL_S / LOADED_PAGES
    summarise(
        f"C  as A, but render+prep+OCR taken from the {LOADED_WORKERS}-worker run "
        f"({LOADED_WORKERS} x {LOADED_WALL_S} s / {LOADED_PAGES} pages)",
        loaded,
        NER_MEAN_MS,
        CHUNKS_PER_PAGE,
        embed_ms,
    )
    summarise(
        "D  two-pass language handling, derived: "
        "render once + OCR(all languages) + OCR(single), every page",
        RENDER_PREP_S + OCR_ALL_S + OCR_SINGLE_S,
        NER_MEAN_MS,
        CHUNKS_PER_PAGE,
        embed_ms,
    )
    print()
    print("Sensitivity (scenario A)")
    summarise(
        "A with NER median instead of mean",
        RENDER_PREP_S + OCR_SINGLE_S,
        NER_MEDIAN_MS,
        CHUNKS_PER_PAGE,
        embed_ms,
    )
    summarise(
        "A with 3.10 chunks/page (3196 / 1032 non-empty pages)",
        RENDER_PREP_S + OCR_SINGLE_S,
        NER_MEAN_MS,
        CHUNKS_PER_PAGE_ALT,
        embed_ms,
    )


if __name__ == "__main__":
    main()
