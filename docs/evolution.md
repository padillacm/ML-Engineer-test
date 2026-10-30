# Evolving the solution

Every default of the service was measured on a synthetic corpus ([Benchmarks](benchmark-summary.md)). This page says how to take it further on real documents: which data to collect, what to measure again, which kind of model to try for each finding, and what each choice costs in privacy. The order follows the **measured** size of each loss; where the size on real documents is unknown, the first step is to measure it, not to guess it. No effort estimates are given, because none was measured; each row says what it needs instead.

## 1. Where the measured losses are

| Area | What the benchmark measured | Source | What it implies |
|---|---|---|---|
| Retrieval | The answer reached the model's context for 74.7 % of the questions (448 of 600); when it did, the model answered 95.5 % of them correctly | [record](BENCHMARKS.md) 16 | Most of the end-to-end loss (72.3 % accuracy) is retrieval, not the model |
| Language model | Gemma 4 12B scored 72.8 % against 72.3 % for e4b, inside each other's interval | 16 | A larger model does not fix the loss on this corpus |
| Cross-lingual questions | Same-language minus cross-lingual recall@5: +12.8 points for e5-base, +3.8 for `snowflake-arctic-embed-m-v2.0` | 6 | Matters only if real users ask in another language than the document's |
| Relevance threshold | Keeps 95 % of answerable questions on the corpus; on the three real samples answerable questions score 0.755-0.803, just under 0.808 | 8, [R1](KNOWN_LIMITATIONS.md) | Real documents will move the threshold |
| OCR on hard scans | Name recall about 87-90 % and box IoU >= 0.5 about 81-84 % on the hard tier, against 99-100 % on clean pages | 2.6, 2.8, [O1](KNOWN_LIMITATIONS.md) | The impact depends on the share of hard scans in real traffic (20 % in the corpus) |
| Names | NER precision 94.2, recall 89.9; matching recall 87.5 % with zero false matches | 4, 5 | Organisations read as people (N1) and OCR typos (M2) are the residual errors |
| Injection | The detector model alone flags 75 % of direct and 35 % of indirect injections; the patterns cover the rest of the suite they were written against | 22, [S3](KNOWN_LIMITATIONS.md) | Unknown on real attacks; the suite is a floor check |
| Capacity | One CPU container: about 5,200 extractions or 2,100 ingestions per hour; 8,000 ingestions on the GPU build | 23 | 1,000 PDFs per hour already fit; capacity is not the first lever |

## 2. Data to collect first

Everything below depends on it: the thresholds and rules were tuned on the corpus they are reported on, with no held-out split ([record](BENCHMARKS.md) section 10).

| Data | What it is for | Size that gives usable intervals |
|---|---|---|
| Real scans of the target domain, names and boxes labelled, hard scans included | OCR, NER and box accuracy; the real share of hard scans | a few hundred pages, as the corpus' 1,045 |
| Real questions, answerable and unanswerable, with their answer passage, including paraphrased and cross-lingual ones | Retrieval recall, the relevance threshold, model accuracy | a few hundred, as the 600 of section 16 (about ±4 points) |
| Name lists with real near-miss names of other people | The match threshold | as the 200 lists with 893 decoys of section 5 |
| Pages flagged by the detector in real use, read by a person, and known attacks | The injection threshold and the false-flag rate | every flagged page until the rate is stable |
| Production signals: refusal and flag rates, score distributions, OCR confidence, latency per stage | Drift and capacity | continuous; the outcome log lines already carry the best score, the refusal reason, the injection score, the support share, the flag counts and the stage times, without document text (OCR confidence is not logged) |
| Reviewer corrections in the audit viewer | New labels for OCR, NER and boxes | grows with use (needs a way to store them, not built) |

## 3. What to measure again

With the same method as now: the same items for every configuration, paired differences with 95 % intervals, and a change adopted only when its interval excludes zero.

1. The three thresholds, by the [recalibration procedure](benchmark-summary.md#recalibration-procedure): relevance (0.808), match (0.90), injection (0.5).
2. Retrieval: recall@5 and answer-in-context on real questions, dense alone against hybrid (dense + BM25), with and without a reranker.
3. OCR and names: CER, name recall and box IoU on real scans, split by scan quality.
4. Answer accuracy and refusals end to end, and when the answer was in the context, to keep the retrieval and model losses apart.
5. Throughput with real document lengths, more than 12 concurrent jobs and replicas sharing one GPU, none of which was measured.

## 4. If you find X, try Y

Candidates are named so they can be compared, **not measured unless the row says so**. Licences are as listed on each model card in October 2026; check them again before adoption.

| If the measurement shows | Try | Candidates (licence) | What it needs | Adopt when |
|---|---|---|---|---|
| Answer-in-context well below the model's accuracy in context (today 74.7 % against 95.5 %) | A reranker on the `gpu` profile (**measured on the corpus and the samples, section 4.1**), then hybrid retrieval | `BAAI/bge-reranker-v2-m3` (Apache-2.0); BM25 sparse vectors in Qdrant (no model) | Code in the retrieval adapter; the reranker adds a model call per question | Paired answer-in-context gain whose interval excludes zero, p95 within budget |
| A large cross-lingual gap on real questions | A cross-lingual embedder | `Snowflake/snowflake-arctic-embed-m-v2.0` (Apache-2.0, runs code shipped with the model), `BAAI/bge-m3` (MIT); both measured in section 6 | Re-ingestion of every document (a new collection) | Overall recall@5 not worse on same-language questions |
| Many answerable questions under 0.808 | Recalibrate first on the CPU image; on the `gpu` profile the reranker's score as the gate (**measured, section 4.1**: on the samples no cosine threshold separates) | as above | A setting; a reranker as above | 95 % of answerable kept on real questions |
| Correct paraphrased answers refused by the support check (S4) | An entailment check instead of word overlap | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (MIT) | A fourth in-process model | Fewer wrong refusals with no rise in unsupported answers |
| Real hard scans are frequent and lose names | Stronger OCR, first for low-confidence pages only | `tessdata_best` (Apache-2.0, same engine); PaddleOCR or docTR (Apache-2.0; check their dependencies' licences) | A new OCR adapter and image layer; a GPU helps the neural engines | Name recall gain on hard pages, cost per page known |
| Organisations or signatures reported as people (N1, N9) | Fine-tune the current NER on the real labels | `Davlan/xlm-roberta-base-ner-hrl` fine-tuned; GLiNER multi v2.1 (Apache-2.0) was behind on public data (section 3) | Labelled real pages; a training pipeline | Precision gain without recall loss, on a held-out split |
| Real injections that the detector misses | Tune the threshold, then a second detector or a review queue | Granite Guardian (Apache-2.0; 2-8B parameters, a GPU) | A GPU budget, or a person reading flagged pages | Higher flag rate on real attacks, false flags acceptable to reviewers |
| Multi-document or comparative questions fail | The router of [Design](../DESIGN.md) section 5 | Gemma 4 12B (measured: no gain on this corpus) or a hosted model | Logged route signals | The router's paired accuracy within the larger model's interval at lower cost |
| Throughput short of the traffic | Replicas, OCR workers, then faster runtimes | ONNX or quantised versions of the current models | Measurement of a GPU-bound load (P18) | Measured on the target hardware |

### 4.1 Measured: a reranker on the `gpu` profile

Two cross-encoder rerankers were run over the first 20 candidates of e5-base, with the protocol of [record](BENCHMARKS.md) sections 6-8: the 600-document corpus, `s800_o12` chunks, one global index, paired bootstrap with 95 % intervals. They were also run on the three sample PDFs, chunked by the service's own OCR and chunker. The dense baseline reproduces the record's numbers. The scripts are kept outside the repository, and the run is not yet a section of the record.

| | Dense today (e5-base) | + `BAAI/bge-reranker-v2-m3` | + `Qwen/Qwen3-Reranker-0.6B` |
|---|---|---|---|
| Answer in the first chunk (2,871 same-language questions) | 38.4 % | 52.8 % (+14.4 [+12.7, +16.1]) | 47.3 % (+8.9 [+7.1, +10.6]) |
| Answer on the pages sent to the model | 77.0 % | 84.0 % (+6.9 [+5.7, +8.3]) | 79.7 % (+2.7 [+1.2, +4.4]) |
| Same, 600 cross-lingual questions | 58.0 % | 66.5 % (+8.5 [+5.2, +12.0]) | 66.0 % (+8.0 [+4.5, +11.5]) |
| Same, the 578 English questions (30 candidates) | 83.4 % | 87.0 % (+3.6 [+0.9, +6.7]) | 72.5 % (-10.9 [-14.5, -7.1]) |
| Unanswerable refused by a gate that keeps 95 % of answerable, cross-document items left out (200) | 66.0 % (cosine 0.808) | 92.0 % (score -2.31) | 95.5 % (score 3.03) |
| Absent-entity questions refused (100) | 38 % | 87 % | 91 % |
| Answerable questions kept with the answer in context | 73.7 % | 80.0 % (+6.3 [+4.8, +7.8]) | 75.9 % (+2.2 [+0.6, +3.8]) |
| Sample PDFs: answerable questions kept, of 25 | 11 | 24 | 18 |
| Sample PDFs: unanswerable questions refused, of 12 | 11 | 12 | 12 |
| Added time per question on the RTX 4070 Ti, p50 / p95 | - | 56 / 67 ms | 249 / 344 ms |
| Peak GPU memory of the reranker | - | 1.2 GB | 2.6 GB |

- **The gate is the strongest reason.** On the sample PDFs the answerable questions score 0.755-0.887 and the unanswerable ones up to 0.815, so no cosine threshold separates them: 0.78 keeps 21 of 25 and refuses 9 of 12, 0.75 keeps 25 and refuses 6. With bge the lowest answerable question scores -3.34 and the highest unanswerable one -3.73. Its threshold was set on the corpus, not on the samples. This addresses [R1](KNOWN_LIMITATIONS.md) and [R2](KNOWN_LIMITATIONS.md) on the `gpu` profile.
- **bge gains in every language and noise tier** (with 30 candidates: clean +8.3, hard scans +3.1 points of answer in context). Its 20, 30 and 50 candidates give results inside each other's intervals, so 20 is enough.
- **Qwen3 was not chosen.** It gains less, loses 10.9 points on English questions, is 4-5 times slower, and needs twice the memory.
- **Not for the CPU image.** On 8 CPU threads bge takes about 6 s per question (p95 7.5 s), so the CPU image keeps the cosine gate, recalibrated (section 6, step 2).
- **Not measured yet.**
  - **Memory:** the GPU memory with Gemma and 12-client ingestion running. The worst measured case left 2.91 GB free ([record](BENCHMARKS.md) 23.11); about 1.6 GB would remain, which is an estimate.
  - **End to end:** the answer accuracy on the 600 questions of [record](BENCHMARKS.md) section 16.
  - **Real questions.** The corpus is synthetic. The 37 sample questions were written after reading the OCR text, before any score was seen.

**Next step.** On the `gpu` profile, add `bge-reranker-v2-m3` in the retrieval adapter: rerank the top 20 of e5 and gate on the reranker's top-1 score instead of the cosine. Adopt it once the two unmeasured checks pass: GPU memory under the worst-case load, and end-to-end accuracy not worse than today within its interval.

## 5. Privacy trade-off of each choice

The service today keeps every document on the host: models run in process, the `gpu` profile's language model runs locally, logs carry no document text by default, and `/api/extract` stores nothing. Each evolution changes that differently:

| Choice | What leaves the host or is kept | Trade-off | Mitigation |
|---|---|---|---|
| Hosted language model instead of local Gemma | Each question and its passages, which contain names | No GPU needed, but personal data goes to a provider; on this corpus it was not more accurate (69.5 %) | A data-processing agreement, a region and zero-retention option; or keep the local model |
| Hosted OCR, embedding or reranking APIs | Whole scans or their text | Possibly better quality with no hosting, but the most data leaves | Prefer the local candidates of section 4, all of which run in process |
| Collecting labelled real documents | Annotators see personal data; the labels are personal data | Without them nothing in section 3 can be measured | Legal basis, minimal sets, access control, retention limit; pseudonymising the names is not possible, since names are what is measured |
| Fine-tuning on real names | Model weights may memorise names | Better precision, but the weights become personal data, and removing one person means retraining | Prefer rules and off-the-shelf models while they suffice; keep the training set's lineage |
| Storing questions and reviewer corrections | Question text and corrected names | Needed for recalibration and new labels | Store scores and codes by default, text only for consented samples with a retention limit |
| Keeping ingested documents | Page text sits in Qdrant beside every chunk | Needed for answers and sources; there is no endpoint to delete a document ([S8](KNOWN_LIMITATIONS.md)) | A delete-by-document endpoint before real personal data is ingested |
| A second detector or a review queue | A person reads flagged pages | Catches real attacks the model misses | Limit the queue to flagged pages and to authorised reviewers |

## 6. Order of work

1. **Real data and a delete endpoint.** Every later decision rests on the data (section 2), and the delete endpoint is the privacy precondition for holding it.
2. **Recalibrate the thresholds.** It is a setting, with no code change, and R1 already shows that real documents move it.
3. **Retrieval.** It is the largest measured loss (section 1), and a larger model did not help. On the `gpu` profile the first change is the reranker of section 4.1, already measured on the corpus and the samples.
4. **OCR on hard scans and names,** in the order the real share of hard scans and the real error types put them.
5. **Guardrails,** sized by what the flagged real pages show.
6. **Capacity,** when measured traffic approaches the measured capacity.

Each step is re-measured before the next, so a finding can reorder the rest.
