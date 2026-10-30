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
| Answer-in-context well below the model's accuracy in context (today 74.7 % against 95.5 %) | Hybrid retrieval, then a reranker | BM25 sparse vectors in Qdrant (no model); `BAAI/bge-reranker-v2-m3` (Apache-2.0) | Code in the retrieval adapter; the reranker adds a model call per question | Paired answer-in-context gain whose interval excludes zero, p95 within budget |
| A large cross-lingual gap on real questions | A cross-lingual embedder | `Snowflake/snowflake-arctic-embed-m-v2.0` (Apache-2.0, runs code shipped with the model), `BAAI/bge-m3` (MIT); both measured in section 6 | Re-ingestion of every document (a new collection) | Overall recall@5 not worse on same-language questions |
| Many answerable questions under 0.808 | Recalibrate first; a reranker score as the gate if the cosines stay compressed | as above | A setting; a reranker as above | 95 % of answerable kept on real questions |
| Correct paraphrased answers refused by the support check (S4) | An entailment check instead of word overlap | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (MIT) | A fourth in-process model | Fewer wrong refusals with no rise in unsupported answers |
| Real hard scans are frequent and lose names | Stronger OCR, first for low-confidence pages only | `tessdata_best` (Apache-2.0, same engine); PaddleOCR or docTR (Apache-2.0; check their dependencies' licences) | A new OCR adapter and image layer; a GPU helps the neural engines | Name recall gain on hard pages, cost per page known |
| Organisations or signatures reported as people (N1, N9) | Fine-tune the current NER on the real labels | `Davlan/xlm-roberta-base-ner-hrl` fine-tuned; GLiNER multi v2.1 (Apache-2.0) was behind on public data (section 3) | Labelled real pages; a training pipeline | Precision gain without recall loss, on a held-out split |
| Real injections that the detector misses | Tune the threshold, then a second detector or a review queue | Granite Guardian (Apache-2.0; 2-8B parameters, a GPU) | A GPU budget, or a person reading flagged pages | Higher flag rate on real attacks, false flags acceptable to reviewers |
| Multi-document or comparative questions fail | The router of [Design](../DESIGN.md) section 5 | Gemma 4 12B (measured: no gain on this corpus) or a hosted model | Logged route signals | The router's paired accuracy within the larger model's interval at lower cost |
| Throughput short of the traffic | Replicas, OCR workers, then faster runtimes | ONNX or quantised versions of the current models | Measurement of a GPU-bound load (P18) | Measured on the target hardware |

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
3. **Retrieval.** It is the largest measured loss (section 1), and a larger model did not help.
4. **OCR on hard scans and names,** in the order the real share of hard scans and the real error types put them.
5. **Guardrails,** sized by what the flagged real pages show.
6. **Capacity,** when measured traffic approaches the measured capacity.

Each step is re-measured before the next, so a finding can reorder the rest.
