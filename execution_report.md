# EvidenceForge evaluation

## Version 0.3.0 local validation

Nine tests passed on Windows/Python 3.12.14 in 73.59 seconds, including real E5 inference, independent HTTP processes sharing SQLite state, preserved ACLs after restart without the startup corpus, shared quotas and generation leases. After the model adapter change, the four operations cases passed again in 26.65 seconds. Ruff checks passed. The BM25 test reproduces the original 300-query published benchmark; the earlier full E5 metrics below retain their original dates because ranking did not change.

### Real local model

Qwen3-1.7B Q8_0 ran through llama.cpp build 11476 on CPU, four threads and a 4,096-token context. The model file's full SHA-256 was verified before loading. Twelve evenly spaced human annotations were selected from the pinned PersianQuAD test split, searching its 35 original contexts. Reference answers were used only for scoring. This sample was used during implementation and is not a held-out accuracy benchmark.

| Output contract | Gold context retrieved | Annotated answer contained in returned quotation | Status counts |
|---|---:|---:|---|
| boolean_and_quotation | 12/12 | 0/12 | {'generation_abstained': 12} |
| quotation_transcription | 12/12 | 5/12 | {'generation_invalid': 7, 'grounded_extract': 5} |
| span_id_selection | 12/12 | 9/12 | {'grounded_extract': 12} |

The first contract abstained on all twelve cases. Requiring copied quotations accepted five correct source quotations but rejected seven outputs. The final contract asks for an existing span ID and retrieves the source text in code. All 12 accepted final quotations were checked against the original Unicode source offsets. This prevents generated answer wording, but does not establish that every selected passage answers the question.

Three final selections did not contain an annotated reference answer; one selected a different context from the author's labelled context. The transcription contract's five accepted outputs all contained a reference answer, whereas span selection increases answer coverage while accepting three selections without a reference answer. This is a measured tradeoff, not a claim of perfect question answering. Returned selections carry `generation.requires_review=true`.

In all three source-removal cases, reference answer text was absent from the remaining corpus, yet the model still selected a passage. The source-text validator accepted those quotations because they existed in the supplied sources. This directly demonstrates the remaining abstention weakness: source membership is not an answerability test. This configuration is not validated for unattended answering.

The final run's median request time was 45.697 seconds and maximum was 49.573 seconds. These are observations on a shared Windows CPU, not controlled speed comparisons between contracts. The output budget is 64 tokens and the model writes an ID rather than answer prose.

Removing the annotated source produced {'grounded_extract': 3}. These are controlled source-removal stress cases, not human-labelled unanswerable questions; the report records whether reference answer text still occurs elsewhere. Simultaneous requests returned ['generation_busy', 'grounded_extract', 'generation_busy']. A restricted collection returned `no_evidence` without evidence, and the unavailable backend returned `generation_unavailable`. Remaining comprehension errors and CPU latency must be considered before using this configuration for unattended question answering.

[Reproducible model summary and provenance](reports/local-generation-v03.json). Full local HTTP records are retained outside the public source tree; the public summary contains question IDs, measured outcomes and raw-record hashes rather than redistributing corpus text.

### Persistent answer cache

One actual published question took 1.375 seconds on an application-cache miss against the already-loaded model. The model server's own prompt cache could be warm. After restarting the API process, the same answer came from SQLite in 0.043 seconds. Restricting the original documents then returned `no_evidence`, without serving the prior answer. These are single-request observations, not a cold-model benchmark. Cache entries hold a hashed context key and selected span ID, with a default five-minute lifetime and a 256-entry bound. Tenant, role, corpus revision, source text, model and policy changes partition the cache; imports invalidate it.

### Concurrent retrieval

One live HTTP process answered 100 distinct official SciFact test queries from eight client threads with SQLite enabled. All 100 responses matched their sequential references. The run measured 37.44 requests/second, median 184.087 ms and nearest-rank p95 402.302 ms. The measurement used a warm BM25 index, disabled dense inference and generation, and a quota of 2,000 requests/minute. It is not a production capacity guarantee. [Raw summary](reports/concurrent-bm25-v03.json).

### Deployment boundaries

SQLite solves persistence and worker coordination on one local host; it is not a distributed store. Environment API keys remain the identity mechanism. Real organization identity, TLS termination, filesystem protection, backups and external connection limits require deployment configuration. The installed live backend was llama.cpp; the optional Ollama adapter remains unverified against an installed Ollama runtime. [Current workflow results](https://github.com/saintedwind/evidenceforge-transformer/actions/workflows/ci.yml) and [release validation](https://github.com/saintedwind/evidenceforge-transformer/releases/tag/v0.3.0) record GitHub checks separately from these local runs.

## Earlier 0.2.1 record

## GitHub Actions, 6 November 2026

The [Linux validation run](https://github.com/saintedwind/evidenceforge-transformer/actions/runs/42278182844) passed on Python 3.11 and 3.12. Each job installed the package, passed Ruff checks, fetched the pinned SciFact files and ran the HTTP suite against the published records. Each reported four passed tests and one skipped test: the E5 HTTP case requires local weights and was checked separately on Windows below.

The Python 3.11 job also built the Docker image, started the non-root service with the real SciFact corpus mounted read-only, checked its health endpoint and retrieved source citations for an official test query. The first CI attempt failed when the readiness probe encountered a connection reset before startup completed. The probe now retries connection resets and timeouts as well as URL errors; the subsequent container run passed. This is a startup and query check, with throughput and sustained load outside its scope.

## Publication check, 6 November 2026

Version 0.2.1 passed five local test cases on Python 3.12.14. The suite includes twelve official SciFact queries sent concurrently through four HTTP workers and checks that their responses match the sequential requests. The dense HTTP case populates the manager's E5 view, imports a different access policy, then verifies that the former manager receives no documents. All source records and questions come from the published datasets.

The real E5 HTTP check used the first 48 unchanged SciFact rows. Its cold dense request took 16.579 seconds and the following hybrid request took 0.404 seconds. These are individual local requests. Ruff lint and formatting checks passed.

The retrieval results below retain their original run date. Version 0.2.1 adds regression coverage without changing the ranking algorithm.

## Evaluation run, 5 November 2026

Run on 5 November 2026 on Windows with Python 3.12.14. Both evaluations use records from the dataset publishers. The access tests assign roles to the existing SciFact corpus during the request. The dense HTTP test copies the first 48 corpus rows verbatim into its temporary directory.

## SciFact

The benchmark used 5,183 documents, 300 official test queries and 339 relevance judgments. Candidate depth was 100 per retriever in the benchmark and in the API. The CLI fetched and verified the pinned data files. The baseline reproduced the BM25 scores bundled with 0.1.0.

| Method | Recall@5 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|
| BM25 | 0.7209 | 0.6646 | 0.6340 |
| E5 | 0.7343 | 0.6770 | 0.6451 |
| BM25 + E5, RRF | 0.7455 | 0.6911 | 0.6657 |

The release encoder ran on CPU with six threads. The full run took 725.595 seconds, including model startup, passage encoding and all three rankings. The local checkpoint came from E5 revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`. Its Safetensors SHA-256 is recorded in the JSON. Retrieval metrics use the published judgments, which leave some documents unjudged. [Full result](reports/scifact.json) · [BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact)

## PersianQuAD

The published test file contains 35 contexts and 1,000 answerable question-context annotations. Grouping identical question text gives 997 queries. Each query searches those 35 contexts. Its positive labels come from the paragraphs paired with the question by the dataset authors.

| Method | Recall@5 | nDCG@10 | MRR@10 |
|---|---:|---:|---:|
| BM25 | 0.9900 | 0.9780 | 0.9726 |
| E5 | 0.9900 | 0.9708 | 0.9631 |
| BM25 + E5, RRF | 0.9960 | 0.9819 | 0.9768 |

BM25 scored higher than E5 alone on nDCG and MRR. RRF scored highest across the three metrics. This small corpus measures retrieval of the annotated context, with question answering and open-domain search outside the test scope. [Full result](reports/persianquad.json) · [Publisher](https://github.com/BigData-IsfahanUni/PersianQuAD)

To reproduce the Persian run, download the pinned [test.json](https://raw.githubusercontent.com/BigData-IsfahanUni/PersianQuAD/37f5c158ea4c3c308e630f3038fcfc331f58c6e3/Dataset/test.json) and run this command after installing the dense extra.

```text
python scripts/evaluate_persianquad.py path/to/test.json --output persianquad-result.json
```

## Service and package checks

- Four test cases passed against the published data. The HTTP cases started Uvicorn and exercised missing keys, mode reporting, verbatim offsets, role and tenant separation, refused imports and corpus replacement. The public and reader identities received zero results from the restricted collection. The manager retained access. An import without a policy and imports outside the allowed path were rejected.
- The real E5 service returned all 20 requested results in dense and hybrid modes on the 48-record subset. The first dense request took 22.866 seconds, including passage encoding. The next hybrid request on the cached view took 0.592 seconds. These are individual requests on a CPU shared with the benchmark.
- Twelve repeated warm BM25 HTTP requests on the full corpus returned the same response. Median latency was 8.317 ms and p95 was 9.593 ms. This is a local repeated-query measurement. [Raw timings](reports/bm25-http.json)
- The built 0.2.0 wheel installed into an isolated directory. Its BM25 entry point ran all 300 queries with the core dependencies and NumPy absent. [Installed-wheel result](reports/installed-bm25.json)
- Ruff lint and formatting checks passed. Local verification covered Windows and Python 3.12. Python 3.11 and Docker were checked later in GitHub Actions as recorded above. Ollama generation remains unverified.
