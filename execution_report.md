# EvidenceForge evaluation

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
