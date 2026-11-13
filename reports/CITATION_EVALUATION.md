# Citation windows on PersianQuAD

The released 0.3.0 service starts short citations near the first query term that appears in the document. Common words can anchor that window far from the supporting passage. The new selector looks for a 280-character window that covers distinct complete query tokens, weighting each by the inverse of its frequency in the document. The excerpt is still an exact slice of the original text. Document ranking, access controls and the separate 800-character path for local generation are unchanged.

## Comparison over HTTP

Each variant ran as a separate Uvicorn process on loopback, with one warm BM25 index over the same 35 published PersianQuAD test contexts and generation disabled. Both received all 1,000 answerable question IDs from the [pinned PersianQuAD test file](https://raw.githubusercontent.com/BigData-IsfahanUni/PersianQuAD/37f5c158ea4c3c308e630f3038fcfc331f58c6e3/Dataset/test.json). The request contains only the publisher's question text. Scoring requires the publisher's annotated context and at least one complete reference answer at its verified character offsets. One publisher annotation had a text/offset mismatch and was excluded; that question retained other valid annotations. No questions were excluded or rewritten.

| Measure | Released 0.3.0 | Weighted citation window |
|---|---:|---:|
| Annotated context at rank 1 | 956 / 1,000 | 956 / 1,000 |
| Annotated context in top 5 | 990 / 1,000 | 990 / 1,000 |
| Complete answer span at rank 1 | 396 / 1,000 | 717 / 1,000 |
| Complete answer span in top 5 | 410 / 1,000 | 736 / 1,000 |
| Median HTTP round trip | 3.105 ms | 3.714 ms |
| p95 HTTP round trip | 4.415 ms | 8.454 ms |

The first five document IDs, scores and order matched for every request. All 9,968 returned citations across the two variants matched the original text and length contract. At five results, 382 questions gained a visible answer span and 56 lost one, a net gain of 326. A separate execution reproduced all quality counts, with p95 of 5.636 ms for 0.3.0 and 7.872 ms for the new window. These are sequential loopback timings on one Windows host; they do not establish a production latency bound.

This is a small, previously used corpus and the full comparison is a development evaluation. It does not measure generalization to other corpora, unanswerable queries, whether a user interprets the excerpt correctly, or improvements in model generation. The fixed window can still cut a sentence or omit an answer when the question and evidence use different wording. The [machine-readable summary](persianquad-citation-windows.json) records the input and baseline hashes.

The SciFact regression suite also checks a published claim (query `49`, positively judged abstract `5953485`). The released window omits the complete phrase describing the interaction; the new HTTP test requires that phrase inside the citation and checks its original offsets. The 300-query SciFact BM25 test continues to guard the document-level retrieval metrics. All ten project tests passed on Windows with Python 3.12.14, including real E5 inference on 48 SciFact documents and SQLite persistence and access checks. Ruff lint and formatting checks passed locally.

## Reproduce

Install the local package, then download the pinned test file and the [published wheel](https://github.com/saintedwind/evidenceforge-transformer/releases/download/v0.3.0/evidenceforge-0.3.0-py3-none-any.whl). The evaluator verifies both SHA-256 digests before starting the servers. It writes aggregate metrics by default; add `--per-question` to retain publisher IDs and paired results without copying question or answer text into the report.

```text
python -m pip install -e ".[data,test]"
python scripts/evaluate_citations.py test.json --baseline-wheel evidenceforge-0.3.0-py3-none-any.whl --output citation-windows.json
evidenceforge fetch --data data/scifact
python -m pytest -q
```
