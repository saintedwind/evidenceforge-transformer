from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import time
from importlib.metadata import version
from pathlib import Path

from .dense import MODEL_ID, MODEL_REVISION, E5Encoder
from .ingest import load_beir_corpus
from .retrieval import CANDIDATE_DEPTH, SearchIndex, rrf


def metrics(ranked: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    """Macro-averaged retrieval scores on queries with positive judgments."""
    recalls, ndcgs, mrrs = [], [], []
    for query_id, judgments in qrels.items():
        relevant = {doc_id for doc_id, grade in judgments.items() if grade > 0}
        if not relevant:
            continue
        docs = list(dict.fromkeys(ranked.get(query_id, [])))
        recalls.append(len(set(docs[:5]) & relevant) / len(relevant))
        dcg = sum(
            (2 ** judgments.get(doc_id, 0) - 1) / math.log2(rank + 2)
            for rank, doc_id in enumerate(docs[:10])
        )
        ideal = sorted(judgments.values(), reverse=True)[:10]
        idcg = sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(ideal))
        ndcgs.append(dcg / idcg)
        mrrs.append(
            next((1 / rank for rank, doc_id in enumerate(docs[:10], 1) if doc_id in relevant), 0.0)
        )
    if not recalls:
        raise ValueError("evaluation contains no queries with positive judgments")
    return {
        "recall@5": sum(recalls) / len(recalls),
        "ndcg@10": sum(ndcgs) / len(ndcgs),
        "mrr@10": sum(mrrs) / len(mrrs),
    }


def read_queries(data: Path) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    with (data / "queries.jsonl").open(encoding="utf-8") as source:
        queries = {row["_id"]: row["text"] for row in map(json.loads, source)}
    qrels: dict[str, dict[str, int]] = {}
    with (data / "qrels/test.tsv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    if not qrels or set(qrels) - queries.keys():
        raise ValueError("test judgments are empty or reference unknown queries")
    return queries, qrels


def run_benchmark(
    data_dir: str | Path,
    output: str | Path,
    dense: bool = False,
    model_path: str | Path | None = None,
    threads: int = 6,
) -> dict:
    start = time.perf_counter()
    data = Path(data_dir)
    docs = tuple(load_beir_corpus(data / "corpus.jsonl"))
    queries, qrels = read_queries(data)
    doc_ids = {doc.doc_id for doc in docs}
    if any(judgments.keys() - doc_ids for judgments in qrels.values()):
        raise ValueError("test judgments reference documents outside the corpus")

    encoder = E5Encoder(model_path, threads) if dense else None
    index = SearchIndex(docs, encoder)
    sparse_ranks, dense_ranks, hybrid_ranks = {}, {}, {}
    for query_id in qrels:
        query = queries[query_id]
        sparse = index.sparse_search(query, CANDIDATE_DEPTH)
        sparse_ranks[query_id] = [doc_id for doc_id, _ in sparse]
        if encoder is not None:
            vectors = index.dense_search(query, CANDIDATE_DEPTH)
            dense_ranks[query_id] = [doc_id for doc_id, _ in vectors]
            fused = rrf([sparse, vectors], top_k=CANDIDATE_DEPTH)
            hybrid_ranks[query_id] = [doc_id for doc_id, _ in fused]

    result = {
        "corpus_documents": len(docs),
        "test_qrels": sum(map(len, qrels.values())),
        "evaluated_queries": sum(any(grade > 0 for grade in q.values()) for q in qrels.values()),
        "candidate_depth": CANDIDATE_DEPTH,
        "runtime_seconds": round(time.perf_counter() - start, 3),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "data_sha256": {
            name: _sha256(data / name)
            for name in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")
        },
        "bm25": metrics(sparse_ranks, qrels),
    }
    if encoder is not None:
        result["dense"] = metrics(dense_ranks, qrels)
        result["hybrid_rrf"] = metrics(hybrid_ranks, qrels)
        result["encoder"] = {
            "model": MODEL_ID if model_path is None else "local checkpoint",
            "revision": MODEL_REVISION if model_path is None else None,
            "weights_sha256": _sha256(Path(model_path) / "model.safetensors")
            if model_path
            else None,
            "device": "cpu",
            "threads": threads,
            "versions": {name: version(name) for name in ("numpy", "torch", "transformers")},
        }

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()
