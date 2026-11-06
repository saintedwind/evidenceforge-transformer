"""Rank the published PersianQuAD test contexts for their existing questions."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from evidenceforge.benchmark import metrics
from evidenceforge.dense import E5Encoder
from evidenceforge.models import Document
from evidenceforge.retrieval import SearchIndex, rrf

SOURCE = (
    "https://raw.githubusercontent.com/BigData-IsfahanUni/PersianQuAD/"
    "37f5c158ea4c3c308e630f3038fcfc331f58c6e3/Dataset/test.json"
)
SHA256 = "b9f6236506b59b5b0a5c8606392fa7bda3d1f1610478b471bf75e874e765b26d"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_json", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.test_json.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("use the pinned PersianQuAD test.json file")
    source = json.loads(raw)
    docs, qrels = [], {}
    annotations = 0
    for article_number, article in enumerate(source["data"]):
        for paragraph_number, paragraph in enumerate(article["paragraphs"]):
            doc_id = f"{article_number:03d}/{paragraph_number:03d}"
            docs.append(Document(doc_id, article["title"], paragraph["context"]))
            for row in paragraph["qas"]:
                if not row["is_impossible"] and row["answers"]:
                    qrels.setdefault(row["question"], {})[doc_id] = 1
                    annotations += 1

    start = time.perf_counter()
    index = SearchIndex(tuple(docs), E5Encoder(args.model_path, args.threads))
    sparse_ranks, dense_ranks, hybrid_ranks = {}, {}, {}
    for question in qrels:
        sparse = index.sparse_search(question, len(docs))
        dense = index.dense_search(question, len(docs))
        sparse_ranks[question] = [doc_id for doc_id, _ in sparse]
        dense_ranks[question] = [doc_id for doc_id, _ in dense]
        hybrid_ranks[question] = [doc_id for doc_id, _ in rrf([sparse, dense], top_k=len(docs))]

    result = {
        "source": SOURCE,
        "source_sha256": SHA256,
        "published_contexts": len(docs),
        "published_answerable_annotations": annotations,
        "evaluated_unique_questions": len(qrels),
        "seconds": round(time.perf_counter() - start, 3),
        "scope": "Find annotated source paragraphs within the 35 published test contexts.",
        "bm25": metrics(sparse_ranks, qrels),
        "dense": metrics(dense_ranks, qrels),
        "hybrid_rrf": metrics(hybrid_ranks, qrels),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
