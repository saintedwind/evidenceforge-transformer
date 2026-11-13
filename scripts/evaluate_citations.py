"""Compare the released and local citation windows over real HTTP on PersianQuAD."""

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from evaluate_persianquad import SHA256, SOURCE

from evidenceforge.ingest import write_jsonl
from evidenceforge.models import Document

ROOT = Path(__file__).resolve().parents[1]
WHEEL_URL = (
    "https://github.com/saintedwind/evidenceforge-transformer/releases/download/v0.3.0/"
    "evidenceforge-0.3.0-py3-none-any.whl"
)
WHEEL_SHA256 = "1cf052b102fa1cc73e7729bc0d873c88ed787b7fdb6fc0ca1dcb1e91008f122f"
TOP_K = 5


def checked_bytes(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch: {path.name}")
    return raw


def load_cases(source):
    docs, cases = [], []
    invalid = 0
    for article_no, article in enumerate(json.loads(source)["data"]):
        for paragraph_no, paragraph in enumerate(article["paragraphs"]):
            doc = Document(
                f"{article_no:03d}/{paragraph_no:03d}", article["title"], paragraph["context"]
            )
            docs.append(doc)
            for item in paragraph["qas"]:
                if item["is_impossible"] or not item["answers"]:
                    continue
                spans = []
                for answer in item["answers"]:
                    start, text = answer["answer_start"], answer["text"]
                    end = start + len(text)
                    if text and start >= 0 and doc.text[start:end] == text:
                        spans.append((start, end))
                    else:
                        invalid += 1
                if not spans:
                    raise ValueError(f"No usable answer offsets for {item['id']}")
                cases.append((item["id"], item["question"], doc.doc_id, spans))
    return docs, cases, invalid


def run_http(package, corpus, cases, originals, log_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(filter(None, (str(package), os.getenv("PYTHONPATH")))),
        EVIDENCEFORGE_CORPUS=str(corpus),
        EVIDENCEFORGE_API_KEYS="citation-eval=public:reader",
        EVIDENCEFORGE_REQUESTS_PER_MINUTE=str(len(cases) + 1),
        EVIDENCEFORGE_DENSE="0",
        EVIDENCEFORGE_LLM_MODEL="",
        OLLAMA_MODEL="",
        EVIDENCEFORGE_STATE="",
        EVIDENCEFORGE_IMPORT_ROOT="",
    )
    rows, timings, checked = [], [], 0
    with (
        log_path.open("wb") as log,
        subprocess.Popen(
            [sys.executable, "-m", "evidenceforge.cli", "serve", "--port", str(port)],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=log,
        ) as process,
    ):
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}",
                headers={"x-api-key": "citation-eval"},
                timeout=20,
                trust_env=False,
            ) as client:
                for _ in range(150):
                    if process.poll() is not None:
                        raise RuntimeError(f"API stopped; see {log_path}")
                    try:
                        if client.get("/ready", timeout=1).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.2)
                else:
                    raise RuntimeError(f"API did not become ready; see {log_path}")

                def ask(question):
                    response = client.post(
                        "/query", json={"query": question, "mode": "bm25", "top_k": TOP_K}
                    )
                    return response.raise_for_status().json()["citations"]

                ask(cases[0][1])  # Warm the lazy index before measuring request times.
                for case_id, question, gold_id, spans in cases:
                    start = time.perf_counter()
                    citations = ask(question)
                    timings.append((time.perf_counter() - start) * 1000)
                    rank, contains_answer = None, False
                    ranking = []
                    for position, citation in enumerate(citations, 1):
                        doc_id = citation["doc_id"]
                        begin, end = citation["start"], citation["end"]
                        text = originals[doc_id]
                        if not (
                            0 <= begin <= end <= len(text)
                            and end - begin == min(280, len(text))
                            and text[begin:end] == citation["snippet"]
                        ):
                            raise ValueError(f"Invalid original offsets: {case_id}, {doc_id}")
                        checked += 1
                        ranking.append((doc_id, citation["score"]))
                        if doc_id == gold_id:
                            rank = position
                            contains_answer = any(begin <= a and b <= end for a, b in spans)
                    rows.append((case_id, rank, contains_answer, ranking))
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
    return rows, {
        "gold_context_at_1": sum(rank == 1 for _, rank, _, _ in rows),
        "gold_context_at_5": sum(rank is not None for _, rank, _, _ in rows),
        "answer_span_at_1": sum(rank == 1 and covered for _, rank, covered, _ in rows),
        "answer_span_at_5": sum(covered for _, _, covered, _ in rows),
        "verbatim_citations_checked": checked,
        "median_http_ms": round(statistics.median(timings), 3),
        "p95_http_ms": round(sorted(timings)[math.ceil(len(timings) * 0.95) - 1], 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_json", type=Path)
    parser.add_argument("--baseline-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--per-question", action="store_true", help="include publisher IDs and outcomes"
    )
    args = parser.parse_args()
    checked_bytes(args.baseline_wheel, WHEEL_SHA256)
    docs, cases, invalid = load_cases(checked_bytes(args.test_json, SHA256))
    originals = {doc.doc_id: doc.text for doc in docs}
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="citation-eval-", dir=output.parent) as temporary:
        work = Path(temporary)
        corpus = work / "corpus.jsonl"
        write_jsonl(docs, corpus)
        before, baseline = run_http(
            args.baseline_wheel.resolve(),
            corpus,
            cases,
            originals,
            output.with_suffix(".before.service.log"),
        )
        after, current = run_http(
            ROOT / "src", corpus, cases, originals, output.with_suffix(".after.service.log")
        )
    outcomes = []
    for (case_id, rank, old, ranking), (new_id, new_rank, new, new_ranking) in zip(
        before, after, strict=True
    ):
        if (case_id, rank, ranking) != (new_id, new_rank, new_ranking):
            raise ValueError(f"Document IDs, order or scores changed: {case_id}")
        outcomes.append({"id": case_id, "gold_rank": rank, "before": old, "after": new})
    report = {
        "source": SOURCE,
        "source_sha256": SHA256,
        "baseline_wheel": WHEEL_URL,
        "baseline_sha256": WHEEL_SHA256,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "publisher_contexts": len(docs),
        "answerable_question_ids": len(cases),
        "invalid_publisher_offsets_excluded": invalid,
        "mode": "bm25",
        "top_k": TOP_K,
        "baseline": baseline,
        "current": current,
        "same_rankings_and_scores": len(outcomes),
        "gained_answer_span": sum(not row["before"] and row["after"] for row in outcomes),
        "lost_answer_span": sum(row["before"] and not row["after"] for row in outcomes),
    }
    payload = report | {"per_question": outcomes} if args.per_question else report
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
