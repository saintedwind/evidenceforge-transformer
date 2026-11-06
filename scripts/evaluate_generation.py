"""Exercise a real local model using the pinned, human-annotated PersianQuAD split."""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from evaluate_persianquad import SHA256, SOURCE

from evidenceforge.generation import SourceGenerator
from evidenceforge.ingest import write_jsonl
from evidenceforge.models import Citation, Document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--cache-check",
        action="store_true",
        help="check a real response across an API restart and ACL replacement",
    )
    args = parser.parse_args()
    raw = args.test_json.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Use the pinned publisher's test.json")
    if not 3 <= args.limit <= 100:
        raise ValueError("limit must be between 3 and 100")
    if not os.getenv("EVIDENCEFORGE_LLM_MODEL"):
        raise ValueError("Configure an actual local model before evaluation")
    docs, questions = [], []
    for article_number, article in enumerate(json.loads(raw)["data"]):
        for paragraph_number, paragraph in enumerate(article["paragraphs"]):
            doc_id = f"{article_number:03d}/{paragraph_number:03d}"
            docs.append(Document(doc_id, article["title"], paragraph["context"]))
            for item in paragraph["qas"]:
                if not item["is_impossible"] and item["answers"]:
                    questions.append(
                        {
                            "id": item["id"],
                            "question": item["question"],
                            "doc_id": doc_id,
                            "answers": item["answers"],
                        }
                    )
    # Fixed, evenly spaced publisher order; no answer-based case selection.
    chosen = [questions[i * len(questions) // args.limit] for i in range(args.limit)]
    if args.cache_check:
        chosen = chosen[:1]
    by_id = {doc.doc_id: doc for doc in docs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="generation-", dir=args.output.parent) as temporary:
        work = Path(temporary)
        corpus = work / "corpus.jsonl"
        write_jsonl(docs, corpus)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        env = dict(
            os.environ,
            EVIDENCEFORGE_CORPUS=str(corpus),
            EVIDENCEFORGE_IMPORT_ROOT=str(work),
            EVIDENCEFORGE_STATE=str(work / "state.sqlite3"),
            EVIDENCEFORGE_API_KEYS="evaluation-key=public:reader",
            EVIDENCEFORGE_ADMIN_TOKEN="evaluation-admin",
            EVIDENCEFORGE_DENSE="0",
            EVIDENCEFORGE_REQUESTS_PER_MINUTE="1000",
        )
        log = args.output.with_suffix(".service.log").open("wb")
        process = subprocess.Popen(
            [sys.executable, "-m", "evidenceforge.cli", "serve", "--port", str(port)],
            env=env,
            stdout=log,
            stderr=log,
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}",
                timeout=190,
                trust_env=False,
                headers={"x-api-key": "evaluation-key"},
            ) as client:
                for _ in range(150):
                    if process.poll() is not None:
                        raise RuntimeError("Evaluation service stopped; inspect its log")
                    try:
                        if client.get("/ready", timeout=1).status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.2)
                else:
                    raise RuntimeError("Evaluation service did not become ready")

                def ask(item):
                    start = time.perf_counter()
                    response = client.post(
                        "/query",
                        json={
                            "query": item["question"],
                            "mode": "bm25",
                            "top_k": 3,
                            "generate": True,
                        },
                    )
                    value = response.json()
                    evidence = value.get("generation", {}).get("evidence", [])
                    for span in evidence:
                        assert (
                            by_id[span["doc_id"]].text[span["start"] : span["end"]] == span["quote"]
                        )
                    gold = [answer["text"] for answer in item["answers"]]
                    result = {
                        "question_id": item["id"],
                        "question": item["question"],
                        "gold_context_id": item["doc_id"],
                        "published_answers": gold,
                        "http_status": response.status_code,
                        "seconds": round(time.perf_counter() - start, 3),
                        "gold_context_retrieved": any(
                            c["doc_id"] == item["doc_id"] for c in value.get("citations", [])
                        ),
                        "selected_gold_context": any(
                            c["doc_id"] == item["doc_id"] for c in evidence
                        ),
                        "gold_answer_in_selected_quote": any(
                            a in span["quote"] for a in gold for span in evidence
                        ),
                        "response": value,
                    }
                    return result

                cases = []
                for item in chosen:
                    case = ask(item)
                    cases.append(case)
                    print(
                        json.dumps(
                            {
                                key: case[key]
                                for key in (
                                    "question_id",
                                    "seconds",
                                    "gold_answer_in_selected_quote",
                                )
                            }
                        ),
                        flush=True,
                    )
                if args.cache_check:
                    assert cases[0]["response"]["status"] == "grounded_extract"
                    assert cases[0]["response"]["generation"]["cached"] is False
                    process.terminate()
                    process.wait(timeout=20)
                    process = subprocess.Popen(
                        [sys.executable, "-m", "evidenceforge.cli", "serve", "--port", str(port)],
                        env=env,
                        stdout=log,
                        stderr=log,
                    )
                    for _ in range(150):
                        try:
                            if client.get("/ready", timeout=1).status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        time.sleep(0.2)
                    else:
                        raise RuntimeError("Restart did not become ready")
                    repeated = ask(chosen[0])
                    assert repeated["response"]["generation"]["cached"] is True
                    assert repeated["response"]["answer"] == cases[0]["response"]["answer"]
                    restricted = work / "restricted.jsonl"
                    write_jsonl(
                        [
                            replace(doc, tenant_id="review", allowed_roles=frozenset({"manager"}))
                            for doc in docs
                        ],
                        restricted,
                    )
                    assert (
                        client.post(
                            "/ingest",
                            json={"path": restricted.name},
                            headers={"x-admin-token": "evaluation-admin"},
                        ).status_code
                        == 200
                    )
                    revoked = ask(chosen[0])
                    assert revoked["response"]["status"] == "no_evidence"
                    report = {
                        "source": SOURCE,
                        "source_sha256": SHA256,
                        "model": os.environ["EVIDENCEFORGE_LLM_MODEL"],
                        "question_id": chosen[0]["id"],
                        "application_cache_miss_seconds": cases[0]["seconds"],
                        "persisted_cache_hit_seconds": repeated["seconds"],
                        "cache_hit_after_api_restart": True,
                        "same_answer": True,
                        "revoked_collection_status": revoked["response"]["status"],
                        "scope": "One real published question; two successive API processes sharing SQLite. The model was already loaded and its own prompt cache could be warm. No mocked model response or generated source text.",
                    }
                    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                    print(json.dumps(report), flush=True)
                    return
                removed = []
                subset = work / "withheld.jsonl"
                for item in chosen[:3]:
                    remaining = [doc for doc in docs if doc.doc_id != item["doc_id"]]
                    write_jsonl(remaining, subset)
                    response = client.post(
                        "/ingest",
                        json={"path": subset.name},
                        headers={"x-admin-token": "evaluation-admin"},
                    )
                    assert response.status_code == 200
                    case = ask(item)
                    case["gold_answer_occurs_elsewhere"] = any(
                        a["text"] in d.text for a in item["answers"] for d in remaining
                    )
                    removed.append(case)
                assert (
                    client.post(
                        "/ingest",
                        json={"path": corpus.name},
                        headers={"x-admin-token": "evaluation-admin"},
                    ).status_code
                    == 200
                )
                with ThreadPoolExecutor(max_workers=3) as workers:
                    simultaneous = list(workers.map(ask, [chosen[0]] * 3))
                restricted = {
                    "path": corpus.name,
                    "policy": {"tenant_id": "review", "allowed_roles": ["manager"]},
                }
                # This collection carries explicit public ACLs, so write a policy-free copy
                # of those same published rows before applying the access-test policy.
                plain = work / "plain.jsonl"
                with plain.open("w", encoding="utf-8", newline="\n") as stream:
                    for doc in docs:
                        stream.write(
                            json.dumps(
                                {"_id": doc.doc_id, "title": doc.title, "text": doc.text},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                restricted["path"] = plain.name
                assert (
                    client.post(
                        "/ingest", json=restricted, headers={"x-admin-token": "evaluation-admin"}
                    ).status_code
                    == 200
                )
                denied = ask(chosen[0])
                assert (
                    denied["response"]["status"] == "no_evidence"
                    and not denied["response"]["citations"]
                )

                first = cases[0]["response"]["citations"]
                citations = tuple(Citation(**value) for value in first)
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    offline_port = listener.getsockname()[1]
                offline = SourceGenerator(
                    url=f"http://127.0.0.1:{offline_port}/v1/chat/completions"
                )
                offline_status = offline.generate(chosen[0]["question"], citations).status
                assert offline_status == "generation_unavailable"
                report = {
                    "source": SOURCE,
                    "source_sha256": SHA256,
                    "model": os.environ["EVIDENCEFORGE_LLM_MODEL"],
                    "selection": "Evenly spaced publisher annotation order; gold answers used only for scoring.",
                    "scope": "Small integration sample, not a held-out model quality benchmark. Source-removal cases are controlled stress tests, not human-labelled unanswerable questions.",
                    "published_contexts": len(docs),
                    "cases": cases,
                    "summary": {
                        "questions": len(cases),
                        "statuses": dict(Counter(c["response"]["status"] for c in cases)),
                        "gold_context_retrieved": sum(c["gold_context_retrieved"] for c in cases),
                        "selected_gold_context": sum(c["selected_gold_context"] for c in cases),
                        "gold_answer_in_selected_quote": sum(
                            c["gold_answer_in_selected_quote"] for c in cases
                        ),
                    },
                    "source_removal_cases": removed,
                    "simultaneous_request_statuses": [
                        c["response"]["status"] for c in simultaneous
                    ],
                    "restricted_collection_status": denied["response"]["status"],
                    "offline_model_status": offline_status,
                }
                args.output.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
        finally:
            process.terminate()
            process.wait(timeout=20)
            log.close()


if __name__ == "__main__":
    main()
