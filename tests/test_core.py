import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from evidenceforge.benchmark import read_queries, run_benchmark
from evidenceforge.ingest import _document, load_beir_corpus

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = {"x-api-key": "review-public"}
READER = {"x-api-key": "review-reader"}
MANAGER = {"x-api-key": "review-manager"}
ADMIN = {"x-admin-token": "review-admin"}


@pytest.fixture(scope="session")
def data():
    path = Path(os.getenv("EVIDENCEFORGE_TEST_DATA", ROOT / "data/scifact")).resolve()
    if not (path / "corpus.jsonl").is_file():
        pytest.fail("Set EVIDENCEFORGE_TEST_DATA or fetch the official SciFact dataset first.")
    return path


@pytest.fixture(scope="session")
def official_query(data):
    queries, qrels = read_queries(data)
    return queries[next(iter(qrels))]


@contextmanager
def live_service(
    corpus: Path, model: str | None = None, *, state: Path | None = None, rate: int = 120
):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        EVIDENCEFORGE_CORPUS=str(corpus),
        EVIDENCEFORGE_IMPORT_ROOT=str(corpus.parent),
        EVIDENCEFORGE_API_KEYS=(
            "review-public=public:reader,review-reader=review:reader,review-manager=review:manager"
        ),
        EVIDENCEFORGE_ADMIN_TOKEN="review-admin",
        EVIDENCEFORGE_DENSE="1" if model else "0",
        EVIDENCEFORGE_MODEL_PATH=model or "",
        OLLAMA_MODEL="",
        EVIDENCEFORGE_LLM_MODEL="",
        EVIDENCEFORGE_STATE=str(state) if state else "",
        EVIDENCEFORGE_REQUESTS_PER_MINUTE=str(rate),
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "evidenceforge.cli", "serve", "--port", str(port)],
        env=env,
        cwd=ROOT,
    )
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}", timeout=90, trust_env=False
        ) as client:
            for _ in range(300):
                if process.poll() is not None:
                    pytest.fail(f"Uvicorn exited at startup with status {process.returncode}")
                try:
                    if client.get("/health", timeout=1).status_code == 200:
                        break
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                time.sleep(0.2)
            else:
                pytest.fail("Uvicorn did not become ready")
            yield client
    finally:
        process.terminate()
        process.wait(timeout=15)


def test_official_bm25_benchmark(data, tmp_path):
    actual = run_benchmark(data, tmp_path / "benchmark.json")
    reference = json.loads((ROOT / "benchmark_scifact_bm25.json").read_text(encoding="utf-8"))
    assert actual["corpus_documents"] == 5183
    assert actual["evaluated_queries"] == 300
    assert actual["test_qrels"] == 339
    assert actual["bm25"] == pytest.approx(reference["bm25"], abs=1e-12)


def test_document_acl_uses_existing_source_record(data):
    with (data / "corpus.jsonl").open(encoding="utf-8") as source:
        row = json.loads(source.readline())
    scoped = _document(row | {"tenant_id": "review", "allowed_roles": ["manager"]}, None)
    assert scoped.doc_id == row["_id"]
    assert scoped.text == row["text"]
    assert scoped.tenant_id == "review"
    assert scoped.allowed_roles == {"manager"}
    with pytest.raises(ValueError, match="complete ACL"):
        _document(row | {"tenant_id": "review"}, None)


def test_live_bm25_and_acl_replacement(data, official_query):
    corpus = data / "corpus.jsonl"
    by_id = {doc.doc_id: doc for doc in load_beir_corpus(corpus)}
    payload = {"query": official_query, "mode": "hybrid"}
    with live_service(corpus) as client:
        assert client.post("/query", json=payload).status_code == 401
        assert (
            client.post("/query", json=payload, headers={"x-api-key": "unknown"}).status_code == 401
        )
        assert (
            client.post("/query", json=payload | {"mode": "dense"}, headers=PUBLIC).status_code
            == 503
        )

        before = client.post("/query", json=payload, headers=PUBLIC)
        assert before.status_code == 200
        evidence = before.json()
        assert evidence["requested_mode"] == "hybrid" and evidence["mode"] == "bm25"
        assert evidence["status"] == "evidence_candidates"
        assert evidence["citations"]
        for citation in evidence["citations"]:
            original = by_id[citation["doc_id"]].text
            assert original[citation["start"] : citation["end"]] == citation["snippet"]

        assert client.post("/ingest", json={"path": corpus.name}).status_code == 403
        restricted = {
            "path": corpus.name,
            "policy": {"tenant_id": "review", "allowed_roles": ["manager"]},
        }
        imported = client.post("/ingest", json=restricted, headers=ADMIN)
        assert imported.status_code == 200 and imported.json()["ingested"] == 5183
        for identity in (PUBLIC, READER):
            denied = client.post("/query", json=payload, headers=identity).json()
            assert denied["status"] == "no_evidence" and denied["citations"] == []
        manager = client.post("/query", json=payload, headers=MANAGER).json()
        assert manager["citations"] == evidence["citations"]

        for invalid in ("../corpus.jsonl", "qrels/test.tsv", "missing.jsonl"):
            rejected = client.post("/ingest", json={"path": invalid}, headers=ADMIN)
            assert rejected.status_code == 422
            assert rejected.json() == {"detail": "unable to import corpus"}
        assert client.post("/query", json=payload, headers=MANAGER).json() == manager
        assert client.post("/ingest", json={"path": corpus.name}, headers=ADMIN).status_code == 422
        assert client.post("/query", json=payload, headers=MANAGER).json() == manager
        restore = {"path": corpus.name, "policy": {"tenant_id": "public", "allowed_roles": []}}
        assert client.post("/ingest", json=restore, headers=ADMIN).status_code == 200
        assert client.post("/query", json=payload, headers=MANAGER).json()["citations"] == []
        restored = client.post("/query", json=payload, headers=PUBLIC).json()
        assert restored["citations"] == evidence["citations"]
        assert restored["corpus_revision"] > evidence["corpus_revision"]
        print("Official SciFact over HTTP: public/manager/reader checked before and after import.")


def test_live_dense_and_hybrid(data, official_query, tmp_path):
    model = os.getenv("EVIDENCEFORGE_TEST_MODEL")
    if model is None:
        pytest.skip("Set EVIDENCEFORGE_TEST_MODEL to a local E5 checkpoint to run inference.")
    # Keep the first 48 published records byte-for-byte for the live CPU check.
    corpus = tmp_path / "corpus.jsonl"
    with (data / "corpus.jsonl").open("rb") as source, corpus.open("wb") as destination:
        for _ in range(48):
            destination.write(source.readline())
    by_id = {doc.doc_id: doc for doc in load_beir_corpus(corpus)}
    assert len(by_id) == 48
    payload = {"query": official_query, "mode": "hybrid", "top_k": 20}
    with live_service(corpus, model) as client:
        assert client.get("/health").json()["dense"] is True
        # Restrict the original records before requesting any dense candidates.
        policy = {"tenant_id": "review", "allowed_roles": ["manager"]}
        response = client.post(
            "/ingest", json={"path": corpus.name, "policy": policy}, headers=ADMIN
        )
        assert response.status_code == 200 and response.json()["ingested"] == 48

        for identity in (PUBLIC, READER):
            denied = client.post("/query", json=payload, headers=identity).json()
            assert denied["status"] == "no_evidence" and not denied["citations"]

        start = time.perf_counter()
        dense = client.post("/query", json=payload | {"mode": "dense"}, headers=MANAGER)
        cold = time.perf_counter() - start
        start = time.perf_counter()
        hybrid = client.post("/query", json=payload, headers=MANAGER)
        warm = time.perf_counter() - start
        assert dense.status_code == hybrid.status_code == 200
        assert dense.json()["mode"] == "dense" and hybrid.json()["mode"] == "hybrid"
        assert len(dense.json()["citations"]) == len(hybrid.json()["citations"]) == 20
        assert all(c["doc_id"] in by_id for c in hybrid.json()["citations"])
        # Revoke the manager after its dense view has been populated.
        public_policy = {"tenant_id": "public", "allowed_roles": []}
        changed = client.post(
            "/ingest", json={"path": corpus.name, "policy": public_policy}, headers=ADMIN
        )
        assert changed.status_code == 200
        revoked = client.post("/query", json=payload, headers=MANAGER).json()
        assert revoked["status"] == "no_evidence" and revoked["citations"] == []
        public = client.post("/query", json=payload | {"mode": "bm25"}, headers=PUBLIC).json()
        assert public["citations"]
        print(
            f"Real E5 HTTP on 48 SciFact documents: cold dense {cold:.3f}s, warm hybrid {warm:.3f}s."
        )


def test_live_concurrent_queries(data):
    queries, qrels = read_queries(data)
    payloads = [{"query": queries[query_id], "mode": "bm25"} for query_id in list(qrels)[:12]]
    with live_service(data / "corpus.jsonl") as client:
        expected = [
            client.post("/query", json=payload, headers=PUBLIC).json() for payload in payloads
        ]

        def retrieve(payload):
            response = client.post("/query", json=payload, headers=PUBLIC)
            assert response.status_code == 200
            return response.json()

        with ThreadPoolExecutor(max_workers=4) as workers:
            actual = list(workers.map(retrieve, payloads))
        assert actual == expected
