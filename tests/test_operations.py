import json

import pytest
from test_core import ADMIN, MANAGER, PUBLIC, live_service
from test_core import data as data
from test_core import official_query as official_query

from evidenceforge.generation import SourceGenerator
from evidenceforge.ingest import load_beir_corpus
from evidenceforge.models import Citation
from evidenceforge.storage import CorpusStore


def test_generation_lease_shared_across_store_connections(data, tmp_path):
    path = tmp_path / "leases.sqlite3"
    first, second = CorpusStore(path), CorpusStore(path)
    first.replace(load_beir_corpus(data / "corpus.jsonl")[:48], initialize=True)
    assert first.acquire_generation("first-worker", 30)
    assert not second.acquire_generation("second-worker", 30)
    second.release_generation("second-worker")
    assert not second.acquire_generation("second-worker", 30)
    first.release_generation("first-worker")
    assert second.acquire_generation("second-worker", 30)
    second.release_generation("second-worker")


def test_live_durable_import_across_processes_and_restart(data, official_query, tmp_path):
    state = tmp_path / "shared.sqlite3"
    corpus = data / "corpus.jsonl"
    query = {"query": official_query, "mode": "bm25"}
    restricted = {
        "path": corpus.name,
        "policy": {"tenant_id": "review", "allowed_roles": ["manager"]},
    }
    with live_service(corpus, state=state) as first, live_service(corpus, state=state) as second:
        assert first.post("/query", json=query, headers=PUBLIC).json()["citations"]
        before = second.post("/query", json=query, headers=PUBLIC).json()
        committed = first.post("/ingest", json=restricted, headers=ADMIN)
        assert committed.status_code == 200
        for client in (first, second):
            denied = client.post("/query", json=query, headers=PUBLIC).json()
            assert denied["status"] == "no_evidence" and not denied["citations"]
            assert denied["corpus_revision"] > before["corpus_revision"]
            assert client.post("/query", json=query, headers=MANAGER).json()["citations"]
    with live_service(tmp_path / "missing-startup.jsonl", state=state) as restarted:
        assert restarted.get("/health").json()["documents"] == 5183
        assert restarted.get("/ready").status_code == 200
        assert (
            restarted.post("/query", json=query, headers=PUBLIC).json()["status"] == "no_evidence"
        )
        assert restarted.post("/query", json=query, headers=MANAGER).json()["citations"]


def test_live_body_limit_and_shared_request_quota(data, official_query, tmp_path):
    state = tmp_path / "limited.sqlite3"
    corpus = data / "corpus.jsonl"
    with (
        live_service(corpus, state=state, rate=2) as first,
        live_service(corpus, state=state, rate=2) as second,
    ):
        with corpus.open("rb") as source:
            large = source.read(16 * 1024 + 1)
        assert first.post("/query", content=large, headers=PUBLIC).status_code == 413
        responses = [
            client.post("/query", json={"query": official_query}, headers=PUBLIC)
            for client in (first, second, first, second, first)
        ]
        assert any(response.status_code == 429 for response in responses)
        assert all("x-request-id" in response.headers for response in responses)


def test_quote_validation_uses_unmodified_published_text(data):
    docs = load_beir_corpus(data / "corpus.jsonl")[:2]
    citations = tuple(
        Citation(d.doc_id, d.title, d.text[:280], 1, 0, min(len(d.text), 280)) for d in docs
    )
    quote = citations[0].snippet[:80]
    valid = {"doc_id": docs[0].doc_id, "quote": quote}
    result = SourceGenerator.validate(json.dumps(valid), citations)
    assert result.status == "grounded_extract"
    span = result.evidence[0]
    assert docs[0].text[span["start"] : span["end"]] == span["quote"]
    valid["doc_id"] = docs[1].doc_id
    assert SourceGenerator.validate(json.dumps(valid), citations).status == "generation_invalid"
    with pytest.raises(ValueError, match="loopback"):
        SourceGenerator(model="local", url="https://example.com/v1/chat/completions")
