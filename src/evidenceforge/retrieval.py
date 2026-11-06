from __future__ import annotations

import json
import re
from collections import Counter, OrderedDict
from threading import Lock

from .acl import prefilter
from .models import Citation, Document, Mode, Principal, SearchResponse
from .sparse import BM25Index, tokenize

CANDIDATE_DEPTH = 100
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def rrf(
    rankings: list[list[tuple[str, float]]], k: int = 60, top_k: int = 10
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def evidence_span(text: str, query: str, limit: int = 280) -> tuple[int, int]:
    terms = set(tokenize(query))
    if limit > 280:
        matches = [
            (m.start(), m.group().casefold())
            for m in _TOKEN.finditer(text)
            if m.group().casefold() in terms
        ]
        frequencies = Counter(term for _, term in matches)
        weights = {term: 1 / count for term, count in frequencies.items()}
        starts = sorted(
            {0}
            | {min(max(0, offset - limit // 4), max(0, len(text) - limit)) for offset, _ in matches}
        )
        left = right = 0
        counts = Counter()
        coverage = 0.0
        best, best_score = 0, -1.0
        for start in starts:
            while right < len(matches) and matches[right][0] < start + limit:
                term = matches[right][1]
                if not counts[term]:
                    coverage += weights[term]
                counts[term] += 1
                right += 1
            while left < right and matches[left][0] < start:
                term = matches[left][1]
                counts[term] -= 1
                if not counts[term]:
                    coverage -= weights[term]
                left += 1
            if coverage > best_score + 1e-12:
                best, best_score = start, coverage
        return best, min(len(text), best + limit)
    hit = next(
        (match.start() for match in _TOKEN.finditer(text) if match.group().casefold() in terms), 0
    )
    start = min(max(0, hit - limit // 4), max(0, len(text) - limit))
    return start, min(len(text), start + limit)


class SearchIndex:
    """Indexes for one authorized document view."""

    def __init__(self, docs: tuple[Document, ...], encoder=None):
        self.docs = docs
        self.by_id = {doc.doc_id: doc for doc in docs}
        self.encoder = encoder
        self._sparse: BM25Index | None = None
        self._vectors = None
        self._lock = Lock()

    def sparse_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        with self._lock:
            if self._sparse is None:
                self._sparse = BM25Index(self.docs)
            index = self._sparse
        return index.search(query, top_k)

    def dense_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self.encoder is None or not self.docs:
            return []
        with self._lock:
            if self._vectors is None:
                self._vectors = self.encoder.encode_passages(self.docs)
            vectors = self._vectors
        scores = vectors @ self.encoder.encode_query(query)
        pairs = [(doc.doc_id, float(score)) for doc, score in zip(self.docs, scores, strict=True)]
        return sorted(pairs, key=lambda item: (-item[1], item[0]))[:top_k]


class SearchService:
    def __init__(self, docs: list[Document], dense=None, generator=None, store=None):
        self.dense = dense
        self.generator = generator
        self._lock = Lock()
        self._views: OrderedDict[Principal, SearchIndex] = OrderedDict()
        self._docs = tuple(docs)
        self.store = store
        self.revision = -1 if store else 0
        if store:
            if docs:
                store.replace(docs, initialize=True)
            self.refresh()

    @property
    def docs(self) -> tuple[Document, ...]:
        return self._docs

    def replace_corpus(self, docs: list[Document]) -> None:
        if self.store:
            self.store.replace(docs)
            self.refresh()
            return
        with self._lock:
            self._docs = tuple(docs)
            self.revision += 1
            self._views.clear()

    def refresh(self) -> None:
        if self.store:
            revision, docs = self.store.snapshot(self.revision)
            with self._lock:
                if docs is not None and revision > self.revision:
                    self._docs = tuple(docs)
                    self.revision = revision
                    self._views.clear()

    def _view(self, principal: Principal) -> tuple[SearchIndex, int]:
        with self._lock:
            index = self._views.get(principal)
            if index is None:
                index = SearchIndex(tuple(prefilter(self._docs, principal)), self.dense)
                self._views[principal] = index
                if len(self._views) > 8:
                    self._views.popitem(last=False)
            else:
                self._views.move_to_end(principal)
            return index, self.revision

    def query(
        self,
        query: str,
        principal: Principal,
        mode: Mode = "hybrid",
        top_k: int = 5,
        generate: bool = False,
    ) -> SearchResponse:
        effective_mode = "bm25" if mode == "hybrid" and self.dense is None else mode
        self.refresh()
        index, revision = self._view(principal)
        if not index.docs:
            return SearchResponse("I don't know.", (), effective_mode, "no_evidence", revision)

        depth = max(CANDIDATE_DEPTH, top_k)
        sparse = index.sparse_search(query, depth) if effective_mode in ("bm25", "hybrid") else []
        dense = index.dense_search(query, depth) if effective_mode in ("dense", "hybrid") else []
        if effective_mode == "hybrid":
            ranked = rrf([sparse, dense], top_k=top_k)
        else:
            ranked = (dense if effective_mode == "dense" else sparse)[:top_k]

        citations = []
        for doc_id, score in ranked:
            doc = index.by_id[doc_id]
            start, end = evidence_span(doc.text, query, 800 if generate else 280)
            citations.append(Citation(doc_id, doc.title, doc.text[start:end], score, start, end))
        sources = tuple(citations)
        if not sources:
            return SearchResponse("I don't know.", (), effective_mode, "no_evidence", revision)
        if generate and self.generator is not None:
            namespace = json.dumps([principal.tenant_id, sorted(principal.roles), revision])
            result = self.generator.generate(query, sources, namespace)
            return SearchResponse(
                result.answer,
                sources,
                effective_mode,
                result.status,
                revision,
                {
                    "model": result.model,
                    "evidence": list(result.evidence),
                    "policy": "verbatim_source_selection",
                    "cached": result.cached,
                    "requires_review": True,
                },
            )
        return SearchResponse(
            "I don't know.", sources, effective_mode, "evidence_candidates", revision
        )
