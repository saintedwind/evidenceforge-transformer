from __future__ import annotations

import re
from collections import OrderedDict
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
    def __init__(self, docs: list[Document], dense=None, generator=None):
        self.dense = dense
        self.generator = generator
        self._lock = Lock()
        self._views: OrderedDict[Principal, SearchIndex] = OrderedDict()
        self._docs = tuple(docs)

    @property
    def docs(self) -> tuple[Document, ...]:
        return self._docs

    def replace_corpus(self, docs: list[Document]) -> None:
        with self._lock:
            self._docs = tuple(docs)
            self._views.clear()

    def _view(self, principal: Principal) -> SearchIndex:
        with self._lock:
            index = self._views.get(principal)
            if index is None:
                index = SearchIndex(tuple(prefilter(self._docs, principal)), self.dense)
                self._views[principal] = index
                if len(self._views) > 8:
                    self._views.popitem(last=False)
            else:
                self._views.move_to_end(principal)
            return index

    def query(
        self,
        query: str,
        principal: Principal,
        mode: Mode = "hybrid",
        top_k: int = 5,
        generate: bool = False,
    ) -> SearchResponse:
        effective_mode = "bm25" if mode == "hybrid" and self.dense is None else mode
        index = self._view(principal)
        if not index.docs:
            return SearchResponse("I don't know.", (), effective_mode, "no_evidence")

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
            start, end = evidence_span(doc.text, query)
            citations.append(Citation(doc_id, doc.title, doc.text[start:end], score, start, end))
        sources = tuple(citations)
        if not sources:
            return SearchResponse("I don't know.", (), effective_mode, "no_evidence")
        if generate and self.generator is not None:
            draft = self.generator.generate(query, sources)
            answer = self.generator.validate(draft, sources)
            if answer:
                return SearchResponse(answer, sources, effective_mode, "draft")
            return SearchResponse("I don't know.", sources, effective_mode, "draft_unavailable")
        return SearchResponse("I don't know.", sources, effective_mode, "evidence_candidates")
