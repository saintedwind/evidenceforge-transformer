from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .models import Document

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


class BM25Index:
    """Small transparent BM25 (Okapi) inverted index; document-level ranking."""

    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.tf = []
        self.lengths = []
        self.postings = defaultdict(list)
        self.df = Counter()
        for i, d in enumerate(docs):
            counts = Counter(tokenize(d.title + " " + d.text))
            self.tf.append(counts)
            self.lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                self.postings[term].append((i, tf))
                self.df[term] += 1
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        scores = defaultdict(float)
        n = len(self.docs)
        for term in tokenize(query):
            df = self.df.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, tf in self.postings[term]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avgdl or 1))
                scores[i] += idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores, key=lambda i: (-scores[i], self.docs[i].doc_id))[:top_k]
        return [(self.docs[i].doc_id, float(scores[i])) for i in ranked]
