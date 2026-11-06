from __future__ import annotations

import os
import re

import httpx

from .models import Citation


class OllamaGenerator:
    """Optional local generation; source-ID checking is citation validation, not entailment."""

    def __init__(self, model: str | None = None, url: str | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "")
        self.url = url or os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

    def generate(self, query: str, citations: tuple[Citation, ...]) -> str:
        if not self.model:
            return ""
        sources = "\n".join(f"[{c.doc_id}] {c.title}: {c.snippet}" for c in citations)
        prompt = f"Answer only from the supplied sources. Cite source IDs in square brackets. If unsupported, say I don't know.\nQuestion: {query}\nSources:\n{sources}"
        try:
            r = httpx.post(
                self.url, json={"model": self.model, "prompt": prompt, "stream": False}, timeout=45
            )
            r.raise_for_status()
            return str(r.json().get("response", ""))
        except (httpx.HTTPError, ValueError):
            return ""

    @staticmethod
    def validate(answer: str, citations: tuple[Citation, ...]) -> str:
        ids = {c.doc_id for c in citations}
        used = re.findall(r"\[([^\]]+)\]", answer)
        if not answer.strip() or not used or any(x not in ids for x in used):
            return ""
        return answer
