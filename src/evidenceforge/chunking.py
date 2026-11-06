from __future__ import annotations

import re
from collections.abc import Iterator

from .models import Document

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def sentence_chunks(text: str, max_chars: int = 1200) -> list[str]:
    """Deterministically group sentences; long sentences split at whitespace."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    parts = [p.strip() for p in _SENTENCE.split(text.strip()) if p.strip()]
    out: list[str] = []
    buf = ""
    for part in parts:
        while len(part) > max_chars:
            if buf:
                out.append(buf)
                buf = ""
            cut = part.rfind(" ", 0, max_chars + 1)
            cut = cut if cut > 0 else max_chars
            out.append(part[:cut].strip())
            part = part[cut:].strip()
        candidate = f"{buf} {part}".strip()
        if buf and len(candidate) > max_chars:
            out.append(buf)
            buf = part
        else:
            buf = candidate
    if buf:
        out.append(buf)
    return out


def document_chunks(doc: Document, max_chars: int = 1200) -> Iterator[Document]:
    for i, chunk in enumerate(sentence_chunks(doc.text, max_chars)):
        yield Document(
            f"{doc.doc_id}#chunk-{i}",
            doc.title,
            chunk,
            doc.tenant_id,
            doc.allowed_roles,
            doc.metadata,
        )
