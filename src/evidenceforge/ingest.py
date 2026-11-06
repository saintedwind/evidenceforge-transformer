from __future__ import annotations

import json
from pathlib import Path

from .models import CorpusPolicy, Document

MAX_CORPUS_BYTES = 64 * 1024 * 1024
MAX_LINE_BYTES = 256 * 1024
MAX_DOCUMENTS = 50_000


def load_beir_corpus(
    path: str | Path, policy: CorpusPolicy | None = None, *, require_policy: bool = False
) -> list[Document]:
    """Read BEIR JSONL, preserving explicit per-document access policy."""
    documents: list[Document] = []
    ids: set[str] = set()
    bytes_read = 0
    with Path(path).open("rb") as source:
        line_number = 0
        while line := source.readline(MAX_LINE_BYTES + 1):
            line_number += 1
            bytes_read += len(line)
            if bytes_read > MAX_CORPUS_BYTES or len(line) > MAX_LINE_BYTES:
                raise ValueError("corpus or line exceeds its size limit")
            if not line.strip():
                continue
            try:
                row = json.loads(line.decode("utf-8"))
                document = _document(row, policy, require_policy=require_policy)
            except (ValueError, TypeError) as error:
                raise ValueError(f"invalid document at line {line_number}") from error
            if document.doc_id in ids:
                raise ValueError(f"duplicate document ID at line {line_number}")
            if len(documents) == MAX_DOCUMENTS:
                raise ValueError("corpus exceeds its document limit")
            ids.add(document.doc_id)
            documents.append(document)
    if not documents:
        raise ValueError("corpus is empty")
    return documents


def _document(
    row: object, policy: CorpusPolicy | None, *, require_policy: bool = False
) -> Document:
    if not isinstance(row, dict):
        raise ValueError("each row must be a JSON object")
    doc_id = row.get("_id")
    if type(doc_id) is int:
        doc_id = str(doc_id)
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("a document ID is required")
    title, text = row.get("title", ""), row.get("text")
    metadata = row.get("metadata", {})
    if not isinstance(title, str) or not isinstance(text, str) or not isinstance(metadata, dict):
        raise ValueError("invalid title, text or metadata")
    acl_fields = {"tenant_id", "allowed_roles"} & row.keys()
    if acl_fields:
        if policy is not None or len(acl_fields) != 2:
            raise ValueError("provide a complete ACL in one place")
        policy = CorpusPolicy.model_validate({field: row[field] for field in acl_fields})
    if policy is None:
        if require_policy:
            raise ValueError("an explicit import policy is required")
        return Document(doc_id, title, text, metadata=metadata)
    return Document(
        doc_id, title, text, policy.tenant_id, frozenset(policy.allowed_roles), metadata
    )


def write_jsonl(docs: list[Document], path: str | Path) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as output:
        for doc in docs:
            row = {
                "_id": doc.doc_id,
                "title": doc.title,
                "text": doc.text,
                "metadata": doc.metadata,
                "tenant_id": doc.tenant_id,
                "allowed_roles": sorted(doc.allowed_roles),
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(docs)
