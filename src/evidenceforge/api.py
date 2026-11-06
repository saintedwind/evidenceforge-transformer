import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from hmac import compare_digest
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request

from . import __version__
from .auth import principals_from_env
from .dense import E5Encoder
from .generation import OllamaGenerator
from .ingest import load_beir_corpus
from .models import IngestRequest, Principal, QueryRequest
from .retrieval import SearchService


def _configured_service() -> SearchService:
    corpus = os.getenv("EVIDENCEFORGE_CORPUS")
    docs = load_beir_corpus(corpus) if corpus else []
    encoder = None
    if os.getenv("EVIDENCEFORGE_DENSE") == "1":
        encoder = E5Encoder(
            os.getenv("EVIDENCEFORGE_MODEL_PATH") or None,
            int(os.getenv("EVIDENCEFORGE_THREADS", "6")),
        )
    generator = OllamaGenerator() if os.getenv("OLLAMA_MODEL") else None
    return SearchService(docs, encoder, generator)


def _resolve_import(root: Path, path: str) -> Path:
    supplied = Path(path)
    if supplied.drive and not supplied.is_absolute():
        raise ValueError("relative drive paths are unsupported")
    candidate = root / supplied
    if not candidate.is_relative_to(root) or ".." in candidate.parts:
        raise ValueError("path outside import root")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or resolved.suffix.lower() != ".jsonl":
        raise ValueError("path outside import root or unsupported extension")
    if not resolved.is_file():
        raise ValueError("import must be a file")
    return resolved


def create_app(service: SearchService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.principals = principals_from_env()
        app.state.admin_token = os.getenv("EVIDENCEFORGE_ADMIN_TOKEN", "").encode("utf-8")
        root = os.getenv("EVIDENCEFORGE_IMPORT_ROOT")
        app.state.import_root = Path(root).resolve(strict=True) if root else None
        if app.state.import_root is not None and not app.state.import_root.is_dir():
            raise ValueError("EVIDENCEFORGE_IMPORT_ROOT must be a directory")
        app.state.search = service if service is not None else _configured_service()
        yield

    app = FastAPI(title="EvidenceForge", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "dense": request.app.state.search.dense is not None}

    def current(request: Request) -> Principal:
        principal = request.app.state.principals.get(request.headers.get("x-api-key", ""))
        if principal is None:
            raise HTTPException(401, "invalid API key")
        return principal

    @app.post("/query")
    def query(
        req: QueryRequest, request: Request, principal: Annotated[Principal, Depends(current)]
    ):
        search = request.app.state.search
        if req.mode == "dense" and search.dense is None:
            raise HTTPException(503, "dense model is not configured")
        if req.generate and search.generator is None:
            raise HTTPException(503, "text generation is not configured")
        result = search.query(req.query, principal, req.mode, req.top_k, req.generate)
        return {
            "answer": result.answer,
            "requested_mode": req.mode,
            "mode": result.mode,
            "status": result.status,
            "citations": [asdict(citation) for citation in result.citations],
        }

    @app.post("/ingest")
    def ingest(payload: IngestRequest, request: Request):
        token = request.app.state.admin_token
        supplied = request.headers.get("x-admin-token", "").encode("utf-8")
        if not token or not compare_digest(supplied, token):
            raise HTTPException(403, "admin authentication required")
        root = request.app.state.import_root
        if root is None:
            raise HTTPException(503, "import root is not configured")
        try:
            source = _resolve_import(root, payload.path)
            documents = load_beir_corpus(source, payload.policy, require_policy=True)
        except (OSError, ValueError, RuntimeError):
            raise HTTPException(422, "unable to import corpus") from None
        request.app.state.search.replace_corpus(documents)
        return {"ingested": len(documents)}

    return app


app = create_app()
