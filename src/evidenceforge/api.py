import hashlib
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from hmac import compare_digest
from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import __version__
from .auth import principals_from_env
from .dense import E5Encoder
from .generation import SourceGenerator
from .ingest import load_beir_corpus
from .models import IngestRequest, Principal, QueryRequest
from .operations import RequestLimits
from .retrieval import SearchService
from .storage import CorpusStore


def _configured_service() -> SearchService:
    state = os.getenv("EVIDENCEFORGE_STATE")
    store = CorpusStore(Path(state).resolve()) if state else None
    corpus = os.getenv("EVIDENCEFORGE_CORPUS")
    docs = load_beir_corpus(corpus) if corpus and not (store and store.current_revision()) else []
    encoder = None
    if os.getenv("EVIDENCEFORGE_DENSE") == "1":
        encoder = E5Encoder(
            os.getenv("EVIDENCEFORGE_MODEL_PATH") or None,
            int(os.getenv("EVIDENCEFORGE_THREADS", "6")),
        )
    generator = (
        SourceGenerator(store=store)
        if os.getenv("EVIDENCEFORGE_LLM_MODEL") or os.getenv("OLLAMA_MODEL")
        else None
    )
    return SearchService(docs, encoder, generator, store)


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
        app.state.rate_lock = Lock()
        app.state.rates = {}
        app.state.rate_limit = int(os.getenv("EVIDENCEFORGE_REQUESTS_PER_MINUTE", "120"))
        if app.state.rate_limit < 1:
            raise ValueError("request limit must be positive")
        yield

    app = FastAPI(title="EvidenceForge", version=__version__, lifespan=lifespan)
    app.add_middleware(RequestLimits)

    @app.get("/health")
    def health(request: Request):
        search = request.app.state.search
        search.refresh()
        return {
            "status": "ok",
            "dense": search.dense is not None,
            "generation": search.generator is not None,
            "corpus_revision": search.revision,
            "documents": len(search.docs),
            "durable": search.store is not None,
        }

    @app.get("/ready")
    def ready(request: Request):
        search = request.app.state.search
        search.refresh()
        if not search.docs:
            raise HTTPException(503, "corpus is empty")
        if search.generator and not search.generator.ready():
            raise HTTPException(503, "local model is not ready")
        return {"status": "ready", "corpus_revision": search.revision}

    def current(request: Request) -> Principal:
        principal = request.app.state.principals.get(request.headers.get("x-api-key", ""))
        if principal is None:
            raise HTTPException(401, "invalid API key")
        identity = hashlib.sha256(request.headers["x-api-key"].encode()).hexdigest()
        state = request.app.state
        if state.search.store:
            allowed = state.search.store.consume(identity, state.rate_limit)
        else:
            bucket = int(time.time() // 60)
            with state.rate_lock:
                state.rates = {key: value for key, value in state.rates.items() if key[1] == bucket}
                key = (identity, bucket)
                state.rates[key] = state.rates.get(key, 0) + 1
                allowed = state.rates[key] <= state.rate_limit
        if not allowed:
            raise HTTPException(429, "request limit exceeded", headers={"Retry-After": "60"})
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
        revision = search.store.current_revision() if search.store else search.revision
        if revision != result.corpus_revision:
            raise HTTPException(409, "corpus changed during the request; retry")
        body = {
            "answer": result.answer,
            "requested_mode": req.mode,
            "mode": result.mode,
            "status": result.status,
            "citations": [asdict(citation) for citation in result.citations],
            "corpus_revision": result.corpus_revision,
            "answer_source": "internal_documents"
            if result.status == "grounded_extract"
            else "none",
            "generation": result.generation,
            "model_knowledge_used": False,
        }
        code = (
            503
            if result.status in {"generation_busy", "generation_timeout", "generation_unavailable"}
            else 200
        )
        return JSONResponse(body, status_code=code)

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
        return {"ingested": len(documents), "corpus_revision": request.app.state.search.revision}

    return app


app = create_app()
