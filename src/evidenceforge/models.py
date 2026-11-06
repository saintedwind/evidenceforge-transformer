from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

Name = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")]
Mode = Literal["bm25", "dense", "hybrid"]


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: Mode = "hybrid"
    generate: bool = False

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()


class CorpusPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    tenant_id: Name
    allowed_roles: list[Name] = Field(max_length=64)


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=500)
    policy: CorpusPolicy | None = None


@dataclass(frozen=True, slots=True)
class Principal:
    tenant_id: str
    roles: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    title: str
    text: str
    tenant_id: str = "public"
    allowed_roles: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Citation:
    doc_id: str
    title: str
    snippet: str
    score: float
    start: int = 0
    end: int = 0


@dataclass(frozen=True, slots=True)
class SearchResponse:
    answer: str
    citations: tuple[Citation, ...]
    mode: Mode
    status: str
