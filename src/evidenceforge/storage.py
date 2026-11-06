"""Transactional corpus snapshots shared by local server workers."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from .models import Document


class CorpusStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with closing(self.connect()) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, revision INTEGER)"
            )
            db.execute("INSERT OR IGNORE INTO state VALUES (1, 0)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS generation_cache (key TEXT PRIMARY KEY, span_id TEXT, expires REAL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS leases (name TEXT PRIMARY KEY, owner TEXT, expires REAL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS documents (position INTEGER PRIMARY KEY, body TEXT)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS requests (identity TEXT, bucket INTEGER, "
                "count INTEGER, PRIMARY KEY(identity, bucket))"
            )

    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.execute("PRAGMA synchronous=FULL")
        return db

    def replace(self, docs: list[Document], *, initialize: bool = False) -> int:
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            revision = db.execute("SELECT revision FROM state WHERE id=1").fetchone()[0]
            if initialize and revision:
                return revision
            db.execute("DELETE FROM documents")
            db.execute("DELETE FROM generation_cache")
            rows = (
                (
                    i,
                    json.dumps(
                        {
                            "doc_id": d.doc_id,
                            "title": d.title,
                            "text": d.text,
                            "tenant_id": d.tenant_id,
                            "allowed_roles": sorted(d.allowed_roles),
                            "metadata": d.metadata,
                        },
                        ensure_ascii=False,
                    ),
                )
                for i, d in enumerate(docs)
            )
            db.executemany("INSERT INTO documents VALUES (?, ?)", rows)
            db.execute("UPDATE state SET revision=revision+1 WHERE id=1")
            return revision + 1

    def snapshot(self, after: int = -1) -> tuple[int, list[Document] | None]:
        with closing(self.connect()) as db, db:
            db.execute("BEGIN")
            revision = db.execute("SELECT revision FROM state WHERE id=1").fetchone()[0]
            if revision == after:
                return revision, None
            docs = []
            for (body,) in db.execute("SELECT body FROM documents ORDER BY position"):
                row = json.loads(body)
                row["allowed_roles"] = frozenset(row["allowed_roles"])
                docs.append(Document(**row))
            return revision, docs

    def consume(self, identity: str, limit: int) -> bool:
        bucket = int(time.time() // 60)
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM requests WHERE bucket < ?", (bucket,))
            db.execute(
                "INSERT INTO requests VALUES (?, ?, 1) ON CONFLICT(identity, bucket) "
                "DO UPDATE SET count=count+1",
                (identity, bucket),
            )
            count = db.execute(
                "SELECT count FROM requests WHERE identity=? AND bucket=?", (identity, bucket)
            ).fetchone()[0]
            return count <= limit

    def current_revision(self) -> int:
        with closing(self.connect()) as db:
            return db.execute("SELECT revision FROM state WHERE id=1").fetchone()[0]

    def acquire_generation(self, owner: str, duration: float) -> bool:
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM leases WHERE expires < ?", (time.time(),))
            return (
                db.execute(
                    "INSERT OR IGNORE INTO leases VALUES ('generation', ?, ?)",
                    (owner, time.time() + duration),
                ).rowcount
                == 1
            )

    def release_generation(self, owner: str) -> None:
        with closing(self.connect()) as db, db:
            db.execute("DELETE FROM leases WHERE name='generation' AND owner=?", (owner,))

    def cached_span(self, key: str) -> str | None:
        with closing(self.connect()) as db:
            row = db.execute(
                "SELECT span_id FROM generation_cache WHERE key=? AND expires>?", (key, time.time())
            ).fetchone()
            return row[0] if row else None

    def cache_span(self, key: str, span_id: str, seconds: int) -> None:
        with closing(self.connect()) as db, db:
            db.execute("DELETE FROM generation_cache WHERE expires<=?", (time.time(),))
            db.execute(
                "INSERT OR REPLACE INTO generation_cache VALUES (?, ?, ?)",
                (key, span_id, time.time() + seconds),
            )
            db.execute(
                "DELETE FROM generation_cache WHERE key NOT IN "
                "(SELECT key FROM generation_cache ORDER BY expires DESC LIMIT 256)"
            )
