"""SQLite-backed document, chunk, and vector store.

Design notes
------------
* One file holds documents, chunks, the FTS5/BM25 index and the dense vectors,
  so lexical and dense views of the corpus can never drift apart.
* Vectors are little-endian ``float32`` blobs. For a knowledge base of the size
  this project targets (tens of thousands of chunks) an exact numpy scan is both
  faster and more accurate than an approximate index, and it removes a
  dependency. :meth:`SQLiteStore.vector_matrix` caches the assembled matrix and
  invalidates it on write.
* All writes go through short transactions and the connection is per-thread, so
  the FastAPI thread pool can use the store without a global lock.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from kb.errors import NotFoundError
from kb.models import (
    Chunk,
    ChunkKind,
    CollectionStats,
    Document,
    SourceType,
    content_hash,
    parse_locator,
    utcnow,
)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = "1"


def _dt(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def vector_to_blob(vector: Sequence[float] | np.ndarray) -> bytes:
    """Serialise a vector to a little-endian float32 blob."""
    arr = np.asarray(vector, dtype="<f4")
    return arr.tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    """Deserialise a float32 blob back into a numpy array."""
    return np.frombuffer(blob, dtype="<f4")


class SQLiteStore:
    """Persistent store for documents, chunks, vectors and retrieval events."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._shared: sqlite3.Connection | None = None
        self._matrix_cache: dict[tuple[str, str], tuple[int, np.ndarray, list[str]]] = {}
        self._write_counter = 0
        self._lock = threading.RLock()
        self._init_schema()

    # ------------------------------------------------------------------ #
    # connection management
    # ------------------------------------------------------------------ #

    @property
    def conn(self) -> sqlite3.Connection:
        if str(self.path) == ":memory:":
            # An in-memory database cannot be reopened per thread, so tests share one.
            if self._shared is None:
                self._shared = self._connect()
            return self._shared
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")
        if str(self.path) != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _init_schema(self) -> None:
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with self.conn as conn:
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (SCHEMA_VERSION,),
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a write transaction and invalidate the vector matrix cache."""
        conn = self.conn
        try:
            with conn:
                yield conn
        finally:
            with self._lock:
                self._write_counter += 1

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
        if self._shared is not None:
            self._shared.close()
            self._shared = None

    # ------------------------------------------------------------------ #
    # collections
    # ------------------------------------------------------------------ #

    def ensure_collection(
        self,
        name: str,
        *,
        description: str = "",
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        now = utcnow().isoformat()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO collections (name, description, embedding_model, embedding_dim,
                                         created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description     = COALESCE(NULLIF(excluded.description, ''), collections.description),
                    embedding_model = COALESCE(excluded.embedding_model, collections.embedding_model),
                    embedding_dim   = COALESCE(excluded.embedding_dim, collections.embedding_dim),
                    updated_at      = excluded.updated_at
                """,
                (name, description, embedding_model, embedding_dim, now, now),
            )

    def list_collections(self) -> list[str]:
        rows = self.conn.execute(
            """
            SELECT name FROM collections
            UNION
            SELECT DISTINCT collection AS name FROM documents
            ORDER BY name
            """
        ).fetchall()
        return [r["name"] for r in rows]

    def collection_stats(self, collection: str = "default") -> CollectionStats:
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS n_documents, COALESCE(SUM(token_estimate), 0) AS total_tokens
            FROM documents WHERE collection = ?
            """,
            (collection,),
        ).fetchone()
        chunk_row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE collection = ?", (collection,)
        ).fetchone()
        emb_row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM embeddings WHERE collection = ?", (collection,)
        ).fetchone()
        by_source = {
            r["source_type"]: r["n"]
            for r in self.conn.execute(
                "SELECT source_type, COUNT(*) AS n FROM documents WHERE collection = ? "
                "GROUP BY source_type ORDER BY n DESC",
                (collection,),
            )
        }
        meta = self.conn.execute(
            "SELECT embedding_model, embedding_dim FROM collections WHERE name = ?",
            (collection,),
        ).fetchone()
        return CollectionStats(
            collection=collection,
            n_documents=row["n_documents"],
            n_chunks=chunk_row["n"],
            n_embedded=emb_row["n"],
            total_tokens=row["total_tokens"],
            by_source_type=by_source,
            embedding_model=meta["embedding_model"] if meta else None,
            embedding_dim=meta["embedding_dim"] if meta else None,
        )

    def delete_collection(self, collection: str) -> int:
        with self.transaction() as conn:
            n = conn.execute(
                "SELECT COUNT(*) AS n FROM documents WHERE collection = ?", (collection,)
            ).fetchone()["n"]
            conn.execute("DELETE FROM documents WHERE collection = ?", (collection,))
            conn.execute("DELETE FROM chunks WHERE collection = ?", (collection,))
            conn.execute("DELETE FROM embeddings WHERE collection = ?", (collection,))
            conn.execute("DELETE FROM retrieval_events WHERE collection = ?", (collection,))
            conn.execute("DELETE FROM collections WHERE name = ?", (collection,))
        return int(n)

    # ------------------------------------------------------------------ #
    # documents
    # ------------------------------------------------------------------ #

    def find_document_by_hash(self, collection: str, doc_hash: str) -> Document | None:
        row = self.conn.execute(
            "SELECT * FROM documents WHERE collection = ? AND content_hash = ?",
            (collection, doc_hash),
        ).fetchone()
        return _row_to_document(row) if row else None

    def add_document(self, document: Document, chunks: Sequence[Chunk]) -> Document:
        """Insert a document and its chunks atomically.

        The FTS index and vector-cache invalidation are handled by triggers and
        the transaction wrapper, so callers never touch them.
        """
        doc = document.model_copy(
            update={
                "n_chunks": len(chunks),
                "content_hash": document.content_hash or content_hash(document.uri),
                "updated_at": utcnow(),
            }
        )
        self.ensure_collection(doc.collection)
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, collection, source_type, title, uri, content_hash,
                                       byte_size, token_estimate, n_chunks, language, author,
                                       published_at, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.id,
                    doc.collection,
                    doc.source_type.value,
                    doc.title,
                    doc.uri,
                    doc.content_hash,
                    doc.byte_size,
                    doc.token_estimate,
                    doc.n_chunks,
                    doc.language,
                    doc.author,
                    _dt(doc.published_at),
                    _dt(doc.created_at),
                    _dt(doc.updated_at),
                    json.dumps(doc.metadata),
                ),
            )
            conn.executemany(
                """
                INSERT INTO chunks (id, document_id, collection, ordinal, text, kind, locator,
                                    token_estimate, content_hash, document_title, source_type,
                                    heading_context, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.id,
                        doc.id,
                        doc.collection,
                        c.ordinal,
                        c.text,
                        c.kind.value,
                        c.locator.model_dump_json(),
                        c.token_estimate,
                        c.content_hash,
                        doc.title,
                        doc.source_type.value,
                        c.heading_context,
                        json.dumps(c.metadata),
                    )
                    for c in chunks
                ],
            )
        return doc

    def get_document(self, document_id: str) -> Document:
        row = self.conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"document {document_id!r} not found")
        return _row_to_document(row)

    def list_documents(
        self,
        collection: str = "default",
        *,
        limit: int = 100,
        offset: int = 0,
        source_type: SourceType | None = None,
        search: str | None = None,
    ) -> list[Document]:
        sql = "SELECT * FROM documents WHERE collection = ?"
        params: list[Any] = [collection]
        if source_type is not None:
            sql += " AND source_type = ?"
            params.append(source_type.value)
        if search:
            sql += " AND (title LIKE ? OR uri LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])
        sql += " ORDER BY created_at DESC, id LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        return [_row_to_document(r) for r in self.conn.execute(sql, params)]

    def count_documents(self, collection: str = "default") -> int:
        return int(
            self.conn.execute(
                "SELECT COUNT(*) AS n FROM documents WHERE collection = ?", (collection,)
            ).fetchone()["n"]
        )

    def delete_document(self, document_id: str) -> None:
        with self.transaction() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            if cur.rowcount == 0:
                raise NotFoundError(f"document {document_id!r} not found")
            # Chunks cascade; embeddings cascade from chunks.
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

    # ------------------------------------------------------------------ #
    # chunks
    # ------------------------------------------------------------------ #

    def get_chunk(self, chunk_id: str) -> Chunk:
        row = self.conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"chunk {chunk_id!r} not found")
        return _row_to_chunk(row)

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        """Fetch many chunks, preserving the order of ``chunk_ids``."""
        if not chunk_ids:
            return []
        by_id: dict[str, Chunk] = {}
        for batch in _batched(list(chunk_ids), 500):
            placeholders = ",".join("?" * len(batch))
            for row in self.conn.execute(
                f"SELECT * FROM chunks WHERE id IN ({placeholders})", batch
            ):
                by_id[row["id"]] = _row_to_chunk(row)
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def document_chunks(self, document_id: str) -> list[Chunk]:
        return [
            _row_to_chunk(r)
            for r in self.conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY ordinal", (document_id,)
            )
        ]

    def iter_chunks(
        self, collection: str = "default", *, batch_size: int = 500
    ) -> Iterator[list[Chunk]]:
        """Stream chunks in batches — used by embedding backfill and evaluation."""
        offset = 0
        while True:
            rows = self.conn.execute(
                "SELECT * FROM chunks WHERE collection = ? ORDER BY seq LIMIT ? OFFSET ?",
                (collection, batch_size, offset),
            ).fetchall()
            if not rows:
                return
            yield [_row_to_chunk(r) for r in rows]
            offset += len(rows)

    def count_chunks(self, collection: str = "default") -> int:
        return int(
            self.conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE collection = ?", (collection,)
            ).fetchone()["n"]
        )

    def chunk_neighbours(self, chunk_id: str, window: int = 1) -> list[Chunk]:
        """Chunks immediately before/after ``chunk_id`` in its document.

        Used to widen context around a hit without widening the retrieval unit —
        small chunks retrieve precisely, neighbours restore readability.
        """
        row = self.conn.execute(
            "SELECT document_id, ordinal FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            raise NotFoundError(f"chunk {chunk_id!r} not found")
        lo = max(0, row["ordinal"] - window)
        hi = row["ordinal"] + window
        return [
            _row_to_chunk(r)
            for r in self.conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? AND ordinal BETWEEN ? AND ? "
                "ORDER BY ordinal",
                (row["document_id"], lo, hi),
            )
        ]

    # ------------------------------------------------------------------ #
    # lexical search (BM25 via FTS5)
    # ------------------------------------------------------------------ #

    def search_lexical(
        self,
        match_query: str,
        *,
        collection: str = "default",
        limit: int = 50,
        source_types: Sequence[str] | None = None,
        document_ids: Sequence[str] | None = None,
        column_weights: tuple[float, float, float] = (1.0, 0.4, 0.25),
    ) -> list[tuple[str, float]]:
        """Run a BM25 query and return ``(chunk_id, score)`` best-first.

        FTS5's ``bm25()`` returns *more negative is better*; we negate it so all
        scores in the system are "higher is better". Column weights let a match
        in the body outrank a match in a heading or a title.
        """
        sql = [
            "SELECT c.id AS id, -bm25(chunks_fts, ?, ?, ?) AS score",
            "FROM chunks_fts JOIN chunks c ON c.seq = chunks_fts.rowid",
            "WHERE chunks_fts MATCH ? AND c.collection = ?",
        ]
        params: list[Any] = [*column_weights, match_query, collection]
        if source_types:
            sql.append(f"AND c.source_type IN ({','.join('?' * len(source_types))})")
            params.extend(source_types)
        if document_ids:
            sql.append(f"AND c.document_id IN ({','.join('?' * len(document_ids))})")
            params.extend(document_ids)
        sql.append("ORDER BY score DESC LIMIT ?")
        params.append(limit)
        rows = self.conn.execute("\n".join(sql), params).fetchall()
        return [(r["id"], float(r["score"])) for r in rows]

    # ------------------------------------------------------------------ #
    # embeddings
    # ------------------------------------------------------------------ #

    def upsert_embeddings(
        self,
        items: Iterable[tuple[str, np.ndarray]],
        *,
        collection: str,
        model: str,
    ) -> int:
        now = utcnow().isoformat()
        rows = []
        dim = 0
        for chunk_id, vector in items:
            arr = np.asarray(vector, dtype="<f4")
            dim = int(arr.shape[0])
            rows.append(
                (chunk_id, collection, model, dim, arr.tobytes(), float(np.linalg.norm(arr)), now)
            )
        if not rows:
            return 0
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO embeddings (chunk_id, collection, model, dim, vector, norm, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    collection = excluded.collection,
                    model      = excluded.model,
                    dim        = excluded.dim,
                    vector     = excluded.vector,
                    norm       = excluded.norm,
                    created_at = excluded.created_at
                """,
                rows,
            )
        self.ensure_collection(collection, embedding_model=model, embedding_dim=dim)
        return len(rows)

    def unembedded_chunk_ids(
        self, collection: str = "default", *, model: str | None = None, limit: int = 1000
    ) -> list[str]:
        """Chunk ids that have no vector for ``model`` yet."""
        if model:
            sql = """
                SELECT c.id FROM chunks c
                LEFT JOIN embeddings e ON e.chunk_id = c.id AND e.model = ?
                WHERE c.collection = ? AND e.chunk_id IS NULL
                ORDER BY c.seq LIMIT ?
            """
            params: list[Any] = [model, collection, limit]
        else:
            sql = """
                SELECT c.id FROM chunks c
                LEFT JOIN embeddings e ON e.chunk_id = c.id
                WHERE c.collection = ? AND e.chunk_id IS NULL
                ORDER BY c.seq LIMIT ?
            """
            params = [collection, limit]
        return [r["id"] for r in self.conn.execute(sql, params)]

    def vector_matrix(
        self, collection: str = "default", *, model: str
    ) -> tuple[np.ndarray, list[str]]:
        """Return ``(matrix, chunk_ids)`` with L2-normalised rows.

        Cached and invalidated by the write counter, so repeated queries between
        writes cost one dictionary lookup.
        """
        key = (collection, model)
        with self._lock:
            cached = self._matrix_cache.get(key)
            if cached is not None and cached[0] == self._write_counter:
                return cached[1], cached[2]

        rows = self.conn.execute(
            "SELECT chunk_id, vector, dim FROM embeddings "
            "WHERE collection = ? AND model = ? ORDER BY chunk_id",
            (collection, model),
        ).fetchall()
        if not rows:
            empty = np.zeros((0, 0), dtype="float32")
            with self._lock:
                self._matrix_cache[key] = (self._write_counter, empty, [])
            return empty, []

        dim = int(rows[0]["dim"])
        matrix = np.empty((len(rows), dim), dtype="float32")
        ids: list[str] = []
        for i, row in enumerate(rows):
            matrix[i] = blob_to_vector(row["vector"])
            ids.append(row["chunk_id"])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        np.divide(matrix, np.maximum(norms, 1e-12), out=matrix)
        with self._lock:
            self._matrix_cache[key] = (self._write_counter, matrix, ids)
        return matrix, ids

    def get_embeddings(self, chunk_ids: Sequence[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for batch in _batched(list(chunk_ids), 500):
            placeholders = ",".join("?" * len(batch))
            for row in self.conn.execute(
                f"SELECT chunk_id, vector FROM embeddings WHERE chunk_id IN ({placeholders})",
                batch,
            ):
                out[row["chunk_id"]] = blob_to_vector(row["vector"])
        return out

    def chunk_metadata_map(self, collection: str = "default") -> dict[str, dict[str, Any]]:
        """Lightweight ``chunk_id -> {document_id, title, ordinal, ...}`` map.

        Used by filtering and by the corpus map, where loading full chunk text
        for the whole collection would be wasteful.
        """
        return {
            r["id"]: {
                "document_id": r["document_id"],
                "document_title": r["document_title"],
                "source_type": r["source_type"],
                "ordinal": r["ordinal"],
                "kind": r["kind"],
                "token_estimate": r["token_estimate"],
            }
            for r in self.conn.execute(
                "SELECT id, document_id, document_title, source_type, ordinal, kind, "
                "token_estimate FROM chunks WHERE collection = ?",
                (collection,),
            )
        }

    # ------------------------------------------------------------------ #
    # retrieval telemetry
    # ------------------------------------------------------------------ #

    def log_retrieval(
        self,
        query: str,
        hits: Sequence[tuple[str, float]],
        *,
        collection: str = "default",
        strategy: str = "",
    ) -> None:
        if not hits:
            return
        now = utcnow().isoformat()
        qhash = content_hash(query)[:16]
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO retrieval_events
                    (collection, query, query_hash, chunk_id, rank, score, strategy, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (collection, query, qhash, cid, rank, float(score), strategy, now)
                    for rank, (cid, score) in enumerate(hits, start=1)
                ],
            )

    def retrieval_heatmap(
        self, collection: str = "default", *, limit: int = 500
    ) -> list[dict[str, Any]]:
        """Per-chunk retrieval counts — which parts of the corpus earn their keep."""
        rows = self.conn.execute(
            """
            SELECT e.chunk_id                AS chunk_id,
                   COUNT(*)                  AS hits,
                   AVG(e.rank)               AS avg_rank,
                   AVG(e.score)              AS avg_score,
                   MAX(e.created_at)         AS last_seen,
                   c.document_id             AS document_id,
                   c.document_title          AS document_title
            FROM retrieval_events e
            JOIN chunks c ON c.id = e.chunk_id
            WHERE e.collection = ?
            GROUP BY e.chunk_id
            ORDER BY hits DESC, avg_rank ASC
            LIMIT ?
            """,
            (collection, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def recent_queries(self, collection: str = "default", *, limit: int = 50) -> list[str]:
        rows = self.conn.execute(
            "SELECT query, MAX(created_at) AS ts FROM retrieval_events WHERE collection = ? "
            "GROUP BY query_hash ORDER BY ts DESC LIMIT ?",
            (collection, limit),
        ).fetchall()
        return [r["query"] for r in rows]


# --------------------------------------------------------------------------- #
# row mapping
# --------------------------------------------------------------------------- #


def _row_to_document(row: sqlite3.Row) -> Document:
    return Document(
        id=row["id"],
        collection=row["collection"],
        source_type=SourceType(row["source_type"]),
        title=row["title"],
        uri=row["uri"],
        content_hash=row["content_hash"],
        byte_size=row["byte_size"],
        token_estimate=row["token_estimate"],
        n_chunks=row["n_chunks"],
        language=row["language"],
        author=row["author"],
        published_at=row["published_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        metadata=json.loads(row["metadata"] or "{}"),
    )


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        document_id=row["document_id"],
        collection=row["collection"],
        ordinal=row["ordinal"],
        text=row["text"],
        kind=ChunkKind(row["kind"]),
        locator=parse_locator(json.loads(row["locator"])),
        token_estimate=row["token_estimate"],
        content_hash=row["content_hash"],
        document_title=row["document_title"],
        source_type=SourceType(row["source_type"]) if row["source_type"] else None,
        heading_context=row["heading_context"],
        metadata=json.loads(row["metadata"] or "{}"),
    )


def _batched(items: list[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
