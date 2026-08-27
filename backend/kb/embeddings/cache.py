"""Content-addressed embedding cache.

Re-embedding text that has not changed is the single most wasteful thing an
ingestion pipeline does — re-running a corpus after a chunker tweak re-sends
every unchanged chunk. Keying on ``sha256(model + text)`` makes repeated
ingestion nearly free and makes evaluation sweeps affordable, since sweeping
retrieval parameters does not change the vectors at all.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from kb.embeddings.base import Embedder


def _key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


class CachedEmbedder(Embedder):
    """Wraps an embedder with a persistent on-disk cache."""

    def __init__(self, inner: Embedder, cache_path: str | Path) -> None:
        self.inner = inner
        self.model = inner.model
        self.dim = inner.dim
        self.path = Path(cache_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embedding_cache ("
            "  key TEXT PRIMARY KEY, model TEXT NOT NULL, dim INTEGER NOT NULL, vector BLOB NOT NULL)"
        )
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        keys = [_key(self.model, t) for t in texts]
        cached = self._fetch(keys)

        # Deduplicate within the batch as well as against the cache. Corpora
        # repeat themselves constantly — licence headers, boilerplate footers,
        # the same paragraph in two exports — and each duplicate would otherwise
        # be a separate billed call.
        pending: dict[str, str] = {}
        for key, text in zip(keys, texts, strict=True):
            if key not in cached:
                pending.setdefault(key, text)

        self.hits += sum(1 for k in keys if k in cached)
        self.misses += len(keys) - sum(1 for k in keys if k in cached)

        if pending:
            pending_keys = list(pending)
            fresh = self.inner.embed_documents([pending[k] for k in pending_keys])
            self._store(list(zip(pending_keys, fresh, strict=True)))
            for key, vector in zip(pending_keys, fresh, strict=True):
                cached[key] = vector

        return np.vstack([cached[k] for k in keys]).astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        # Queries are not cached: they are unique by nature and caching them
        # would grow the cache without ever being read again.
        return self.inner.embed_query(text)

    def _fetch(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        unique = list(dict.fromkeys(keys))
        for i in range(0, len(unique), 500):
            batch = unique[i : i + 500]
            placeholders = ",".join("?" * len(batch))
            rows = self._conn.execute(
                f"SELECT key, vector FROM embedding_cache WHERE key IN ({placeholders})", batch
            ).fetchall()
            for key, blob in rows:
                out[key] = np.frombuffer(blob, dtype="<f4")
        return out

    def _store(self, items: Sequence[tuple[str, np.ndarray]]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO embedding_cache (key, model, dim, vector) VALUES (?, ?, ?, ?)",
            [
                (key, self.model, int(vec.shape[0]), np.asarray(vec, dtype="<f4").tobytes())
                for key, vec in items
            ],
        )
        self._conn.commit()

    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }

    def close(self) -> None:
        self._conn.close()
