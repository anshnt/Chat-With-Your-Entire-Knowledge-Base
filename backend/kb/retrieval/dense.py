"""Dense vector retrieval.

Similarity is an exact cosine scan over an L2-normalised ``float32`` matrix held
by the store. At this project's scale that is the right call rather than a
compromise: 50k chunks × 1024 dims is ~200 MB and a full scan lands in single-
digit milliseconds under numpy's BLAS, with none of the recall loss, index build
time, or tuning surface of an ANN structure. The seam for swapping in an
approximate index is :meth:`DenseRetriever.search`, and nothing above it would
change.

Filtering is applied *before* the top-k selection, not after — post-filtering a
fixed-size result set silently reduces recall exactly when the filter is
selective, which is the case that matters.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from kb.embeddings.base import Embedder
from kb.models import ScoredChunk, SourceType
from kb.store import SQLiteStore


class DenseRetriever:
    """Exact cosine top-k over the collection's vectors."""

    name = "dense"

    def __init__(self, store: SQLiteStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def search(
        self,
        query: str,
        *,
        collection: str = "default",
        limit: int = 50,
        source_types: Sequence[SourceType] | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        matrix, chunk_ids = self.store.vector_matrix(collection, model=self.embedder.model)
        if matrix.size == 0 or not chunk_ids:
            return []

        query_vec = np.asarray(self.embedder.embed_query(query), dtype="float32")
        if query_vec.shape[0] != matrix.shape[1]:
            # Guard against querying a collection built with a different model.
            raise ValueError(
                f"query dim {query_vec.shape[0]} != index dim {matrix.shape[1]} "
                f"for model {self.embedder.model!r}; re-embed the collection"
            )
        norm = float(np.linalg.norm(query_vec))
        if norm > 1e-12:
            query_vec = query_vec / norm

        scores = matrix @ query_vec

        mask = self._build_mask(collection, chunk_ids, source_types, document_ids)
        if mask is not None:
            if not mask.any():
                return []
            scores = np.where(mask, scores, -np.inf)

        k = min(limit, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        ordered_ids = [chunk_ids[i] for i in top]
        chunks = {c.id: c for c in self.store.get_chunks(ordered_ids)}
        out: list[ScoredChunk] = []
        for rank, idx in enumerate(top, start=1):
            chunk = chunks.get(chunk_ids[idx])
            if chunk is None:
                continue
            score = float(scores[idx])
            out.append(
                ScoredChunk(
                    chunk=chunk,
                    score=score,
                    dense_score=score,
                    dense_rank=rank,
                    retrievers=[self.name],
                )
            )
        return out

    def _build_mask(
        self,
        collection: str,
        chunk_ids: list[str],
        source_types: Sequence[SourceType] | None,
        document_ids: Sequence[str] | None,
    ) -> np.ndarray | None:
        if not source_types and not document_ids:
            return None
        meta = self.store.chunk_metadata_map(collection)
        allowed_sources = {s.value for s in source_types} if source_types else None
        allowed_docs = set(document_ids) if document_ids else None
        mask = np.zeros(len(chunk_ids), dtype=bool)
        for i, chunk_id in enumerate(chunk_ids):
            info = meta.get(chunk_id)
            if info is None:
                continue
            if allowed_sources is not None and info["source_type"] not in allowed_sources:
                continue
            if allowed_docs is not None and info["document_id"] not in allowed_docs:
                continue
            mask[i] = True
        return mask

    def similar_to_chunk(
        self, chunk_id: str, *, collection: str = "default", limit: int = 10
    ) -> list[ScoredChunk]:
        """Nearest neighbours of an existing chunk — "more like this"."""
        vectors = self.store.get_embeddings([chunk_id])
        vector = vectors.get(chunk_id)
        if vector is None:
            return []
        matrix, chunk_ids = self.store.vector_matrix(collection, model=self.embedder.model)
        if matrix.size == 0:
            return []
        norm = float(np.linalg.norm(vector))
        query_vec = vector / norm if norm > 1e-12 else vector
        scores = matrix @ query_vec.astype("float32")
        k = min(limit + 1, len(chunk_ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        picked = [i for i in top if chunk_ids[i] != chunk_id][:limit]
        chunks = {c.id: c for c in self.store.get_chunks([chunk_ids[i] for i in picked])}
        return [
            ScoredChunk(
                chunk=chunks[chunk_ids[i]],
                score=float(scores[i]),
                dense_score=float(scores[i]),
                dense_rank=rank,
                retrievers=[self.name],
            )
            for rank, i in enumerate(picked, start=1)
            if chunk_ids[i] in chunks
        ]
