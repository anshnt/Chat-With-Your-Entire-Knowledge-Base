"""The corpus map: what the visualisation endpoints actually return.

Assembles a projection, clusters, cluster labels and per-chunk retrieval counts
into one payload a client can render without further computation.

The design constraint that shapes this module is that a corpus map is a *whole
corpus* operation — it touches every vector and every chunk's text — so it must
not be recomputed per request. Results are cached against the store's write
counter, which is the same invalidation signal the vector matrix uses, so the map
is recomputed exactly when the corpus changes and never otherwise.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from kb.viz.clustering import Cluster, cluster_corpus
from kb.viz.projection import normalize_to_unit_square, project

log = logging.getLogger(__name__)

#: Above this many chunks, a scatter plot is a solid blob and the browser
#: struggles. Sampling keeps the map legible and the payload sane; the response
#: says how many were sampled so the number is never silently misleading.
DEFAULT_MAX_POINTS = 2000

#: Longest snippet per point. The map shows thousands of them, so this dominates
#: the payload size.
SNIPPET_CHARS = 180


@dataclass(slots=True)
class MapPoint:
    """One chunk, placed."""

    chunk_id: str
    document_id: str
    document_title: str
    x: float
    y: float
    cluster: int
    source_type: str | None
    kind: str
    snippet: str
    heading: str
    tokens: int
    retrievals: int = 0
    """How often this chunk has been retrieved. Zero means it has never once
    earned its place in an answer."""


@dataclass(slots=True)
class CorpusMap:
    """A rendered corpus map."""

    collection: str
    points: list[MapPoint]
    clusters: list[dict[str, Any]]
    method: str
    n_chunks: int
    n_plotted: int
    sampled: bool
    explained_variance: float | None = None
    notes: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def coverage(self) -> float:
        """Share of plotted chunks that have ever been retrieved.

        The single most useful number on the page: it says how much of the corpus
        is doing any work at all.
        """
        if not self.points:
            return 0.0
        used = sum(1 for point in self.points if point.retrievals > 0)
        return round(used / len(self.points), 4)


class CorpusMapBuilder:
    """Builds and caches corpus maps for a knowledge base."""

    def __init__(self, knowledge_base: Any) -> None:
        self.kb = knowledge_base
        self._cache: dict[tuple, tuple[int, CorpusMap]] = {}

    def build(
        self,
        collection: str = "default",
        *,
        method: Literal["auto", "umap", "tsne", "pca"] = "auto",
        k: int | None = None,
        max_points: int = DEFAULT_MAX_POINTS,
        include_retrievals: bool = True,
    ) -> CorpusMap:
        """Build the map, or return the cached one if the corpus has not changed."""
        key = (collection, method, k, max_points, include_retrievals)
        version = self.kb.store._write_counter
        cached = self._cache.get(key)
        if cached is not None and cached[0] == version:
            return cached[1]

        result = self._build(
            collection,
            method=method,
            k=k,
            max_points=max_points,
            include_retrievals=include_retrievals,
        )
        self._cache[key] = (version, result)
        return result

    # ------------------------------------------------------------------ #

    def _build(
        self,
        collection: str,
        *,
        method: str,
        k: int | None,
        max_points: int,
        include_retrievals: bool,
    ) -> CorpusMap:
        started = time.perf_counter()
        store = self.kb.store
        model = self.kb.embedder.model

        matrix, chunk_ids = store.vector_matrix(collection, model=model)
        if matrix.size == 0 or not chunk_ids:
            return CorpusMap(
                collection=collection,
                points=[],
                clusters=[],
                method="none",
                n_chunks=0,
                n_plotted=0,
                sampled=False,
                notes=["no embedded chunks in this collection — run `kb embed` first"],
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        total = len(chunk_ids)
        indices = _sample_indices(total, max_points)
        sampled = len(indices) < total
        selected_ids = [chunk_ids[i] for i in indices]
        selected_matrix = matrix[indices]

        chunks = {c.id: c for c in store.get_chunks(selected_ids)}
        texts = [
            (chunks[cid].heading_context + "\n" + chunks[cid].text) if cid in chunks else ""
            for cid in selected_ids
        ]

        projection = project(selected_matrix, method=method)  # type: ignore[arg-type]
        coordinates = normalize_to_unit_square(projection.coordinates)
        clusters = cluster_corpus(selected_matrix, texts, k=k)

        cluster_of: dict[int, int] = {}
        for cluster in clusters:
            for member in cluster.member_indices:
                cluster_of[member] = cluster.id

        retrievals: dict[str, int] = {}
        if include_retrievals:
            retrievals = {
                row["chunk_id"]: int(row["hits"])
                for row in store.retrieval_heatmap(collection, limit=100_000)
            }

        points: list[MapPoint] = []
        for position, chunk_id in enumerate(selected_ids):
            chunk = chunks.get(chunk_id)
            if chunk is None:
                continue
            body = " ".join(chunk.text.split())
            points.append(
                MapPoint(
                    chunk_id=chunk_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    x=round(float(coordinates[position, 0]), 5),
                    y=round(float(coordinates[position, 1]), 5),
                    cluster=cluster_of.get(position, 0),
                    source_type=chunk.source_type.value if chunk.source_type else None,
                    kind=chunk.kind.value,
                    snippet=body[:SNIPPET_CHARS] + ("…" if len(body) > SNIPPET_CHARS else ""),
                    heading=chunk.heading_context,
                    tokens=chunk.token_estimate,
                    retrievals=retrievals.get(chunk_id, 0),
                )
            )

        notes = list(projection.notes)
        if sampled:
            notes.append(f"sampled {len(points)} of {total} chunks to keep the map legible")

        return CorpusMap(
            collection=collection,
            points=points,
            clusters=[_cluster_payload(c, selected_ids, retrievals) for c in clusters],
            method=projection.method,
            n_chunks=total,
            n_plotted=len(points),
            sampled=sampled,
            explained_variance=projection.explained_variance,
            notes=notes,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )


def _cluster_payload(
    cluster: Cluster, chunk_ids: list[str], retrievals: dict[str, int]
) -> dict[str, Any]:
    member_ids = [chunk_ids[i] for i in cluster.member_indices if i < len(chunk_ids)]
    retrieved = sum(1 for cid in member_ids if retrievals.get(cid, 0) > 0)
    return {
        "id": cluster.id,
        "label": cluster.label,
        "terms": cluster.terms,
        "size": cluster.size,
        "coherence": cluster.coherence,
        "retrieved_share": round(retrieved / cluster.size, 4) if cluster.size else 0.0,
    }


def _sample_indices(total: int, max_points: int) -> list[int]:
    """Evenly-spaced sample, not random.

    Even spacing over the store's ordering keeps the sample stable between calls
    (so the map does not reshuffle on reload) and spreads it across documents,
    since chunks are stored in ingestion order.
    """
    if total <= max_points:
        return list(range(total))
    step = total / max_points
    return [int(i * step) for i in range(max_points)]


def document_graph(
    knowledge_base: Any,
    collection: str = "default",
    *,
    max_documents: int = 120,
    min_similarity: float = 0.35,
    max_edges_per_document: int = 6,
) -> dict[str, Any]:
    """A similarity graph over documents.

    Each document is represented by the mean of its chunk vectors, and an edge is
    drawn when two documents are similar above a threshold. This answers a
    question the scatter plot cannot: *which documents overlap* — near-duplicates,
    a doc and its changelog, the same runbook exported twice. Those are the
    documents that compete for the same top-k slot, so finding them is
    actionable.

    Edges are capped per document, because a dense corpus otherwise produces a
    complete graph that renders as a solid disc.
    """
    store = knowledge_base.store
    model = knowledge_base.embedder.model
    matrix, chunk_ids = store.vector_matrix(collection, model=model)
    if matrix.size == 0:
        return {"collection": collection, "nodes": [], "edges": [], "notes": ["no vectors"]}

    metadata = store.chunk_metadata_map(collection)
    by_document: dict[str, list[int]] = {}
    for position, chunk_id in enumerate(chunk_ids):
        info = metadata.get(chunk_id)
        if info:
            by_document.setdefault(info["document_id"], []).append(position)

    documents = store.list_documents(collection, limit=max_documents * 4)
    titles = {d.id: d.title for d in documents}
    source_types = {d.id: d.source_type.value for d in documents}

    # Largest documents first: they carry the most of the corpus.
    ordered = sorted((d for d in by_document if d in titles), key=lambda d: -len(by_document[d]))[
        :max_documents
    ]
    if not ordered:
        return {"collection": collection, "nodes": [], "edges": [], "notes": ["no documents"]}

    centroids = np.vstack([matrix[by_document[d]].mean(axis=0) for d in ordered])
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(norms, 1e-12)
    similarity = centroids @ centroids.T
    np.fill_diagonal(similarity, -1.0)

    nodes = [
        {
            "id": document_id,
            "title": titles[document_id],
            "source_type": source_types.get(document_id),
            "n_chunks": len(by_document[document_id]),
        }
        for document_id in ordered
    ]

    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for i, document_id in enumerate(ordered):
        row = similarity[i]
        candidates = np.argsort(-row)[:max_edges_per_document]
        for j in candidates:
            weight = float(row[j])
            if weight < min_similarity:
                continue
            pair = tuple(sorted((document_id, ordered[j])))
            if pair in seen:
                continue
            seen.add(pair)  # type: ignore[arg-type]
            edges.append({"source": pair[0], "target": pair[1], "weight": round(weight, 4)})

    edges.sort(key=lambda e: -e["weight"])
    return {
        "collection": collection,
        "nodes": nodes,
        "edges": edges,
        "notes": [] if edges else [f"no document pairs above {min_similarity} similarity"],
    }
