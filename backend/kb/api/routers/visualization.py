"""Corpus visualisation endpoints.

Three views, each answering a question the others cannot:

* ``/map`` — where every chunk sits, which cluster it belongs to, and how often it
  has actually been retrieved. The last column is the one that matters: a cluster
  nobody's queries ever reach is either redundant or unreachable.
* ``/graph`` — which *documents* overlap. Near-duplicates compete for the same
  top-k slot, so finding them is actionable in a way a scatter plot is not.
* ``/coverage`` — the retrieval heatmap as a summary: how much of the corpus is
  doing any work at all.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query

from kb.api.deps import get_knowledge_base
from kb.api.schemas import (
    CorpusMapResponse,
    CoverageResponse,
    DocumentGraphResponse,
    MapClusterView,
    MapPointView,
)
from kb.knowledge_base import KnowledgeBase
from kb.viz import CorpusMapBuilder, document_graph

router = APIRouter(prefix="/api", tags=["visualization"])

#: One builder per process. The map is a whole-corpus operation, so it is cached
#: against the store's write counter rather than recomputed per request.
_builders: dict[int, CorpusMapBuilder] = {}


def _builder(knowledge_base: KnowledgeBase) -> CorpusMapBuilder:
    key = id(knowledge_base)
    builder = _builders.get(key)
    if builder is None:
        builder = CorpusMapBuilder(knowledge_base)
        _builders[key] = builder
    return builder


@router.get(
    "/collections/{collection}/map",
    response_model=CorpusMapResponse,
    summary="2D map of the corpus, clustered and labelled",
)
def corpus_map(
    collection: str,
    method: Literal["auto", "umap", "tsne", "pca"] = Query(
        default="auto", description="Projection method; auto prefers UMAP, then t-SNE, then PCA"
    ),
    clusters: int | None = Query(
        default=None, ge=1, le=40, description="Cluster count; omit to choose one from corpus size"
    ),
    max_points: int = Query(default=2000, ge=10, le=20_000),
    include_retrievals: bool = True,
    kb: KnowledgeBase = Depends(get_knowledge_base),
) -> CorpusMapResponse:
    """Project every embedded chunk to 2D, cluster it, and label each cluster.

    ``explained_variance`` is reported for PCA and is the honest caveat to show
    alongside the plot: a low value means the two axes capture little of the real
    structure and the picture should not be over-read.
    """
    result = _builder(kb).build(
        collection,
        method=method,
        k=clusters,
        max_points=max_points,
        include_retrievals=include_retrievals,
    )
    return CorpusMapResponse(
        collection=result.collection,
        method=result.method,
        n_chunks=result.n_chunks,
        n_plotted=result.n_plotted,
        sampled=result.sampled,
        explained_variance=result.explained_variance,
        retrieval_coverage=result.coverage(),
        notes=result.notes,
        elapsed_ms=result.elapsed_ms,
        clusters=[MapClusterView.model_validate(c) for c in result.clusters],
        points=[
            MapPointView(
                chunk_id=p.chunk_id,
                document_id=p.document_id,
                document_title=p.document_title,
                x=p.x,
                y=p.y,
                cluster=p.cluster,
                source_type=p.source_type,
                kind=p.kind,
                snippet=p.snippet,
                heading=p.heading,
                tokens=p.tokens,
                retrievals=p.retrievals,
            )
            for p in result.points
        ],
    )


@router.get(
    "/collections/{collection}/graph",
    response_model=DocumentGraphResponse,
    summary="Document similarity graph",
)
def graph(
    collection: str,
    max_documents: int = Query(default=120, ge=2, le=1000),
    min_similarity: float = Query(default=0.35, ge=0.0, le=1.0),
    max_edges_per_document: int = Query(default=6, ge=1, le=50),
    kb: KnowledgeBase = Depends(get_knowledge_base),
) -> DocumentGraphResponse:
    """Which documents overlap.

    Each document is the mean of its chunk vectors. Edges are capped per document
    because a dense corpus otherwise produces a complete graph, which renders as
    a solid disc and says nothing.
    """
    payload = document_graph(
        kb,
        collection,
        max_documents=max_documents,
        min_similarity=min_similarity,
        max_edges_per_document=max_edges_per_document,
    )
    return DocumentGraphResponse.model_validate(payload)


@router.get(
    "/collections/{collection}/coverage",
    response_model=CoverageResponse,
    summary="How much of the corpus is ever retrieved",
)
def coverage(
    collection: str,
    limit: int = Query(default=25, ge=1, le=500, description="Rows in each list"),
    kb: KnowledgeBase = Depends(get_knowledge_base),
) -> CoverageResponse:
    """Retrieval coverage, plus the most- and never-retrieved chunks.

    The never-retrieved list is the actionable half: those chunks are either
    redundant, or unreachable because nothing in them matches how people ask.
    """
    stats = kb.stats(collection)
    heatmap = kb.heatmap(collection, limit=100_000)
    retrieved_ids = {row["chunk_id"] for row in heatmap}

    never: list[dict] = []
    for batch in kb.store.iter_chunks(collection):
        for chunk in batch:
            if chunk.id in retrieved_ids:
                continue
            if len(never) >= limit:
                break
            body = " ".join(chunk.text.split())
            never.append(
                {
                    "chunk_id": chunk.id,
                    "document_title": chunk.document_title,
                    "label": chunk.position_label(),
                    "snippet": body[:160] + ("…" if len(body) > 160 else ""),
                }
            )
        if len(never) >= limit:
            break

    total = stats.n_chunks or 1
    return CoverageResponse(
        collection=collection,
        n_chunks=stats.n_chunks,
        n_retrieved=len(retrieved_ids),
        coverage=round(len(retrieved_ids) / total, 4),
        n_queries_logged=len(kb.store.recent_queries(collection, limit=100_000)),
        most_retrieved=[
            {
                "chunk_id": row["chunk_id"],
                "document_title": row["document_title"],
                "hits": row["hits"],
                "avg_rank": round(float(row["avg_rank"]), 2),
            }
            for row in heatmap[:limit]
        ],
        never_retrieved=never,
    )
