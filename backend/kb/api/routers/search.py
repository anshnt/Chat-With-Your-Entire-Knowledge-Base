"""Search endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from kb.api.deps import get_knowledge_base
from kb.api.schemas import SearchHit, SearchRequest, SearchResponse
from kb.knowledge_base import KnowledgeBase
from kb.models import RetrievalResult
from kb.retrieval.hybrid import request_from_settings

router = APIRouter(prefix="/api", tags=["search"])


def _to_response(result: RetrievalResult) -> SearchResponse:
    return SearchResponse(
        query=result.query,
        hits=[SearchHit.from_scored(r) for r in result.results],
        strategy=result.strategy,
        fusion=result.fusion,
        reranked=result.reranked,
        lexical_candidates=result.lexical_candidates,
        dense_candidates=result.dense_candidates,
        fused_candidates=result.fused_candidates,
        timings_ms=result.timings_ms,
        total_ms=result.total_ms(),
    )


@router.post("/search", response_model=SearchResponse, summary="Hybrid retrieval")
def search(
    payload: SearchRequest, kb: KnowledgeBase = Depends(get_knowledge_base)
) -> SearchResponse:
    """Retrieve chunks for a query.

    Returns every stage's score alongside the final ranking, so a client can
    show why a result is present rather than only that it is.
    """
    request = request_from_settings(
        kb.settings,
        payload.query,
        top_k=payload.top_k,
        candidate_k=payload.candidate_k,
        strategy=payload.strategy,
        fusion=payload.fusion,
        lexical_weight=payload.lexical_weight,
        dense_weight=payload.dense_weight,
        rerank=payload.rerank,
        use_mmr=payload.use_mmr,
        mmr_lambda=payload.mmr_lambda,
        source_types=payload.source_types,
        document_ids=payload.document_ids,
    )
    request = request.model_copy(update={"collection": payload.collection})
    return _to_response(kb.retrieve(request))


@router.get("/search", response_model=SearchResponse, summary="Hybrid retrieval (GET)")
def search_get(
    q: str = Query(min_length=1, max_length=4000, description="Query text"),
    collection: str = "default",
    top_k: int | None = Query(default=None, ge=1, le=100),
    kb: KnowledgeBase = Depends(get_knowledge_base),
) -> SearchResponse:
    """Convenience GET form for quick manual checks and shareable links."""
    return search(
        SearchRequest(query=q, collection=collection, top_k=top_k),
        kb=kb,
    )


@router.get(
    "/chunks/{chunk_id}/similar",
    response_model=list[SearchHit],
    summary="Chunks similar to a given chunk",
)
def similar(
    chunk_id: str,
    collection: str = "default",
    limit: int = Query(default=10, ge=1, le=50),
    kb: KnowledgeBase = Depends(get_knowledge_base),
) -> list[SearchHit]:
    """ "More like this" over the dense index."""
    return [
        SearchHit.from_scored(s)
        for s in kb.similar_chunks(chunk_id, collection=collection, limit=limit)
    ]
