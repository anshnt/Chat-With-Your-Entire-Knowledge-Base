"""Corpus browsing, statistics, and retrieval telemetry."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from kb.api.deps import get_knowledge_base
from kb.api.schemas import (
    ChunkContextResponse,
    ChunkListResponse,
    ChunkView,
    DocumentListResponse,
    HeatmapEntry,
    HeatmapResponse,
    StatsResponse,
)
from kb.knowledge_base import KnowledgeBase
from kb.models import Document, SourceType

router = APIRouter(prefix="/api", tags=["corpus"])


@router.get("/documents", response_model=DocumentListResponse, summary="List documents")
def list_documents(
    collection: str = "default",
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    source_type: SourceType | None = None,
    search: str | None = Query(default=None, description="Substring match on title or URI"),
    kb: KnowledgeBase = Depends(get_knowledge_base),
) -> DocumentListResponse:
    documents = kb.documents(
        collection, limit=limit, offset=offset, source_type=source_type, search=search
    )
    return DocumentListResponse(
        documents=documents,
        total=kb.store.count_documents(collection),
        limit=limit,
        offset=offset,
    )


@router.get("/documents/{document_id}", response_model=Document, summary="Get one document")
def get_document(document_id: str, kb: KnowledgeBase = Depends(get_knowledge_base)) -> Document:
    return kb.document(document_id)


@router.get(
    "/documents/{document_id}/chunks",
    response_model=ChunkListResponse,
    summary="Inspect how a document was chunked",
)
def document_chunks(
    document_id: str, kb: KnowledgeBase = Depends(get_knowledge_base)
) -> ChunkListResponse:
    """Chunk boundaries are a retrieval-quality decision, so they are inspectable."""
    return ChunkListResponse(
        document=kb.document(document_id),
        chunks=[ChunkView.from_chunk(c) for c in kb.document_chunks(document_id)],
    )


@router.delete("/documents/{document_id}", response_model=dict, summary="Delete a document")
def delete_document(document_id: str, kb: KnowledgeBase = Depends(get_knowledge_base)) -> dict:
    kb.delete_document(document_id)
    return {"deleted": document_id}


@router.get(
    "/chunks/{chunk_id}/context",
    response_model=ChunkContextResponse,
    summary="A chunk plus its neighbours",
)
def chunk_context(
    chunk_id: str,
    window: int = Query(default=1, ge=0, le=5),
    kb: KnowledgeBase = Depends(get_knowledge_base),
) -> ChunkContextResponse:
    """Small chunks retrieve precisely but read badly; neighbours restore context."""
    chunks = kb.chunk_with_context(chunk_id, window=window)
    return ChunkContextResponse(
        focus_chunk_id=chunk_id, chunks=[ChunkView.from_chunk(c) for c in chunks]
    )


@router.get("/collections", response_model=list[str], summary="List collections")
def list_collections(kb: KnowledgeBase = Depends(get_knowledge_base)) -> list[str]:
    return kb.collections()


@router.get(
    "/collections/{collection}/stats", response_model=StatsResponse, summary="Corpus statistics"
)
def collection_stats(
    collection: str, kb: KnowledgeBase = Depends(get_knowledge_base)
) -> StatsResponse:
    return StatsResponse(stats=kb.stats(collection), collections=kb.collections())


@router.delete(
    "/collections/{collection}", response_model=dict, summary="Delete a whole collection"
)
def delete_collection(collection: str, kb: KnowledgeBase = Depends(get_knowledge_base)) -> dict:
    removed = kb.delete_collection(collection)
    return {"deleted_documents": removed, "collection": collection}


@router.get(
    "/collections/{collection}/heatmap",
    response_model=HeatmapResponse,
    summary="Which chunks actually get retrieved",
)
def heatmap(
    collection: str,
    limit: int = Query(default=200, ge=1, le=2000),
    kb: KnowledgeBase = Depends(get_knowledge_base),
) -> HeatmapResponse:
    """Retrieval counts per chunk.

    Chunks that never appear are either redundant or unreachable — both are
    actionable, and neither is visible without logging retrievals.
    """
    entries = [
        HeatmapEntry(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            document_title=row["document_title"],
            hits=row["hits"],
            avg_rank=round(float(row["avg_rank"]), 3),
            avg_score=round(float(row["avg_score"]), 6),
            last_seen=row["last_seen"],
        )
        for row in kb.heatmap(collection, limit=limit)
    ]
    return HeatmapResponse(collection=collection, entries=entries)


@router.get(
    "/collections/{collection}/queries",
    response_model=list[str],
    summary="Recently issued queries",
)
def recent_queries(
    collection: str,
    limit: int = Query(default=50, ge=1, le=500),
    kb: KnowledgeBase = Depends(get_knowledge_base),
) -> list[str]:
    """Real traffic is the best source of evaluation questions."""
    return kb.store.recent_queries(collection, limit=limit)
