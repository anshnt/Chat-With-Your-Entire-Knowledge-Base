"""Request and response models for the HTTP API.

These are deliberately separate from the domain models in :mod:`kb.models`.
The wire format is a contract with a frontend and can only change in
backward-compatible ways; the domain model should stay free to change. The main
thing the wire format adds is the *rendered* citation — label, deep link, and
snippet — so the client never has to know how a PDF page differs from a YouTube
timestamp.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from kb.models import (
    Chunk,
    CollectionStats,
    Document,
    FusionMethod,
    RetrievalStrategy,
    ScoredChunk,
    SourceType,
)

MAX_SNIPPET_CHARS = 400


class Citation(BaseModel):
    """A retrieved chunk, rendered for display."""

    chunk_id: str
    document_id: str
    document_title: str
    source_type: SourceType | None = None
    label: str = Field(description="e.g. 'p. 12' or 'Architecture › Fusion'")
    deep_link: str | None = Field(default=None, description="URL that jumps to this position")
    locator: dict[str, Any]
    snippet: str
    heading_context: str = ""

    @classmethod
    def from_chunk(cls, chunk: Chunk, *, snippet_chars: int = MAX_SNIPPET_CHARS) -> Citation:
        text = chunk.text.strip()
        snippet = text if len(text) <= snippet_chars else f"{text[:snippet_chars].rstrip()}…"
        return cls(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_title=chunk.document_title,
            source_type=chunk.source_type,
            label=chunk.locator.label(),
            deep_link=chunk.deep_link(),
            locator=chunk.locator.model_dump(mode="json"),
            snippet=snippet,
            heading_context=chunk.heading_context,
        )


class ScoreBreakdown(BaseModel):
    """Per-stage scores, so the UI can show *why* a chunk ranked where it did."""

    final: float
    lexical: float | None = None
    dense: float | None = None
    lexical_rank: int | None = None
    dense_rank: int | None = None
    fusion: float | None = None
    rerank: float | None = None
    retrievers: list[str] = Field(default_factory=list)

    @classmethod
    def from_scored(cls, scored: ScoredChunk) -> ScoreBreakdown:
        return cls(
            final=scored.score,
            lexical=scored.lexical_score,
            dense=scored.dense_score,
            lexical_rank=scored.lexical_rank,
            dense_rank=scored.dense_rank,
            fusion=scored.fusion_score,
            rerank=scored.rerank_score,
            retrievers=list(scored.retrievers),
        )


class SearchHit(BaseModel):
    """One search result."""

    citation: Citation
    scores: ScoreBreakdown
    text: str

    @classmethod
    def from_scored(cls, scored: ScoredChunk) -> SearchHit:
        return cls(
            citation=Citation.from_chunk(scored.chunk),
            scores=ScoreBreakdown.from_scored(scored),
            text=scored.chunk.text,
        )


class SearchRequest(BaseModel):
    """Retrieval parameters. Anything omitted falls back to the server config."""

    query: str = Field(min_length=1, max_length=4000)
    collection: str = "default"
    top_k: int | None = Field(default=None, ge=1, le=100)
    candidate_k: int | None = Field(default=None, ge=1, le=500)
    strategy: RetrievalStrategy | None = None
    fusion: FusionMethod | None = None
    lexical_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    dense_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    rerank: bool | None = None
    use_mmr: bool | None = None
    mmr_lambda: float | None = Field(default=None, ge=0.0, le=1.0)
    source_types: list[SourceType] | None = None
    document_ids: list[str] | None = None


class SearchResponse(BaseModel):
    """Search results with the diagnostics needed to explain them."""

    query: str
    hits: list[SearchHit]
    strategy: RetrievalStrategy
    fusion: FusionMethod | None = None
    reranked: bool = False
    lexical_candidates: int = 0
    dense_candidates: int = 0
    fused_candidates: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    total_ms: float = 0.0


class IngestRequest(BaseModel):
    """Ingest a path/URL, or paste text directly via ``text``."""

    source: str | None = Field(default=None, description="File path, directory, glob, or URL")
    text: str | None = Field(default=None, description="Inline document body")
    title: str | None = None
    collection: str = "default"
    embed: bool = True
    options: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    documents_created: int
    chunks_created: int
    documents_skipped: int
    duplicates_skipped: int
    errors: list[dict[str, str]] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    documents: list[Document] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    documents: list[Document]
    total: int
    limit: int
    offset: int


class ChunkView(BaseModel):
    """A chunk as the source viewer needs it."""

    id: str
    document_id: str
    ordinal: int
    text: str
    kind: str
    citation: Citation
    token_estimate: int

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> ChunkView:
        return cls(
            id=chunk.id,
            document_id=chunk.document_id,
            ordinal=chunk.ordinal,
            text=chunk.text,
            kind=chunk.kind.value,
            citation=Citation.from_chunk(chunk, snippet_chars=160),
            token_estimate=chunk.token_estimate,
        )


class ChunkListResponse(BaseModel):
    document: Document
    chunks: list[ChunkView]


class ChunkContextResponse(BaseModel):
    """A chunk plus its neighbours, so a citation can be read in context."""

    focus_chunk_id: str
    chunks: list[ChunkView]


class StatsResponse(BaseModel):
    stats: CollectionStats
    collections: list[str] = Field(default_factory=list)


class HeatmapEntry(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    hits: int
    avg_rank: float
    avg_score: float
    last_seen: str | None = None


class HeatmapResponse(BaseModel):
    collection: str
    entries: list[HeatmapEntry]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    embedding_model: str
    embedding_dim: int
    retrieval_strategy: str
    reranker: str | None = None
    connectors: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
