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
    Answer,
    AnswerCitation,
    AnswerSentence,
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
            label=chunk.position_label(),
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
    generator: str = ""
    generation_model: str = ""
    verifier: str | None = None
    connectors: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Answering
# --------------------------------------------------------------------------- #


class AskRequest(BaseModel):
    """A question, plus optional retrieval overrides.

    Every retrieval knob is exposed here on purpose: comparing answers under
    different retrieval settings is how you find out whether a disappointing
    answer is a generation problem or a retrieval one.
    """

    query: str = Field(min_length=1, max_length=4000)
    collection: str = "default"
    top_k: int | None = Field(default=None, ge=1, le=50)
    candidate_k: int | None = Field(default=None, ge=1, le=500)
    strategy: RetrievalStrategy | None = None
    fusion: FusionMethod | None = None
    rerank: bool | None = None
    use_mmr: bool | None = None
    source_types: list[SourceType] | None = None
    document_ids: list[str] | None = None
    include_retrieval: bool = Field(
        default=True, description="Include the full retrieval diagnostics in the response"
    )
    verify: bool | None = Field(
        default=None,
        description="Verify citations. Defaults to the server setting; false skips the stage.",
    )


class AskResponse(BaseModel):
    """An answer, its citations, and the evidence for trusting it."""

    query: str
    answer: str
    citations: list[AnswerCitation] = Field(default_factory=list)
    sentences: list[AnswerSentence] = Field(default_factory=list)
    generator: str
    model: str
    refused: bool = False
    verified: bool = False
    faithfulness: float | None = None
    unsupported_count: int = 0
    flagged_count: int = Field(
        default=0, description="Unsupported, uncited, or only partially supported claims"
    )
    context_chunks: int = 0
    context_tokens: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    total_ms: float = 0.0
    retrieval: SearchResponse | None = None

    @classmethod
    def from_answer(cls, answer: Answer, *, include_retrieval: bool = True) -> AskResponse:
        retrieval: SearchResponse | None = None
        if include_retrieval and answer.retrieval is not None:
            result = answer.retrieval
            retrieval = SearchResponse(
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
        return cls(
            query=answer.query,
            answer=answer.text,
            citations=answer.citations,
            sentences=answer.sentences,
            generator=answer.generator,
            model=answer.model,
            refused=answer.refused,
            verified=answer.verified,
            faithfulness=answer.faithfulness,
            unsupported_count=len(answer.unsupported_sentences()),
            flagged_count=len(answer.flagged_sentences()),
            context_chunks=answer.context_chunks,
            context_tokens=answer.context_tokens,
            timings_ms=answer.timings_ms,
            total_ms=answer.total_ms(),
            retrieval=retrieval,
        )


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #


class MapPointView(BaseModel):
    """One chunk, placed on the corpus map."""

    chunk_id: str
    document_id: str
    document_title: str
    x: float = Field(description="Normalised to [0, 1]")
    y: float = Field(description="Normalised to [0, 1]")
    cluster: int
    source_type: str | None = None
    kind: str
    snippet: str
    heading: str = ""
    tokens: int = 0
    retrievals: int = Field(
        default=0, description="How often this chunk has been retrieved; 0 means never"
    )


class MapClusterView(BaseModel):
    """A cluster and what distinguishes it."""

    id: int
    label: str
    terms: list[str] = Field(default_factory=list)
    size: int
    coherence: float = Field(
        default=0.0,
        description="Mean similarity of members to the centroid; low means the label is unreliable",
    )
    retrieved_share: float = 0.0


class CorpusMapResponse(BaseModel):
    collection: str
    method: str = Field(description="The projection that actually ran")
    n_chunks: int
    n_plotted: int
    sampled: bool = False
    explained_variance: float | None = Field(
        default=None,
        description="PCA only: share of variance the two axes capture. Low means the plot is misleading.",
    )
    retrieval_coverage: float = 0.0
    notes: list[str] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    clusters: list[MapClusterView] = Field(default_factory=list)
    points: list[MapPointView] = Field(default_factory=list)


class GraphNode(BaseModel):
    id: str
    title: str
    source_type: str | None = None
    n_chunks: int = 0


class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float


class DocumentGraphResponse(BaseModel):
    collection: str
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CoverageResponse(BaseModel):
    collection: str
    n_chunks: int
    n_retrieved: int
    coverage: float = Field(description="Share of chunks retrieved at least once")
    n_queries_logged: int = 0
    most_retrieved: list[dict[str, Any]] = Field(default_factory=list)
    never_retrieved: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Chunks no query has ever reached: redundant, or unreachable",
    )
