"""The retrieval pipeline.

    query
      ├─ lexical (BM25 / FTS5) ──┐
      └─ dense (cosine)  ────────┼─→ fuse (RRF | weighted | max)
                                 │
                                 ├─→ rerank (optional, cross-encoder / listwise)
                                 ├─→ MMR diversify (optional)
                                 └─→ top-k

Two candidate lists of ``candidate_k`` each are fused, and only then truncated to
``top_k``. Doing it in that order is the whole point of hybrid retrieval: the
chunk that BM25 ranks 40th and the dense retriever ranks 35th is often the right
answer, and it is invisible to either retriever alone at k=8.

Every stage is timed and every per-retriever score is preserved on the way
through, so a bad result can be attributed to a stage instead of guessed at.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

from kb.config import Settings
from kb.embeddings.base import Embedder
from kb.models import (
    FusionMethod,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStrategy,
    ScoredChunk,
)
from kb.retrieval.dense import DenseRetriever
from kb.retrieval.fusion import fuse
from kb.retrieval.lexical import LexicalRetriever
from kb.retrieval.mmr import mmr_rerank
from kb.store import SQLiteStore


class Reranker(Protocol):
    """Second-stage scorer over fused candidates."""

    name: str

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], *, top_n: int
    ) -> list[ScoredChunk]:  # pragma: no cover - protocol
        ...


class Timer:
    """Accumulates per-stage wall time in milliseconds."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    def time(self, label: str):
        return _TimerContext(self, label)


class _TimerContext:
    def __init__(self, timer: Timer, label: str) -> None:
        self.timer = timer
        self.label = label
        self.start = 0.0

    def __enter__(self) -> _TimerContext:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        elapsed = (time.perf_counter() - self.start) * 1000.0
        self.timer.timings[self.label] = round(self.timer.timings.get(self.label, 0.0) + elapsed, 3)


class HybridRetriever:
    """Orchestrates lexical + dense retrieval, fusion, reranking and MMR."""

    def __init__(
        self,
        store: SQLiteStore,
        embedder: Embedder,
        *,
        reranker: Reranker | None = None,
        log_events: bool = True,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.lexical = LexicalRetriever(store)
        self.dense = DenseRetriever(store, embedder)
        self.reranker = reranker
        self.log_events = log_events

    # ------------------------------------------------------------------ #

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        timer = Timer()
        lexical_hits: list[ScoredChunk] = []
        dense_hits: list[ScoredChunk] = []

        wants_lexical = request.strategy in (RetrievalStrategy.LEXICAL, RetrievalStrategy.HYBRID)
        wants_dense = request.strategy in (RetrievalStrategy.DENSE, RetrievalStrategy.HYBRID)

        if wants_lexical:
            with timer.time("lexical_ms"):
                lexical_hits = self.lexical.search(
                    request.query,
                    collection=request.collection,
                    limit=request.candidate_k,
                    source_types=request.source_types,
                    document_ids=request.document_ids,
                )

        if wants_dense:
            with timer.time("dense_ms"):
                dense_hits = self.dense.search(
                    request.query,
                    collection=request.collection,
                    limit=request.candidate_k,
                    source_types=request.source_types,
                    document_ids=request.document_ids,
                )

        fusion_used: FusionMethod | None = None
        if wants_lexical and wants_dense:
            with timer.time("fusion_ms"):
                candidates = fuse(
                    lexical_hits,
                    dense_hits,
                    method=request.fusion,
                    rrf_k=request.rrf_k,
                    lexical_weight=request.lexical_weight,
                    dense_weight=request.dense_weight,
                )
            fusion_used = request.fusion
        else:
            candidates = list(lexical_hits or dense_hits)

        fused_count = len(candidates)

        reranked = False
        if request.rerank and self.reranker is not None and candidates:
            with timer.time("rerank_ms"):
                candidates = self.reranker.rerank(
                    request.query, candidates[: request.rerank_top_n], top_n=request.rerank_top_n
                )
            reranked = True

        if request.use_mmr and candidates:
            with timer.time("mmr_ms"):
                pool = candidates[: max(request.top_k * 4, request.top_k)]
                vectors = self.store.get_embeddings([c.chunk.id for c in pool])
                candidates = mmr_rerank(
                    pool,
                    top_k=request.top_k,
                    lambda_=request.mmr_lambda,
                    vectors=vectors or None,
                )

        results = candidates[: request.top_k]
        if request.min_score is not None:
            results = [r for r in results if r.score >= request.min_score]

        if self.log_events and results:
            with timer.time("log_ms"):
                self.store.log_retrieval(
                    request.query,
                    [(r.chunk.id, r.score) for r in results],
                    collection=request.collection,
                    strategy=request.strategy.value,
                )

        return RetrievalResult(
            query=request.query,
            results=results,
            strategy=request.strategy,
            fusion=fusion_used,
            reranked=reranked,
            lexical_candidates=len(lexical_hits),
            dense_candidates=len(dense_hits),
            fused_candidates=fused_count,
            timings_ms=timer.timings,
        )

    # ------------------------------------------------------------------ #

    def search(self, query: str, **kwargs: object) -> RetrievalResult:
        """Convenience wrapper: ``retriever.search("...", top_k=5)``."""
        return self.retrieve(RetrievalRequest(query=query, **kwargs))  # type: ignore[arg-type]


def request_from_settings(settings: Settings, query: str, **overrides: object) -> RetrievalRequest:
    """Build a :class:`RetrievalRequest` from settings, with explicit overrides.

    Keeps the CLI, API and evaluation harness honest about defaults: they all
    derive requests the same way, so an evaluation result describes the
    configuration the app actually runs.
    """
    payload: dict[str, object] = {
        "query": query,
        "top_k": settings.top_k,
        "candidate_k": settings.candidate_k,
        "strategy": settings.retrieval_strategy,
        "fusion": settings.fusion_method,
        "lexical_weight": settings.lexical_weight,
        "dense_weight": settings.dense_weight,
        "rrf_k": settings.rrf_k,
        "use_mmr": settings.use_mmr,
        "mmr_lambda": settings.mmr_lambda,
        "rerank": settings.rerank_enabled,
        "rerank_top_n": settings.rerank_top_n,
    }
    payload.update({k: v for k, v in overrides.items() if v is not None})
    return RetrievalRequest.model_validate(payload)
