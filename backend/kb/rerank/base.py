"""Reranker contract.

Fusion optimises *recall* at `candidate_k`; the generator only ever sees
`top_k`. Reranking is the stage that converts that recall into precision, and it
is the highest-leverage single addition to a naive RAG pipeline — a cross-encoder
reads the query and the passage **together**, so it can judge relevance in ways
that comparing two independently-computed embeddings structurally cannot.

The cost is that it cannot run over a corpus: scoring is O(candidates) model
calls. That is exactly why it sits after fusion, over tens of candidates rather
than tens of thousands of chunks.
"""

from __future__ import annotations

import abc
import logging
from collections.abc import Sequence

from kb.models import ScoredChunk

log = logging.getLogger(__name__)


class Reranker(abc.ABC):
    """Second-stage scorer over fused retrieval candidates."""

    #: Stable identifier, reported by ``/api/health`` and stored in eval runs.
    name: str = "reranker"

    @abc.abstractmethod
    def score(self, query: str, candidates: Sequence[ScoredChunk]) -> list[float]:
        """Relevance of each candidate to ``query``. Higher is better.

        Implementations return one score per candidate, in the same order.
        """

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], *, top_n: int | None = None
    ) -> list[ScoredChunk]:
        """Reorder ``candidates`` by reranker score.

        The first-stage ``fusion_score`` is kept for provenance and used as the
        tie-break, so a reranker that cannot separate two passages leaves the
        retrieval order intact rather than shuffling it arbitrarily.
        """
        if not candidates:
            return []
        try:
            scores = self.score(query, candidates)
        except Exception as exc:
            # Degrading to the fused order is strictly better than failing the
            # query: the candidates are already relevant, just less well sorted.
            log.warning("reranker %s failed (%s); falling back to fused order", self.name, exc)
            return list(candidates[: top_n or len(candidates)])

        if len(scores) != len(candidates):
            log.warning(
                "reranker %s returned %d scores for %d candidates; keeping fused order",
                self.name,
                len(scores),
                len(candidates),
            )
            return list(candidates[: top_n or len(candidates)])

        reranked: list[ScoredChunk] = []
        for candidate, score in zip(candidates, scores, strict=True):
            item = candidate.model_copy(deep=True)
            item.rerank_score = float(score)
            if item.fusion_score is None:
                item.fusion_score = candidate.score
            item.score = float(score)
            if self.name not in item.retrievers:
                item.retrievers.append(self.name)
            reranked.append(item)

        reranked.sort(
            key=lambda r: (-(r.rerank_score or 0.0), -(r.fusion_score or 0.0), r.chunk.id)
        )
        return reranked[:top_n] if top_n else reranked

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r})"


class NoOpReranker(Reranker):
    """Passes candidates through untouched.

    Exists so that ``rerank_provider=none`` is a real object rather than a
    ``None`` special case threaded through the pipeline.
    """

    name = "none"

    def score(self, query: str, candidates: Sequence[ScoredChunk]) -> list[float]:  # noqa: ARG002
        return [c.score for c in candidates]

    def rerank(
        self,
        query: str,  # noqa: ARG002
        candidates: Sequence[ScoredChunk],
        *,
        top_n: int | None = None,
    ) -> list[ScoredChunk]:
        return list(candidates[: top_n or len(candidates)])
