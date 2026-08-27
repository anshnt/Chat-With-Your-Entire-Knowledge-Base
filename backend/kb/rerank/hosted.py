"""Hosted reranking APIs (Cohere, Voyage).

Both expose the same shape — send a query and N documents, get back indices with
relevance scores — so they share a base class and differ only in the call. Both
are strictly better than the local MiniLM cross-encoder on quality, and strictly
worse on latency and cost; `kb eval` is how you decide which trade-off your
corpus warrants.

Providers return only the top ``top_n`` documents they were asked for, so any
candidate the API omits is assigned a score below every returned one rather than
being dropped — losing candidates silently would corrupt the recall the fusion
stage worked for.
"""

from __future__ import annotations

import abc
import os
import random
import time
from collections.abc import Sequence
from typing import Any

from kb.errors import MissingDependencyError, ProviderError
from kb.models import ScoredChunk
from kb.rerank.base import Reranker

_MAX_ATTEMPTS = 4
_RETRY_BASE_DELAY = 0.5

#: Hosted rerankers charge per token; there is no value in sending a 20k-char
#: chunk when the decision is made in the first few hundred words.
MAX_DOCUMENT_CHARS = 4000


class HostedReranker(Reranker):
    """Shared plumbing for rerank-as-a-service providers."""

    def __init__(self, model: str, *, top_n: int | None = None) -> None:
        self.model = model
        self.top_n = top_n

    @abc.abstractmethod
    def _call(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        """Return ``(index, relevance)`` pairs for the documents it ranked."""

    def score(self, query: str, candidates: Sequence[ScoredChunk]) -> list[float]:
        if not candidates:
            return []
        documents = [_truncate(_passage_text(c)) for c in candidates]
        top_n = min(self.top_n or len(candidates), len(candidates))
        ranked = _with_retries(
            lambda: self._call(query, documents, top_n), what=f"{self.name} rerank"
        )

        scores = [float("-inf")] * len(candidates)
        for index, relevance in ranked:
            if 0 <= index < len(scores):
                scores[index] = relevance

        # Candidates the provider did not return keep the fused order among
        # themselves, ranked below everything it did return.
        returned = [s for s in scores if s != float("-inf")]
        floor = min(returned) if returned else 0.0
        for i, score in enumerate(scores):
            if score == float("-inf"):
                scores[i] = floor - 1.0 - (i / max(len(scores), 1))
        return scores


class CohereReranker(HostedReranker):
    """Cohere Rerank."""

    name = "cohere-rerank"

    def __init__(
        self, model: str = "rerank-english-v3.0", api_key: str = "", top_n: int | None = None
    ) -> None:
        super().__init__(model, top_n=top_n)
        try:
            import cohere  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MissingDependencyError("cohere", "embeddings") from exc
        key = api_key or os.environ.get("COHERE_API_KEY", "")
        if not key:
            raise ProviderError("COHERE_API_KEY is not set")
        self._client = cohere.Client(api_key=key)
        self.name = f"cohere-rerank:{model}"

    def _call(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        response = self._client.rerank(
            query=query, documents=documents, model=self.model, top_n=top_n
        )
        return [(r.index, float(r.relevance_score)) for r in response.results]


class VoyageReranker(HostedReranker):
    """Voyage AI Rerank."""

    name = "voyage-rerank"

    def __init__(
        self, model: str = "rerank-2", api_key: str = "", top_n: int | None = None
    ) -> None:
        super().__init__(model, top_n=top_n)
        try:
            import voyageai  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MissingDependencyError("voyageai", "embeddings") from exc
        key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        if not key:
            raise ProviderError("VOYAGE_API_KEY is not set")
        self._client = voyageai.Client(api_key=key)
        self.name = f"voyage-rerank:{model}"

    def _call(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        response = self._client.rerank(
            query=query, documents=documents, model=self.model, top_k=top_n
        )
        return [(r.index, float(r.relevance_score)) for r in response.results]


def _passage_text(candidate: ScoredChunk) -> str:
    chunk = candidate.chunk
    if chunk.heading_context and chunk.heading_context not in chunk.text[:200]:
        return f"{chunk.heading_context}\n{chunk.text}"
    return chunk.text


def _truncate(text: str, limit: int = MAX_DOCUMENT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit]


def _with_retries(fn: Any, *, what: str) -> Any:
    last: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if not _is_transient(exc) or attempt == _MAX_ATTEMPTS - 1:
                break
            time.sleep(_RETRY_BASE_DELAY * (2**attempt) * (0.5 + random.random()))
    raise ProviderError(f"{what} failed: {last}") from last


def _is_transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        return status == 429 or 500 <= status < 600
    text = str(exc).lower()
    return any(
        s in text for s in ("timeout", "timed out", "connection", "429", "503", "overloaded")
    )
