"""Local cross-encoder reranker (sentence-transformers).

A cross-encoder concatenates the query and the passage into one sequence and runs
full attention across both. That is the entire advantage over a bi-encoder: the
model can condition on the query while reading the passage, so it resolves
"which of these two similar passages actually answers *this*" — a distinction
that is unavailable to any method comparing two independently-computed vectors.

`ms-marco-MiniLM-L-6-v2` is the default because it is ~90 MB and runs on CPU at a
few hundred pairs per second, which is the right operating point for reranking 30
candidates inside a request.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from kb.errors import MissingDependencyError, ProviderError
from kb.models import ScoredChunk
from kb.rerank.base import Reranker

log = logging.getLogger(__name__)

#: Prevents a very long chunk from being truncated so hard that the query's
#: answer falls outside the model's window.
DEFAULT_MAX_LENGTH = 512


class CrossEncoderReranker(Reranker):
    """Reranks with a local cross-encoder model."""

    name = "cross-encoder"

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        batch_size: int = 32,
        device: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MissingDependencyError("sentence-transformers", "local") from exc
        try:
            self._model = CrossEncoder(model, max_length=max_length, device=device)
        except Exception as exc:
            raise ProviderError(f"could not load cross-encoder {model!r}: {exc}") from exc
        self.name = f"cross-encoder:{model.rsplit('/', 1)[-1]}"
        self.model = model
        self.batch_size = batch_size

    def score(self, query: str, candidates: Sequence[ScoredChunk]) -> list[float]:
        if not candidates:
            return []
        pairs = [(query, _passage_text(c)) for c in candidates]
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]


def _passage_text(candidate: ScoredChunk) -> str:
    """What the cross-encoder actually reads.

    The heading path is prepended when it is not already in the text: the model
    has no other way to know that a chunk saying "It defaults to 60" is about
    RRF, and that context is exactly what makes the pair judgeable.
    """
    chunk = candidate.chunk
    if chunk.heading_context and chunk.heading_context not in chunk.text[:200]:
        return f"{chunk.heading_context}\n{chunk.text}"
    return chunk.text
