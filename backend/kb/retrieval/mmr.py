"""Maximal Marginal Relevance — diversification of the final result set.

Top-k by relevance alone has a specific failure mode that hurts RAG badly: the
k best chunks are often k near-copies. Overlapping chunks, a doc and its changelog,
the same paragraph in two exports — all rank together, and the generator ends up
with one fact repeated k times instead of k facts. MMR trades a little relevance
for coverage:

    MMR = argmax_{d ∉ S} [ λ · rel(d, q) − (1 − λ) · max_{s ∈ S} sim(d, s) ]

λ = 1 is plain relevance; λ ≈ 0.7 is a good default for question answering, where
one strongly-relevant duplicate is still worth less than a second angle on the
question.

Similarity uses stored vectors when available and falls back to token Jaccard,
so diversification still works on a collection that has not been embedded.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from kb.chunking.base import tokenize_words
from kb.models import ScoredChunk


def _pairwise_similarity(
    results: Sequence[ScoredChunk], vectors: dict[str, np.ndarray] | None
) -> np.ndarray:
    n = len(results)
    if vectors:
        dim = next(iter(vectors.values())).shape[0]
        matrix = np.zeros((n, dim), dtype="float32")
        have = np.zeros(n, dtype=bool)
        for i, result in enumerate(results):
            vec = vectors.get(result.chunk.id)
            if vec is not None:
                norm = float(np.linalg.norm(vec))
                matrix[i] = vec / norm if norm > 1e-12 else vec
                have[i] = True
        sim = matrix @ matrix.T
        if have.all():
            return np.clip(sim, 0.0, 1.0)
        # Fill rows without a vector using the lexical fallback.
        token_sim = _jaccard_matrix(results)
        missing = ~have
        sim[missing, :] = token_sim[missing, :]
        sim[:, missing] = token_sim[:, missing]
        return np.clip(sim, 0.0, 1.0)
    return _jaccard_matrix(results)


def _jaccard_matrix(results: Sequence[ScoredChunk]) -> np.ndarray:
    token_sets = [set(tokenize_words(r.chunk.text)) for r in results]
    n = len(results)
    sim = np.eye(n, dtype="float32")
    for i in range(n):
        for j in range(i + 1, n):
            a, b = token_sets[i], token_sets[j]
            union = len(a | b)
            value = (len(a & b) / union) if union else 0.0
            sim[i, j] = sim[j, i] = value
    return sim


def mmr_rerank(
    results: Sequence[ScoredChunk],
    *,
    top_k: int,
    lambda_: float = 0.7,
    vectors: dict[str, np.ndarray] | None = None,
) -> list[ScoredChunk]:
    """Select ``top_k`` results balancing relevance against redundancy.

    Relevance is min-max normalised over the candidate set so that λ means the
    same thing regardless of which fusion method produced the scores.
    """
    if not results:
        return []
    if top_k >= len(results) and lambda_ >= 1.0:
        return list(results)

    scores = np.array([r.score for r in results], dtype="float32")
    lo, hi = float(scores.min()), float(scores.max())
    relevance = (scores - lo) / (hi - lo) if hi - lo > 1e-12 else np.ones_like(scores)

    sim = _pairwise_similarity(results, vectors)

    selected: list[int] = []
    remaining = set(range(len(results)))
    limit = min(top_k, len(results))

    while len(selected) < limit and remaining:
        best_idx, best_value = -1, -np.inf
        for idx in remaining:
            redundancy = max((sim[idx, s] for s in selected), default=0.0)
            value = lambda_ * relevance[idx] - (1.0 - lambda_) * redundancy
            if value > best_value:
                best_idx, best_value = idx, value
        selected.append(best_idx)
        remaining.discard(best_idx)

    return [results[i] for i in selected]
