"""Combining lexical and dense rankings.

Reciprocal Rank Fusion is the default because it consumes *ranks*, not scores.
BM25 scores are unbounded and corpus-dependent while cosine scores live in
[-1, 1]; any weighted sum of the two needs per-query normalisation, and min-max
normalisation over a truncated candidate list is unstable — a single outlier at
rank 1 compresses everything below it. RRF sidesteps all of that and, in the
original TREC work, beat the individual systems it combined without tuning.

The weighted and max methods are kept because they are the right answer when the
two score scales *are* comparable, and because having all three makes the
evaluation harness able to demonstrate the difference rather than assert it.
"""

from __future__ import annotations

from collections.abc import Sequence

from kb.models import FusionMethod, ScoredChunk


def _merge_provenance(target: ScoredChunk, source: ScoredChunk) -> None:
    """Copy per-retriever scores from ``source`` onto ``target``."""
    if source.lexical_score is not None:
        target.lexical_score = source.lexical_score
        target.lexical_rank = source.lexical_rank
    if source.dense_score is not None:
        target.dense_score = source.dense_score
        target.dense_rank = source.dense_rank
    for retriever in source.retrievers:
        if retriever not in target.retrievers:
            target.retrievers.append(retriever)


def _index(results: Sequence[ScoredChunk]) -> dict[str, ScoredChunk]:
    return {r.chunk.id: r for r in results}


def _minmax(values: Sequence[float]) -> list[float]:
    """Scale to [0, 1]; an all-equal list maps to all-1.0, not a divide by zero."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    span = hi - lo
    if span < 1e-12:
        return [1.0] * len(values)
    return [(v - lo) / span for v in values]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[ScoredChunk]:
    """Fuse ranked lists as ``sum_i w_i / (k + rank_i)``.

    ``k`` damps the influence of top ranks; the conventional 60 means rank 1 and
    rank 2 differ by ~1.6%, so a single retriever's confident-but-wrong top hit
    cannot dominate a chunk that both retrievers agree on.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")

    fused: dict[str, ScoredChunk] = {}
    scores: dict[str, float] = {}

    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, result in enumerate(ranking, start=1):
            cid = result.chunk.id
            contribution = weight / (k + rank)
            scores[cid] = scores.get(cid, 0.0) + contribution
            existing = fused.get(cid)
            if existing is None:
                fused[cid] = result.model_copy(deep=True)
            else:
                _merge_provenance(existing, result)

    out: list[ScoredChunk] = []
    for cid, score in scores.items():
        item = fused[cid]
        item.fusion_score = score
        item.score = score
        out.append(item)
    out.sort(key=lambda r: (-r.score, r.chunk.id))
    return out


def weighted_fusion(
    lexical: Sequence[ScoredChunk],
    dense: Sequence[ScoredChunk],
    *,
    lexical_weight: float = 0.4,
    dense_weight: float = 0.6,
) -> list[ScoredChunk]:
    """Min-max normalise each list, then take the weighted sum.

    A chunk found by only one retriever keeps that retriever's contribution and
    scores zero for the other — which is the intended behaviour: agreement
    between retrievers should win.
    """
    lex_norm = dict(
        zip(
            [r.chunk.id for r in lexical],
            _minmax([r.lexical_score or r.score for r in lexical]),
            strict=True,
        )
    )
    dense_norm = dict(
        zip(
            [r.chunk.id for r in dense],
            _minmax([r.dense_score or r.score for r in dense]),
            strict=True,
        )
    )

    merged: dict[str, ScoredChunk] = {}
    for result in list(lexical) + list(dense):
        cid = result.chunk.id
        if cid in merged:
            _merge_provenance(merged[cid], result)
        else:
            merged[cid] = result.model_copy(deep=True)

    out: list[ScoredChunk] = []
    for cid, item in merged.items():
        score = lexical_weight * lex_norm.get(cid, 0.0) + dense_weight * dense_norm.get(cid, 0.0)
        item.fusion_score = score
        item.score = score
        out.append(item)
    out.sort(key=lambda r: (-r.score, r.chunk.id))
    return out


def max_fusion(lexical: Sequence[ScoredChunk], dense: Sequence[ScoredChunk]) -> list[ScoredChunk]:
    """Take the better normalised score per chunk. Highest recall, noisiest."""
    lex_norm = dict(
        zip(
            [r.chunk.id for r in lexical],
            _minmax([r.lexical_score or r.score for r in lexical]),
            strict=True,
        )
    )
    dense_norm = dict(
        zip(
            [r.chunk.id for r in dense],
            _minmax([r.dense_score or r.score for r in dense]),
            strict=True,
        )
    )
    merged: dict[str, ScoredChunk] = {}
    for result in list(lexical) + list(dense):
        cid = result.chunk.id
        if cid in merged:
            _merge_provenance(merged[cid], result)
        else:
            merged[cid] = result.model_copy(deep=True)

    out: list[ScoredChunk] = []
    for cid, item in merged.items():
        score = max(lex_norm.get(cid, 0.0), dense_norm.get(cid, 0.0))
        item.fusion_score = score
        item.score = score
        out.append(item)
    out.sort(key=lambda r: (-r.score, r.chunk.id))
    return out


def fuse(
    lexical: Sequence[ScoredChunk],
    dense: Sequence[ScoredChunk],
    *,
    method: FusionMethod = FusionMethod.RRF,
    rrf_k: int = 60,
    lexical_weight: float = 0.4,
    dense_weight: float = 0.6,
) -> list[ScoredChunk]:
    """Dispatch to the configured fusion method."""
    if method is FusionMethod.RRF:
        return reciprocal_rank_fusion(
            [lexical, dense], k=rrf_k, weights=[lexical_weight, dense_weight]
        )
    if method is FusionMethod.WEIGHTED:
        return weighted_fusion(
            lexical, dense, lexical_weight=lexical_weight, dense_weight=dense_weight
        )
    if method is FusionMethod.MAX:
        return max_fusion(lexical, dense)
    raise ValueError(f"unknown fusion method: {method}")
