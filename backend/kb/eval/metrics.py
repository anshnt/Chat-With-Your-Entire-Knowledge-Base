"""Retrieval metrics.

Implemented from their definitions rather than pulled from a library, because the
edge cases are where evaluation harnesses quietly lie:

* **A query with no relevant documents** must be *excluded*, not scored 0. Scoring
  it zero drags the mean down by an amount that depends on how many unanswerable
  questions happen to be in the set, which makes runs incomparable.
* **Recall@k must divide by the number of relevant documents, capped at k.**
  Dividing by the raw count means a query with 20 relevant chunks can never
  exceed 0.4 at k=8, and the metric measures the golden set's shape rather than
  the retriever.
* **nDCG must normalise against the *achievable* ideal at k**, not against a
  perfect ranking of unlimited length, or the ceiling moves with the golden set.
* **Graded relevance is supported** (2 = directly answers, 1 = related), because
  binary relevance cannot express "this chunk is adjacent to the answer" — which
  is most of what a retriever gets wrong.

Every metric takes ``ranked_ids`` (best-first) and a relevance mapping, and
returns a float in [0, 1] — except MAP and MRR, which are also in [0, 1] but are
computed over the full ranking rather than a cutoff.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

#: Relevance grade at or above which a document counts as relevant for the
#: binary metrics (recall, precision, MRR, hit rate).
BINARY_THRESHOLD = 1


def _relevant_ids(relevance: Mapping[str, int], threshold: int = BINARY_THRESHOLD) -> set[str]:
    return {doc_id for doc_id, grade in relevance.items() if grade >= threshold}


def recall_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    """Share of the relevant documents that appear in the top ``k``.

    The denominator is ``min(len(relevant), k)``: a query with 20 relevant chunks
    cannot put more than ``k`` of them in the top ``k``, and dividing by 20 would
    cap the score at ``k/20`` no matter how good the retriever is.
    """
    relevant = _relevant_ids(relevance)
    if not relevant:
        return float("nan")
    top = ranked_ids[:k]
    found = sum(1 for doc_id in top if doc_id in relevant)
    return found / min(len(relevant), k)


def precision_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    """Share of the top ``k`` results that are relevant.

    Divides by ``k``, not by the number of results returned: a retriever that
    returns 2 results, both relevant, has not achieved precision 1.0 at k=8 — it
    has failed to fill the context window.
    """
    relevant = _relevant_ids(relevance)
    if not relevant:
        return float("nan")
    top = ranked_ids[:k]
    return sum(1 for doc_id in top if doc_id in relevant) / k


def hit_rate_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    """1.0 if any relevant document is in the top ``k``.

    The floor for RAG: below this, no amount of generation quality helps, because
    the answer is not in the context.
    """
    relevant = _relevant_ids(relevance)
    if not relevant:
        return float("nan")
    return 1.0 if any(doc_id in relevant for doc_id in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: Sequence[str], relevance: Mapping[str, int]) -> float:
    """1/rank of the first relevant document, or 0 if none is retrieved.

    Sensitive only to the first hit, which is the right metric when one good
    chunk is enough — and the wrong one when the answer needs several.
    """
    relevant = _relevant_ids(relevance)
    if not relevant:
        return float("nan")
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(ranked_ids: Sequence[str], relevance: Mapping[str, int]) -> float:
    """Mean of the precisions at each relevant hit.

    Rewards finding *all* the relevant chunks early, so unlike MRR it
    distinguishes a retriever that finds one good chunk from one that finds four.
    """
    relevant = _relevant_ids(relevance)
    if not relevant:
        return float("nan")
    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / rank
    return total / min(len(relevant), len(ranked_ids)) if hits else 0.0


def dcg_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    """Discounted cumulative gain with the standard ``2^grade - 1`` gain."""
    return sum(
        (2 ** relevance.get(doc_id, 0) - 1) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_ids[:k], start=1)
    )


def ndcg_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    """nDCG@k, normalised against the best ranking *achievable at k*.

    Using the achievable ideal rather than a perfect unbounded ranking keeps the
    ceiling at 1.0 regardless of how many relevant chunks the golden set happens
    to list for a query.
    """
    if not _relevant_ids(relevance):
        return float("nan")
    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    ideal = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
        if grade > 0
    )
    if ideal <= 0:
        return float("nan")
    return dcg_at_k(ranked_ids, relevance, k) / ideal


def first_relevant_rank(ranked_ids: Sequence[str], relevance: Mapping[str, int]) -> int | None:
    """1-based rank of the first relevant document, or ``None``.

    Reported alongside the aggregates because "the answer was at rank 14" is
    diagnosable in a way that "MRR fell by 0.03" is not.
    """
    relevant = _relevant_ids(relevance)
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return rank
    return None
