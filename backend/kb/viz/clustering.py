"""Clustering and automatic cluster labelling.

A map of unlabelled dots is a screensaver. What makes it useful is knowing *what
each region is about*, which means two things: grouping the chunks, and naming
each group in words a person recognises.

**Clustering** is k-means, implemented here in numpy — the dependency-free path
matters because this is the feature most likely to be looked at first, on a fresh
clone, and "install scikit-learn" is a bad first impression. k-means++ seeding
and a fixed seed make it deterministic; the elbow heuristic picks *k* when the
caller does not.

**Labelling** is the interesting part. The obvious approach — most frequent terms
in the cluster — produces "the, and, retrieval" for every cluster. What works is a
contrast statistic: terms that are frequent *inside* the cluster and rare
*outside* it. That is log-odds with a Dirichlet prior, which is the standard
tool for exactly this and is well-behaved on small clusters where raw ratios
explode.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from kb.retrieval.lexical import STOPWORDS

log = logging.getLogger(__name__)

RANDOM_STATE = 42
_WORD_RE = re.compile(r"[a-z][a-z0-9_-]{2,}", re.I)

#: Terms that are frequent in every technical corpus and label nothing.
_LABEL_STOPWORDS = STOPWORDS | {
    "also",
    "because",
    "before",
    "being",
    "between",
    "both",
    "could",
    "does",
    "doing",
    "done",
    "each",
    "either",
    "every",
    "first",
    "from",
    "here",
    "into",
    "just",
    "like",
    "made",
    "make",
    "many",
    "more",
    "most",
    "much",
    "need",
    "only",
    "other",
    "over",
    "same",
    "some",
    "such",
    "than",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "thus",
    "using",
    "very",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
}


@dataclass(slots=True)
class Cluster:
    """One cluster: its members, its centre, and what it is about."""

    id: int
    label: str
    size: int
    member_indices: list[int]
    centroid: np.ndarray | None = None
    terms: list[str] = field(default_factory=list)
    coherence: float = 0.0
    """Mean cosine similarity of members to the centroid. Low means the cluster
    is a grab-bag and its label should not be trusted."""


def kmeans(
    matrix: np.ndarray, k: int, *, max_iterations: int = 100, tolerance: float = 1e-5
) -> tuple[np.ndarray, np.ndarray]:
    """k-means with k-means++ seeding. Returns ``(labels, centroids)``.

    Deterministic: the seeding uses a fixed-seed generator, so the same corpus
    produces the same clusters and a map is comparable across runs.
    """
    n_samples = matrix.shape[0]
    if k <= 1 or n_samples <= k:
        return np.zeros(n_samples, dtype=int), matrix.mean(axis=0, keepdims=True)

    rng = np.random.default_rng(RANDOM_STATE)
    centroids = _kmeans_plusplus(matrix, k, rng)
    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iterations):
        distances = _squared_distances(matrix, centroids)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        shift = 0.0
        for index in range(k):
            members = matrix[labels == index]
            if members.size == 0:
                # An empty cluster would produce NaN centroids. Reseed it to the
                # point furthest from its assigned centroid — the standard fix,
                # and it makes the next iteration strictly better.
                furthest = int(np.argmax(np.min(distances, axis=1)))
                new_centroid = matrix[furthest]
            else:
                new_centroid = members.mean(axis=0)
            shift = max(shift, float(np.linalg.norm(new_centroid - centroids[index])))
            centroids[index] = new_centroid
        if shift < tolerance:
            break

    return labels, centroids


def _kmeans_plusplus(matrix: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: spread the initial centroids by squared distance.

    Random seeding routinely produces two centroids inside one dense topic and
    none in another, which no amount of iteration recovers from.
    """
    n_samples = matrix.shape[0]
    centroids = np.empty((k, matrix.shape[1]), dtype=matrix.dtype)
    centroids[0] = matrix[rng.integers(n_samples)]

    closest = _squared_distances(matrix, centroids[:1]).ravel()
    for index in range(1, k):
        total = float(closest.sum())
        if total <= 0:
            centroids[index] = matrix[rng.integers(n_samples)]
        else:
            probabilities = closest / total
            centroids[index] = matrix[rng.choice(n_samples, p=probabilities)]
        closest = np.minimum(
            closest, _squared_distances(matrix, centroids[index : index + 1]).ravel()
        )
    return centroids


def _squared_distances(matrix: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Pairwise squared euclidean distances via the expanded form.

    ‖a−b‖² = ‖a‖² + ‖b‖² − 2a·b, which turns the whole thing into one matrix
    multiply instead of a Python loop over centroids.
    """
    a2 = np.einsum("ij,ij->i", matrix, matrix)[:, None]
    b2 = np.einsum("ij,ij->i", centroids, centroids)[None, :]
    return np.maximum(a2 + b2 - 2.0 * (matrix @ centroids.T), 0.0)


def suggest_k(n_samples: int, *, minimum: int = 2, maximum: int = 12) -> int:
    """A sensible cluster count for a corpus of this size.

    ``√(n/2)`` is the standard rule of thumb, clamped. Better than a fixed
    default, which gives 8 clusters for 12 chunks and 8 for 12,000.
    """
    if n_samples < 6:
        return 1
    return max(minimum, min(maximum, round(math.sqrt(n_samples / 2))))


def cluster_corpus(
    matrix: np.ndarray,
    texts: Sequence[str],
    *,
    k: int | None = None,
    max_terms: int = 4,
) -> list[Cluster]:
    """Cluster the corpus and label each cluster from its distinctive terms."""
    n_samples = matrix.shape[0]
    if n_samples == 0:
        return []

    cluster_count = k if k is not None else suggest_k(n_samples)
    labels, centroids = kmeans(matrix, cluster_count)

    # Normalise once so coherence is a plain dot product.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.maximum(norms, 1e-12)

    corpus_counts = _term_counts(texts)
    clusters: list[Cluster] = []

    for index in range(centroids.shape[0]):
        members = [i for i in range(n_samples) if labels[i] == index]
        if not members:
            continue
        centroid = centroids[index]
        centroid_norm = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
        coherence = float(np.mean(unit[members] @ centroid_norm)) if members else 0.0

        terms = distinctive_terms([texts[i] for i in members], corpus_counts, limit=max_terms)
        clusters.append(
            Cluster(
                id=index,
                label=" · ".join(terms) if terms else f"cluster {index + 1}",
                size=len(members),
                member_indices=members,
                centroid=centroid,
                terms=terms,
                coherence=round(coherence, 4),
            )
        )

    clusters.sort(key=lambda c: -c.size)
    return clusters


def _term_counts(texts: Sequence[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(_tokens(text))
    return counts


def _document_frequency(texts: Sequence[str]) -> Counter[str]:
    """How many of ``texts`` each term appears in, regardless of how often."""
    frequency: Counter[str] = Counter()
    for text in texts:
        frequency.update(set(_tokens(text)))
    return frequency


def _tokens(text: str) -> list[str]:
    return [
        token.lower() for token in _WORD_RE.findall(text) if token.lower() not in _LABEL_STOPWORDS
    ]


def distinctive_terms(
    cluster_texts: Sequence[str],
    corpus_counts: Counter[str],
    *,
    limit: int = 4,
    prior: float = 0.01,
) -> list[str]:
    """Terms frequent *inside* the cluster and rare *outside* it.

    Log-odds with an informative Dirichlet prior, following Monroe et al.: for a
    term with cluster count ``y_i`` and corpus count ``n_i``,

        δ = log( (y_i + α) / (Y + A − y_i − α) ) − log( (n_i − y_i + α) / (N + A − …) )

    The prior is what makes this work on small clusters. A raw frequency ratio
    puts a term appearing once inside and never outside at the top of every
    list; the prior damps exactly that, and it is why the labels come out as
    "fusion · ranks · rrf" instead of "the · and · retrieval".
    """
    cluster_counts = _term_counts(cluster_texts)
    if not cluster_counts:
        return []

    # A term appearing many times in *one* chunk is that chunk's vocabulary, not
    # the cluster's. Requiring it in two chunks is what stops a single verbose
    # passage from naming the whole region.
    within = _document_frequency(cluster_texts)
    min_chunks = 2 if len(cluster_texts) >= 4 else 1

    cluster_total = sum(cluster_counts.values())
    corpus_total = sum(corpus_counts.values())
    vocabulary_size = max(len(corpus_counts), 1)
    alpha = prior
    prior_total = alpha * vocabulary_size

    scores: list[tuple[float, str]] = []
    for term, count in cluster_counts.items():
        if len(term) < 3:
            continue
        if within[term] < min_chunks:
            continue
        outside = max(corpus_counts.get(term, count) - count, 0)
        inside_odds = (count + alpha) / max(cluster_total + prior_total - count - alpha, 1e-9)
        outside_odds = (outside + alpha) / max(
            corpus_total - cluster_total + prior_total - outside - alpha, 1e-9
        )
        delta = math.log(inside_odds) - math.log(outside_odds)
        # Break ties toward terms that are spread across the cluster.
        scores.append((delta + 0.05 * math.log1p(within[term]), term))

    scores.sort(key=lambda item: (-item[0], item[1]))
    return [term for _, term in scores[:limit]]
