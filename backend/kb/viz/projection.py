"""2D projection of the corpus.

The point of a corpus map is not decoration. It answers questions you cannot ask
any other way: does this knowledge base actually cover the topics people ask
about, are two documents saying the same thing, and is there a cluster nobody's
queries ever reach.

Three projections, chosen by what is installed:

* **UMAP** — best structure preservation, but a heavy dependency (numba, llvmlite).
* **t-SNE** — via scikit-learn; good local structure, no global structure.
* **PCA** — implemented here in ~30 lines of numpy, so *something* always works.

PCA is the fallback rather than an error because a map that exists is worth more
than a perfect map that does not. It is honest about being a linear projection:
the response reports which method actually ran, and the UI says so.

The implementation detail that matters most is **deterministic output**. A map
whose points move on every reload cannot be compared across runs, so every
method is seeded, and the PCA sign convention is fixed (each component's largest
absolute loading is forced positive) because eigenvector signs are otherwise
arbitrary and would mirror the plot at random.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

ProjectionMethod = Literal["umap", "tsne", "pca"]

#: Fixed seed everywhere, so a map is comparable across runs.
RANDOM_STATE = 42


@dataclass(slots=True)
class Projection:
    """A 2D embedding of the corpus, with the method that produced it."""

    coordinates: np.ndarray
    method: ProjectionMethod
    explained_variance: float | None = None
    """Share of variance the two axes capture. PCA only — and the honest number
    to show, because a low value means the picture is misleading."""
    notes: list[str] = field(default_factory=list)


def project(
    vectors: np.ndarray,
    *,
    method: ProjectionMethod | Literal["auto"] = "auto",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    perplexity: float | None = None,
) -> Projection:
    """Reduce ``vectors`` to two dimensions.

    ``method="auto"`` picks the best available: UMAP, then t-SNE, then PCA.
    """
    matrix = np.asarray(vectors, dtype="float64")
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {matrix.shape}")

    n_samples = matrix.shape[0]
    notes: list[str] = []

    if n_samples == 0:
        return Projection(np.zeros((0, 2)), "pca", notes=["empty corpus"])
    if n_samples < 3:
        # Nothing to project: place the points on a line so the UI has something
        # valid rather than a special case.
        coordinates = np.zeros((n_samples, 2))
        coordinates[:, 0] = np.arange(n_samples, dtype="float64")
        return Projection(coordinates, "pca", notes=["too few chunks to project"])

    order: list[ProjectionMethod] = (
        ["umap", "tsne", "pca"] if method == "auto" else [method]  # type: ignore[list-item]
    )

    for candidate in order:
        try:
            if candidate == "umap":
                return Projection(
                    _umap(matrix, n_neighbors=n_neighbors, min_dist=min_dist),
                    "umap",
                    notes=notes,
                )
            if candidate == "tsne":
                return Projection(_tsne(matrix, perplexity=perplexity), "tsne", notes=notes)
            coordinates, explained = _pca(matrix)
            return Projection(coordinates, "pca", explained_variance=explained, notes=notes)
        except ImportError as exc:
            notes.append(f"{candidate} unavailable ({exc.name or exc}); falling back")
            log.info("%s unavailable, falling back: %s", candidate, exc)
        except Exception as exc:
            notes.append(f"{candidate} failed ({exc}); falling back")
            log.warning("%s projection failed: %s", candidate, exc)

    coordinates, explained = _pca(matrix)
    notes.append("used the built-in PCA fallback")
    return Projection(coordinates, "pca", explained_variance=explained, notes=notes)


# --------------------------------------------------------------------------- #
# implementations
# --------------------------------------------------------------------------- #


def _umap(matrix: np.ndarray, *, n_neighbors: int, min_dist: float) -> np.ndarray:
    import umap  # type: ignore[import-not-found]

    # n_neighbors above the sample count is an error, not a no-op.
    neighbors = max(2, min(n_neighbors, matrix.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    return np.asarray(reducer.fit_transform(matrix), dtype="float64")


def _tsne(matrix: np.ndarray, *, perplexity: float | None) -> np.ndarray:
    from sklearn.manifold import TSNE  # type: ignore[import-not-found]

    n_samples = matrix.shape[0]
    # sklearn requires perplexity < n_samples; the usual 30 breaks on small corpora.
    effective = perplexity if perplexity is not None else min(30.0, max(5.0, n_samples / 4))
    effective = min(effective, max(2.0, n_samples - 1.5))
    reducer = TSNE(
        n_components=2,
        perplexity=effective,
        metric="cosine",
        init="pca",
        random_state=RANDOM_STATE,
    )
    return np.asarray(reducer.fit_transform(matrix), dtype="float64")


def _pca(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Two-component PCA via SVD, in numpy only.

    Uses SVD rather than an eigendecomposition of the covariance matrix: it is
    numerically better conditioned and avoids forming a ``dim × dim`` matrix,
    which for 1536-dimensional embeddings is the expensive part.
    """
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    # full_matrices=False keeps this O(n · dim · 2) instead of materialising V.
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)

    top = components[:2]
    # Eigenvector signs are arbitrary, so an unseeded run mirrors the plot at
    # random. Forcing each component's largest-magnitude loading positive makes
    # the output reproducible and comparable across runs.
    for i in range(top.shape[0]):
        dominant = np.argmax(np.abs(top[i]))
        if top[i, dominant] < 0:
            top[i] = -top[i]

    coordinates = centered @ top.T
    if coordinates.shape[1] < 2:  # a rank-1 corpus
        coordinates = np.hstack(
            [coordinates, np.zeros((coordinates.shape[0], 2 - coordinates.shape[1]))]
        )

    total = float((singular_values**2).sum())
    explained = float((singular_values[:2] ** 2).sum() / total) if total > 0 else 0.0
    return coordinates, round(explained, 4)


def normalize_to_unit_square(coordinates: np.ndarray) -> np.ndarray:
    """Scale coordinates into ``[0, 1]²``, preserving aspect ratio.

    Done server-side so every client renders the same layout, and scaled by the
    *larger* axis range rather than per-axis: independent scaling would distort
    the distances the projection exists to show.
    """
    if coordinates.size == 0:
        return coordinates
    minimums = coordinates.min(axis=0)
    maximums = coordinates.max(axis=0)
    span = float(np.max(maximums - minimums))
    if span < 1e-12:
        return np.full_like(coordinates, 0.5)
    centered = coordinates - (minimums + maximums) / 2.0
    return centered / span + 0.5
