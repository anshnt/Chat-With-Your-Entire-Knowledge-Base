"""Embedder contract and registry.

Every embedder exposes the *same* two methods and, importantly, distinguishes
documents from queries. Asymmetric models (Voyage, E5, BGE, Cohere) require
different prefixes or input types for the two sides, and getting that wrong
quietly costs several points of recall — so it is part of the interface rather
than an implementation detail.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence

import numpy as np


class Embedder(abc.ABC):
    """Turns text into dense vectors."""

    #: Provider-facing model identifier, stored alongside every vector so a
    #: collection can never be queried with vectors from a different model.
    model: str = ""
    #: Output dimensionality.
    dim: int = 0

    @abc.abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed corpus text. Returns ``(len(texts), dim)`` float32."""

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query. Returns ``(dim,)`` float32.

        Defaults to the document path; asymmetric models override it.
        """
        return self.embed_documents([text])[0]

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self.embed_query(t) for t in texts]) if texts else np.zeros((0, self.dim))

    # -- helpers ------------------------------------------------------- #

    @staticmethod
    def normalize(matrix: np.ndarray) -> np.ndarray:
        """L2-normalise rows so cosine similarity is a plain dot product."""
        arr = np.asarray(matrix, dtype="float32")
        if arr.ndim == 1:
            norm = float(np.linalg.norm(arr))
            return arr / max(norm, 1e-12)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-12)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(model={self.model!r}, dim={self.dim})"
