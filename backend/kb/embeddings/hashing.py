"""Deterministic local embedder — the default, and the reason CI needs no keys.

This is a signed feature-hashing vectoriser over three views of the text:

* word unigrams (topical overlap),
* word bigrams (short phrases, so "vector search" ≠ "search vector"),
* character 4-grams (robustness to morphology, typos, and code identifiers).

Each feature is hashed to a bucket with a sign drawn from the same hash, which
keeps collisions unbiased in expectation. Sublinear term-frequency damping
(``1 + log tf``) stops repeated words from dominating, and rows are L2-normalised
so cosine similarity is a dot product.

It is not a trained semantic model and does not pretend to be — it has no notion
that "car" relates to "automobile". What it *is*: fast, dependency-free, exactly
reproducible, and strong enough that the whole pipeline (hybrid fusion, MMR,
reranking, evaluation) can be tested end to end and demoed offline. Swap in
``KB_EMBEDDING_PROVIDER=voyage`` for real semantics; nothing else changes.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

from kb.embeddings.base import Embedder

_WORD_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
_CHAR_NGRAM = 4


class HashingEmbedder(Embedder):
    """Signed feature-hashing embedder. Deterministic across runs and machines."""

    def __init__(
        self,
        dim: int = 512,
        *,
        model: str = "hashing-ngram-v1",
        use_bigrams: bool = True,
        use_char_ngrams: bool = True,
        char_weight: float = 0.35,
        bigram_weight: float = 0.7,
    ) -> None:
        if dim < 16:
            raise ValueError("dim must be at least 16")
        self.dim = dim
        self.model = model
        self.use_bigrams = use_bigrams
        self.use_char_ngrams = use_char_ngrams
        self.char_weight = char_weight
        self.bigram_weight = bigram_weight

    # ------------------------------------------------------------------ #

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            out[i] = self._embed_one(text)
        return out

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(text)

    # ------------------------------------------------------------------ #

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        if not text or not text.strip():
            return vec

        words = _WORD_RE.findall(text.lower())
        if words:
            self._accumulate(vec, Counter(words), weight=1.0, namespace="w")
            if self.use_bigrams and len(words) > 1:
                bigrams = Counter(f"{a}_{b}" for a, b in itertools.pairwise(words))
                self._accumulate(vec, bigrams, weight=self.bigram_weight, namespace="b")

        if self.use_char_ngrams:
            squashed = re.sub(r"\s+", " ", text.lower())
            grams = Counter(
                squashed[j : j + _CHAR_NGRAM]
                for j in range(max(0, len(squashed) - _CHAR_NGRAM + 1))
            )
            self._accumulate(vec, grams, weight=self.char_weight, namespace="c")

        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-12 else vec

    def _accumulate(
        self, vec: np.ndarray, counts: Counter[str], *, weight: float, namespace: str
    ) -> None:
        for feature, count in counts.items():
            bucket, sign = self._hash(f"{namespace}:{feature}")
            # Sublinear tf damping: the 10th occurrence of a word says much less
            # than the 2nd, and undamped counts let boilerplate swamp the signal.
            vec[bucket] += sign * weight * (1.0 + math.log(count))

    def _hash(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        bucket = value % self.dim
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return bucket, sign
