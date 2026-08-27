"""Embedder tests.

The hashing embedder is the default and CI has no keys, so its properties matter:
determinism across processes, sane similarity ordering, and correct handling of
degenerate input.
"""

from __future__ import annotations

import numpy as np
import pytest

from kb.config import EmbeddingProvider, Settings
from kb.embeddings import build_embedder
from kb.embeddings.cache import CachedEmbedder
from kb.embeddings.hashing import HashingEmbedder


class TestHashingEmbedder:
    def test_shape_and_dtype(self) -> None:
        embedder = HashingEmbedder(dim=128)
        vectors = embedder.embed_documents(["hello world", "another document"])
        assert vectors.shape == (2, 128)
        assert vectors.dtype == np.float32

    def test_vectors_are_unit_length(self) -> None:
        embedder = HashingEmbedder(dim=64)
        vectors = embedder.embed_documents(["retrieval augmented generation"])
        assert np.linalg.norm(vectors[0]) == pytest.approx(1.0, abs=1e-5)

    def test_deterministic_across_instances(self) -> None:
        text = "reciprocal rank fusion"
        first = HashingEmbedder(dim=64).embed_query(text)
        second = HashingEmbedder(dim=64).embed_query(text)
        assert np.array_equal(first, second)

    def test_identical_text_is_maximally_similar(self) -> None:
        embedder = HashingEmbedder(dim=256)
        a, b = embedder.embed_documents(["hybrid search", "hybrid search"])
        assert float(a @ b) == pytest.approx(1.0, abs=1e-5)

    def test_related_text_scores_above_unrelated(self) -> None:
        embedder = HashingEmbedder(dim=512)
        query = embedder.embed_query("reciprocal rank fusion combines rankings")
        related = embedder.embed_documents(
            ["Reciprocal Rank Fusion combines ranked lists using ranks."]
        )[0]
        unrelated = embedder.embed_documents(
            ["Bananas are a tropical fruit grown in humid climates."]
        )[0]
        assert float(query @ related) > float(query @ unrelated)

    def test_word_order_changes_the_vector(self) -> None:
        """Bigrams mean "vector search" is not the same as "search vector"."""
        embedder = HashingEmbedder(dim=512, use_bigrams=True)
        a, b = embedder.embed_documents(["vector search", "search vector"])
        assert float(a @ b) < 1.0

    def test_empty_text_yields_a_zero_vector(self) -> None:
        vectors = HashingEmbedder(dim=32).embed_documents(["", "   "])
        assert np.all(vectors == 0.0)

    def test_empty_batch(self) -> None:
        assert HashingEmbedder(dim=32).embed_documents([]).shape == (0, 32)

    def test_repeated_words_are_damped(self) -> None:
        """Sublinear tf: 20 copies must not be 20x the single occurrence."""
        embedder = HashingEmbedder(dim=256, use_char_ngrams=False, use_bigrams=False)
        one = embedder.embed_documents(["retrieval"])[0]
        many = embedder.embed_documents(["retrieval " * 20])[0]
        assert float(one @ many) == pytest.approx(1.0, abs=1e-5)

    def test_tiny_dim_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 16"):
            HashingEmbedder(dim=4)


class TestBuildEmbedder:
    def test_hashing_is_the_default(self, tmp_settings: Settings) -> None:
        embedder = build_embedder(tmp_settings)
        assert isinstance(embedder, HashingEmbedder)
        assert embedder.dim == tmp_settings.embedding_dim

    def test_hashing_is_not_wrapped_in_a_cache(self, tmp_settings: Settings) -> None:
        """Caching a pure function of its input would only add I/O."""
        assert not isinstance(build_embedder(tmp_settings), CachedEmbedder)

    def test_model_name_defaults_per_provider(self) -> None:
        assert Settings(embedding_provider=EmbeddingProvider.HASHING).embedding_model == (
            "hashing-ngram-v1"
        )
        assert Settings(embedding_provider=EmbeddingProvider.VOYAGE).embedding_model == "voyage-3"


class _CountingEmbedder(HashingEmbedder):
    """Records how many texts actually reached the underlying model."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__(dim=dim, model="counting-v1")
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        return super().embed_documents(texts)


class TestCachedEmbedder:
    def test_second_call_is_served_from_cache(self, tmp_path) -> None:
        inner = _CountingEmbedder()
        cached = CachedEmbedder(inner, tmp_path / "cache.db")
        texts = ["alpha", "beta"]

        first = cached.embed_documents(texts)
        assert inner.calls == 2

        second = cached.embed_documents(texts)
        assert inner.calls == 2, "cache hit should not reach the model"
        assert np.allclose(first, second)
        assert cached.stats()["hits"] == 2
        cached.close()

    def test_partial_hit_only_embeds_the_new_text(self, tmp_path) -> None:
        inner = _CountingEmbedder()
        cached = CachedEmbedder(inner, tmp_path / "cache.db")
        cached.embed_documents(["alpha"])
        cached.embed_documents(["alpha", "beta"])
        assert inner.calls == 2
        cached.close()

    def test_cache_survives_reopening(self, tmp_path) -> None:
        path = tmp_path / "cache.db"
        first_inner = _CountingEmbedder()
        first = CachedEmbedder(first_inner, path)
        first.embed_documents(["alpha"])
        first.close()

        second_inner = _CountingEmbedder()
        second = CachedEmbedder(second_inner, path)
        second.embed_documents(["alpha"])
        assert second_inner.calls == 0
        second.close()

    def test_queries_bypass_the_cache(self, tmp_path) -> None:
        inner = _CountingEmbedder()
        cached = CachedEmbedder(inner, tmp_path / "cache.db")
        cached.embed_query("a one-off question")
        assert cached.stats()["hits"] == 0
        cached.close()

    def test_duplicate_texts_in_one_batch_embed_once(self, tmp_path) -> None:
        inner = _CountingEmbedder()
        cached = CachedEmbedder(inner, tmp_path / "cache.db")
        vectors = cached.embed_documents(["same", "same", "same"])
        assert vectors.shape == (3, inner.dim)
        assert np.allclose(vectors[0], vectors[2])
        assert inner.calls == 1, "identical texts in one batch should embed once"
        cached.close()
