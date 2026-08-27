"""Embedding providers and the factory that selects one from settings."""

from __future__ import annotations

from kb.config import EmbeddingProvider, Settings
from kb.embeddings.base import Embedder
from kb.embeddings.cache import CachedEmbedder
from kb.embeddings.hashing import HashingEmbedder

__all__ = [
    "CachedEmbedder",
    "Embedder",
    "HashingEmbedder",
    "build_embedder",
]


def build_embedder(settings: Settings) -> Embedder:
    """Construct the configured embedder, wrapped in a disk cache if enabled.

    Hosted providers are imported lazily here so that the default (hashing) path
    never pays for — or requires — the optional SDKs.
    """
    provider = settings.embedding_provider
    embedder: Embedder

    if provider is EmbeddingProvider.HASHING:
        embedder = HashingEmbedder(dim=settings.embedding_dim, model=settings.embedding_model)
    elif provider is EmbeddingProvider.VOYAGE:
        from kb.embeddings.providers import VoyageEmbedder

        embedder = VoyageEmbedder(
            model=settings.embedding_model,
            api_key=settings.voyage_api_key,
            batch_size=settings.embedding_batch_size,
        )
    elif provider is EmbeddingProvider.OPENAI:
        from kb.embeddings.providers import OpenAIEmbedder

        embedder = OpenAIEmbedder(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            batch_size=settings.embedding_batch_size,
        )
    elif provider is EmbeddingProvider.LOCAL:
        from kb.embeddings.providers import LocalEmbedder

        embedder = LocalEmbedder(
            model=settings.embedding_model, batch_size=settings.embedding_batch_size
        )
    else:  # pragma: no cover - exhaustive over the enum
        raise ValueError(f"unsupported embedding provider: {provider}")

    if settings.embedding_cache and provider is not EmbeddingProvider.HASHING:
        settings.ensure_dirs()
        embedder = CachedEmbedder(embedder, settings.cache_dir / "embeddings.db")
    return embedder
