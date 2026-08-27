"""Reranking: the stage that turns fusion's recall into precision."""

from __future__ import annotations

import logging

from kb.config import RerankProvider, Settings
from kb.rerank.base import NoOpReranker, Reranker
from kb.rerank.lexical import LexicalReranker

log = logging.getLogger(__name__)

__all__ = [
    "LexicalReranker",
    "NoOpReranker",
    "Reranker",
    "build_reranker",
]


def build_reranker(settings: Settings) -> Reranker | None:
    """Construct the configured reranker.

    Returns ``None`` when reranking is disabled. Hosted and local-model providers
    are imported lazily so the default (offline lexical) path never requires an
    optional extra, and a provider that cannot be constructed degrades to the
    always-available lexical reranker rather than breaking retrieval — a slightly
    worse ranking beats no answer.
    """
    if not settings.rerank_enabled:
        return None

    provider = settings.rerank_provider
    if provider is RerankProvider.NONE:
        return None
    if provider is RerankProvider.LEXICAL:
        return LexicalReranker()

    try:
        return _build_external(provider, settings)
    except Exception as exc:
        log.warning(
            "reranker %s unavailable (%s); falling back to the offline lexical reranker",
            provider.value,
            exc,
        )
        return LexicalReranker()


def _build_external(provider: RerankProvider, settings: Settings) -> Reranker:
    if provider is RerankProvider.CROSS_ENCODER:
        from kb.rerank.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker(model=settings.rerank_model)

    if provider is RerankProvider.COHERE:
        from kb.rerank.hosted import CohereReranker

        return CohereReranker(
            model=settings.rerank_model or "rerank-english-v3.0",
            api_key=settings.cohere_api_key,
            top_n=settings.rerank_top_n,
        )

    if provider is RerankProvider.VOYAGE:
        from kb.rerank.hosted import VoyageReranker

        return VoyageReranker(
            model=settings.rerank_model or "rerank-2",
            api_key=settings.voyage_api_key,
            top_n=settings.rerank_top_n,
        )

    if provider is RerankProvider.LLM:
        from kb.rerank.llm import LLMReranker

        return LLMReranker(model=settings.rerank_model, api_key=settings.anthropic_api_key)

    raise ValueError(f"unsupported rerank provider: {provider}")
