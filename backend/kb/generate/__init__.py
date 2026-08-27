"""Answer generation: grounded text with validated citations."""

from __future__ import annotations

import logging

from kb.config import GenerationProvider, Settings
from kb.generate.base import Generator
from kb.generate.extractive import ExtractiveGenerator
from kb.generate.prompt import (
    build_prompt,
    format_sources,
    pack_context,
    parse_markers,
    split_into_sentences,
    strip_invalid_markers,
)

log = logging.getLogger(__name__)

__all__ = [
    "ExtractiveGenerator",
    "Generator",
    "build_generator",
    "build_prompt",
    "format_sources",
    "pack_context",
    "parse_markers",
    "split_into_sentences",
    "strip_invalid_markers",
]


def build_generator(settings: Settings) -> Generator:
    """Construct the configured generator.

    Falls back to the extractive generator when a hosted provider is unavailable.
    That fallback is safe in a way most fallbacks are not: an extractive answer is
    verbatim from the sources, so degrading to it cannot introduce an
    unsupported claim.
    """
    provider = settings.generation_provider

    if provider is GenerationProvider.EXTRACTIVE:
        return ExtractiveGenerator(token_budget=settings.context_token_budget)

    try:
        return _build_llm(provider, settings)
    except Exception as exc:
        log.warning(
            "generation provider %s unavailable (%s); using the extractive generator",
            provider.value,
            exc,
        )
        return ExtractiveGenerator(token_budget=settings.context_token_budget)


def _build_llm(provider: GenerationProvider, settings: Settings) -> Generator:
    from kb.generate.llm import LLMGenerator
    from kb.llm import LLMClient

    client: LLMClient

    if provider is GenerationProvider.ANTHROPIC:
        from kb.llm import AnthropicClient

        client = AnthropicClient(
            model=settings.generation_model, api_key=settings.anthropic_api_key
        )
        name = "anthropic"
    elif provider is GenerationProvider.OPENAI:
        from kb.llm import OpenAIClient

        client = OpenAIClient(model=settings.generation_model, api_key=settings.openai_api_key)
        name = "openai"
    else:  # pragma: no cover - exhaustive over the enum
        raise ValueError(f"unsupported generation provider: {provider}")

    return LLMGenerator(
        client,
        name=name,
        max_tokens=settings.generation_max_tokens,
        temperature=settings.generation_temperature,
        token_budget=settings.context_token_budget,
    )
