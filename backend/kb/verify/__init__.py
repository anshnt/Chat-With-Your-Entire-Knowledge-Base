"""Citation verification: checking that a cited source says what it is cited for."""

from __future__ import annotations

import logging

from kb.config import Settings, VerifyProvider
from kb.verify.base import Verifier, is_claim, strip_markers
from kb.verify.lexical import LexicalVerifier

log = logging.getLogger(__name__)

__all__ = [
    "LexicalVerifier",
    "Verifier",
    "build_verifier",
    "is_claim",
    "strip_markers",
]


def build_verifier(settings: Settings) -> Verifier | None:
    """Construct the configured verifier, or ``None`` when disabled.

    Falls back to the offline verifier if a hosted judge cannot be built. That
    fallback is safe in the right direction: the lexical verifier's failure mode
    is a false "unsupported", so degrading to it makes the check stricter rather
    than laxer.
    """
    if not settings.verify_citations:
        return None

    provider = settings.verify_provider
    if provider is VerifyProvider.NONE:
        return None
    if provider is VerifyProvider.LEXICAL:
        return LexicalVerifier(threshold=settings.verification_threshold)

    try:
        return _build_llm(settings)
    except Exception as exc:
        log.warning(
            "verifier %s unavailable (%s); using the offline lexical verifier",
            provider.value,
            exc,
        )
        return LexicalVerifier(threshold=settings.verification_threshold)


def _build_llm(settings: Settings) -> Verifier:
    from kb.llm import AnthropicClient, LLMClient, OpenAIClient
    from kb.verify.llm import LLMVerifier

    client: LLMClient
    if settings.anthropic_api_key:
        client = AnthropicClient(model=settings.verify_model, api_key=settings.anthropic_api_key)
    elif settings.openai_api_key:
        client = OpenAIClient(model=settings.verify_model, api_key=settings.openai_api_key)
    else:
        raise RuntimeError("no LLM credentials configured for verification")
    return LLMVerifier(client, threshold=settings.verification_threshold)
