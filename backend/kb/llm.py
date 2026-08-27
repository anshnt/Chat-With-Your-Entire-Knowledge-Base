"""Thin LLM client used by reranking, generation and citation verification.

Deliberately not a framework. Three needs, one interface:

* ``complete`` — one prompt in, text out (listwise reranking, claim verification)
* ``stream`` — token iterator (answer generation)
* structured retries — provider SDKs raise a dozen exception types; transient
  failures (429, 5xx, timeouts, overload) back off, everything else fails fast
  because a 401 will not fix itself.

Keeping it here rather than inside the generation layer means the reranker and
the verifier do not depend on generation, and swapping providers touches one
file.
"""

from __future__ import annotations

import abc
import logging
import os
import random
import time
from collections.abc import Iterator
from typing import Any

from kb.errors import MissingDependencyError, ProviderError

log = logging.getLogger(__name__)

_MAX_ATTEMPTS = 4
_RETRY_BASE_DELAY = 0.6


class LLMClient(abc.ABC):
    """Minimal chat-completion interface."""

    model: str = ""

    @abc.abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Return the model's full reply as text."""

    def stream(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """Yield reply text incrementally.

        Defaults to a single chunk so a provider without streaming still works
        through the streaming code path.
        """
        yield self.complete(prompt, system=system, max_tokens=max_tokens, temperature=temperature)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(model={self.model!r})"


class AnthropicClient(LLMClient):
    """Anthropic Messages API."""

    #: Overridable without touching code, so the model can be rolled forward by
    #: configuration rather than a release.
    DEFAULT_MODEL = os.environ.get("KB_ANTHROPIC_MODEL", "claude-sonnet-4-5")

    def __init__(self, *, model: str = "", api_key: str = "", timeout: float = 120.0) -> None:
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MissingDependencyError("anthropic", "llm") from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        self._sdk = anthropic
        self._client = anthropic.Anthropic(api_key=key, timeout=timeout)
        self.model = model or self.DEFAULT_MODEL

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        def call() -> str:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system:
                kwargs["system"] = system
            response = self._client.messages.create(**kwargs)
            return "".join(
                block.text for block in response.content if getattr(block, "type", "") == "text"
            )

        return _with_retries(call, what=f"anthropic {self.model}")

    def stream(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        try:
            with self._client.messages.stream(**kwargs) as stream:
                yield from stream.text_stream
        except Exception as exc:
            raise ProviderError(f"anthropic stream failed: {exc}") from exc


class OpenAIClient(LLMClient):
    """OpenAI Chat Completions."""

    DEFAULT_MODEL = os.environ.get("KB_OPENAI_MODEL", "gpt-4o-mini")

    def __init__(self, *, model: str = "", api_key: str = "", timeout: float = 120.0) -> None:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MissingDependencyError("openai", "embeddings") from exc
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ProviderError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=key, timeout=timeout)
        self.model = model or self.DEFAULT_MODEL

    def _messages(self, prompt: str, system: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        return messages

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        def call() -> str:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=self._messages(prompt, system),
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""

        return _with_retries(call, what=f"openai {self.model}")

    def stream(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> Iterator[str]:
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=self._messages(prompt, system),
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            for event in stream:
                delta = event.choices[0].delta.content if event.choices else None
                if delta:
                    yield delta
        except Exception as exc:
            raise ProviderError(f"openai stream failed: {exc}") from exc


def _with_retries(fn: Any, *, what: str) -> Any:
    last: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if not _is_transient(exc) or attempt == _MAX_ATTEMPTS - 1:
                break
            delay = _RETRY_BASE_DELAY * (2**attempt) * (0.5 + random.random())
            log.debug("%s attempt %d failed (%s); retrying in %.1fs", what, attempt + 1, exc, delay)
            time.sleep(delay)
    raise ProviderError(f"{what} failed: {last}") from last


def _is_transient(exc: Exception) -> bool:
    """True for failures worth retrying: rate limits, 5xx, timeouts, overload."""
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        return status == 429 or 500 <= status < 600
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "timeout",
            "timed out",
            "connection",
            "429",
            "503",
            "overloaded",
            "rate limit",
        )
    )
