"""LLM-backed generator.

Thin by design: the base class already owns context packing, marker validation,
citation resolution and sentence splitting, so this is the prompt plus the call.

The system prompt is the part worth reading. Two instructions do most of the work
in practice:

* *"Use only what the sources say. Never add facts from your own knowledge, even
  if you are confident they are correct."* — the failure mode is not a model
  inventing nonsense, it is a model correctly completing a fact that is simply
  not in the corpus, which then carries a citation to a chunk that does not say
  it. That is the exact case citation verification catches.
* *"If the sources do not contain the answer, say so plainly and stop."* — with a
  usable refusal path, a bad retrieval produces an honest "not in the corpus"
  instead of a confident answer stitched from adjacent material.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from kb.generate.base import Generator
from kb.generate.prompt import SYSTEM_PROMPT, build_prompt
from kb.llm import LLMClient
from kb.models import Chunk


class LLMGenerator(Generator):
    """Generates a grounded answer with a language model."""

    supports_streaming = True

    def __init__(
        self,
        client: LLMClient,
        *,
        name: str = "llm",
        max_tokens: int = 1500,
        temperature: float = 0.0,
        token_budget: int = 6000,
        max_chunks: int | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        super().__init__(token_budget=token_budget, max_chunks=max_chunks)
        self.client = client
        self.name = name
        self.model = client.model
        self.max_tokens = max_tokens
        # Default 0.0: for grounded extraction, sampling variety buys nothing and
        # costs reproducibility, which the evaluation harness depends on.
        self.temperature = temperature
        self.system_prompt = system_prompt

    def _generate_text(self, query: str, chunks: Sequence[Chunk]) -> str:
        return self.client.complete(
            build_prompt(query, chunks),
            system=self.system_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def _stream_text(self, query: str, chunks: Sequence[Chunk]) -> Iterator[str]:
        yield from self.client.stream(
            build_prompt(query, chunks),
            system=self.system_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
