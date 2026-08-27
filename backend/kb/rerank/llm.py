"""Listwise LLM reranker.

Pointwise rerankers (cross-encoders, hosted APIs) score each passage against the
query in isolation. A listwise reranker sees all candidates at once and orders
them, which lets it make *comparative* judgements — "passage 3 defines the term,
passage 7 only mentions it" — that no independent scoring pass can express.

Two practical concerns shape the implementation:

* **Cost and latency.** One call for the whole list, not one per passage, and
  passages are truncated: the ordering decision is made in the first few hundred
  words.
* **Robustness.** An LLM asked for a permutation can return a malformed one:
  duplicates, hallucinated indices, omissions, prose around the answer. The
  parser accepts any recognisable list of integers, drops what it cannot use, and
  appends every unmentioned candidate in its original fused order. A bad response
  degrades to the first-stage ranking; it never loses a candidate.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from kb.errors import ProviderError
from kb.models import ScoredChunk
from kb.rerank.base import Reranker

log = logging.getLogger(__name__)

MAX_PASSAGE_CHARS = 1200

SYSTEM_PROMPT = (
    "You rank retrieved passages by how well each one answers a question. "
    "You are precise and you never invent passage numbers."
)

PROMPT_TEMPLATE = """Question: {query}

Passages:
{passages}

Rank the passages from most to least useful for answering the question.

Judge whether a passage contains information that actually answers the question,
not whether it is about the same topic. A passage that defines or states the
answer outranks one that merely mentions the subject.

Reply with only the passage numbers, most useful first, separated by commas.
Include every number exactly once. No other text.

Example reply: 3, 1, 7, 2"""


class LLMReranker(Reranker):
    """Reranks by asking a language model for an ordering."""

    name = "llm-rerank"

    def __init__(
        self,
        *,
        model: str = "",
        api_key: str = "",
        max_passage_chars: int = MAX_PASSAGE_CHARS,
    ) -> None:
        from kb.llm import AnthropicClient

        self._client = AnthropicClient(model=model, api_key=api_key)
        self.model = self._client.model
        self.name = f"llm-rerank:{self.model}"
        self.max_passage_chars = max_passage_chars

    # ------------------------------------------------------------------ #

    def score(self, query: str, candidates: Sequence[ScoredChunk]) -> list[float]:
        """Convert the returned ordering into descending scores.

        Scores are positional rather than absolute because a listwise model
        produces a permutation, not calibrated relevance — pretending otherwise
        would make the numbers meaningless downstream.
        """
        if not candidates:
            return []
        if len(candidates) == 1:
            return [1.0]

        order = self._request_order(query, candidates)
        n = len(candidates)
        scores = [0.0] * n
        for rank, index in enumerate(order):
            scores[index] = float(n - rank)
        return scores

    # ------------------------------------------------------------------ #

    def _request_order(self, query: str, candidates: Sequence[ScoredChunk]) -> list[int]:
        passages = "\n\n".join(
            f"[{i + 1}] {_passage_text(c, self.max_passage_chars)}"
            for i, c in enumerate(candidates)
        )
        prompt = PROMPT_TEMPLATE.format(query=query, passages=passages)
        try:
            reply = self._client.complete(
                prompt, system=SYSTEM_PROMPT, max_tokens=512, temperature=0.0
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"llm rerank failed: {exc}") from exc
        return parse_ordering(reply, len(candidates))


def parse_ordering(reply: str, n: int) -> list[int]:
    """Parse a model's ranking reply into zero-based indices.

    Tolerant by design: duplicates and out-of-range numbers are dropped, and any
    candidate the model failed to mention is appended in its original order. The
    result is always a complete permutation of ``range(n)``.
    """
    seen: set[int] = set()
    order: list[int] = []
    for match in re.finditer(r"\d+", reply or ""):
        value = int(match.group(0)) - 1
        if 0 <= value < n and value not in seen:
            seen.add(value)
            order.append(value)
    order.extend(i for i in range(n) if i not in seen)
    return order


def _passage_text(candidate: ScoredChunk, limit: int) -> str:
    chunk = candidate.chunk
    body = chunk.text
    if chunk.heading_context and chunk.heading_context not in body[:200]:
        body = f"{chunk.heading_context}\n{body}"
    body = " ".join(body.split())
    return body if len(body) <= limit else f"{body[:limit].rstrip()}…"
