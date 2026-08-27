"""Generator contract.

A generator turns a query plus retrieved chunks into an :class:`~kb.models.Answer`.
Everything shared — context packing, marker validation, sentence splitting,
citation construction — lives here, so a provider implementation is only ever the
part that differs: producing text.

That matters because the shared part is where correctness lives. Validating the
model's citation markers against the sources actually supplied is not optional
polish; it is the difference between a citation and a decoration.
"""

from __future__ import annotations

import abc
import logging
import time
from collections.abc import Iterator, Sequence

from kb.generate.prompt import (
    build_prompt,
    looks_like_refusal,
    pack_context,
    split_into_sentences,
    strip_invalid_markers,
)
from kb.models import (
    Answer,
    AnswerCitation,
    Chunk,
    RetrievalResult,
    ScoredChunk,
    estimate_tokens,
)

log = logging.getLogger(__name__)


class Generator(abc.ABC):
    """Produces a grounded answer from retrieved context."""

    name: str = "generator"
    model: str = ""
    supports_streaming: bool = False

    def __init__(self, *, token_budget: int = 6000, max_chunks: int | None = None) -> None:
        self.token_budget = token_budget
        self.max_chunks = max_chunks

    # ------------------------------------------------------------------ #
    # provider surface
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def _generate_text(self, query: str, chunks: Sequence[Chunk]) -> str:
        """Return answer text citing sources as ``[n]``, 1-based over ``chunks``."""

    def _stream_text(self, query: str, chunks: Sequence[Chunk]) -> Iterator[str]:
        """Yield answer text incrementally. Defaults to one chunk."""
        yield self._generate_text(query, chunks)

    # ------------------------------------------------------------------ #
    # shared pipeline
    # ------------------------------------------------------------------ #

    def generate(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        *,
        retrieval: RetrievalResult | None = None,
    ) -> Answer:
        """Generate an answer and resolve its citations to source positions."""
        timings: dict[str, float] = {}
        started = time.perf_counter()
        chunks = pack_context(
            candidates, token_budget=self.token_budget, max_chunks=self.max_chunks
        )
        timings["context_ms"] = _ms_since(started)

        if not chunks:
            return self._empty_answer(query, retrieval, timings)

        generation_started = time.perf_counter()
        raw = self._generate_text(query, chunks)
        timings["generation_ms"] = _ms_since(generation_started)

        return self.finalize(query, raw, chunks, retrieval=retrieval, timings=timings)

    def stream(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        *,
        retrieval: RetrievalResult | None = None,
    ) -> Iterator[tuple[str, Answer | None]]:
        """Yield ``(delta, None)`` while generating, then ``("", answer)`` once.

        Citations cannot be resolved until the text is complete — a marker may be
        mid-emission — so the final answer arrives as a separate terminal event
        rather than being patched in as it goes.
        """
        timings: dict[str, float] = {}
        started = time.perf_counter()
        chunks = pack_context(
            candidates, token_budget=self.token_budget, max_chunks=self.max_chunks
        )
        timings["context_ms"] = _ms_since(started)

        if not chunks:
            answer = self._empty_answer(query, retrieval, timings)
            yield answer.text, None
            yield "", answer
            return

        generation_started = time.perf_counter()
        parts: list[str] = []
        for delta in self._stream_text(query, chunks):
            parts.append(delta)
            yield delta, None
        timings["generation_ms"] = _ms_since(generation_started)

        yield "", self.finalize(query, "".join(parts), chunks, retrieval=retrieval, timings=timings)

    def finalize(
        self,
        query: str,
        raw_text: str,
        chunks: Sequence[Chunk],
        *,
        retrieval: RetrievalResult | None = None,
        timings: dict[str, float] | None = None,
    ) -> Answer:
        """Validate markers, resolve citations, and split into sentences."""
        timings = dict(timings or {})
        started = time.perf_counter()

        valid = set(range(1, len(chunks) + 1))
        text, invalid = strip_invalid_markers(raw_text.strip(), valid)
        if invalid:
            # Worth a warning, not a failure: the answer is still usable, but a
            # model inventing source numbers is a signal about the prompt.
            log.warning(
                "%s cited %d source(s) that were not provided: %s",
                self.name,
                len(invalid),
                invalid,
            )

        sentences = split_into_sentences(text)
        used = sorted({m for s in sentences for m in s.citation_markers})
        citations = [
            AnswerCitation.from_chunk(marker, chunks[marker - 1])
            for marker in used
            if 1 <= marker <= len(chunks)
        ]
        timings["citations_ms"] = _ms_since(started)

        return Answer(
            query=query,
            text=text,
            citations=citations,
            sentences=sentences,
            generator=self.name,
            model=self.model,
            context_chunks=len(chunks),
            context_tokens=sum(c.token_estimate or estimate_tokens(c.text) for c in chunks),
            refused=looks_like_refusal(text),
            retrieval=retrieval,
            timings_ms=timings,
        )

    # ------------------------------------------------------------------ #

    def _empty_answer(
        self,
        query: str,
        retrieval: RetrievalResult | None,
        timings: dict[str, float],
    ) -> Answer:
        """What to say when retrieval found nothing.

        Answering from parametric knowledge here would be the single worst thing
        a grounded system could do: it looks like a cited answer and is not one.
        """
        text = (
            "I could not find anything in the knowledge base that addresses this "
            "question. Try rephrasing it, or ingest a source that covers it."
        )
        return Answer(
            query=query,
            text=text,
            generator=self.name,
            model=self.model,
            refused=True,
            retrieval=retrieval,
            timings_ms=timings,
        )

    def build_prompt(self, query: str, chunks: Sequence[Chunk]) -> str:
        """Exposed so the prompt can be inspected without generating."""
        return build_prompt(query, chunks)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r}, model={self.model!r})"


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)
