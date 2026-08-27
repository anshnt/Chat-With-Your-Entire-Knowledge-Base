"""Citation verification.

A citation nobody checked is decoration. The failure this stage exists to catch
is not a model inventing nonsense — it is a model *correctly completing a fact
that is not in the corpus* and attaching a citation to a chunk that does not say
it. The answer reads perfectly, the chip links to a real page, and the claim is
unsourced. No amount of retrieval quality prevents that; only checking does.

So: for every sentence of the answer, ask whether the chunk it cites actually
supports it, and report the verdict per sentence rather than per answer. An
answer is not uniformly true or false, and "paragraph 2 is unsupported" is not
actionable while "this clause is unsupported" is.

Verdicts
--------
``supported``     the cited chunk states the claim
``partial``       the chunk is related but does not fully state it
``unsupported``   the chunk does not support the claim — the citation is wrong
``uncited``       the sentence makes a factual claim and cites nothing
``not_a_claim``   framing, transitions, questions — nothing to verify

Faithfulness is the share of *claim* sentences that come out supported.
``not_a_claim`` sentences are excluded from the denominator: counting "Here is
what the sources say:" as a verified fact would inflate the score, and a
faithfulness metric that can be gamed by adding filler is worthless.
"""

from __future__ import annotations

import abc
import logging
import re
import time
from collections.abc import Mapping, Sequence

from kb.models import Answer, AnswerSentence, Chunk, SupportVerdict

log = logging.getLogger(__name__)

_MARKER_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")

#: Sentences with no verifiable content. Verifying them is noise, and counting
#: them as verified facts inflates faithfulness.
_NON_CLAIM_PATTERNS = (
    re.compile(r"^(here|below|above)\b.*:$", re.IGNORECASE),
    re.compile(r"^(in summary|to summarise|to summarize|in short|overall)\b", re.IGNORECASE),
    re.compile(r"^(let me know|feel free|hope this helps|if you)\b", re.IGNORECASE),
)

#: A sentence this short is a fragment or a heading, not a claim.
_MIN_CLAIM_CHARS = 25


def strip_markers(text: str) -> str:
    """Remove citation markers, leaving the claim as prose."""
    return re.sub(r"\s{2,}", " ", _MARKER_RE.sub("", text)).strip()


def is_claim(sentence: AnswerSentence) -> bool:
    """True when a sentence asserts something that could be checked.

    Deliberately conservative: over-classifying prose as a claim only produces a
    stricter faithfulness score, whereas under-classifying hides real failures.
    """
    body = strip_markers(sentence.text)
    if len(body) < _MIN_CLAIM_CHARS:
        return False
    if body.endswith("?"):
        return False
    return not any(pattern.match(body) for pattern in _NON_CLAIM_PATTERNS)


class Verifier(abc.ABC):
    """Scores whether a cited chunk supports a claim."""

    name: str = "verifier"

    def __init__(self, *, threshold: float = 0.5, partial_margin: float = 0.2) -> None:
        #: Support at or above this is ``supported``.
        self.threshold = threshold
        #: Support within this margin below the threshold is ``partial`` rather
        #: than ``unsupported`` — a hard cliff would report "the citation is
        #: wrong" for a claim the chunk mostly does state.
        self.partial_margin = partial_margin

    # ------------------------------------------------------------------ #
    # provider surface
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def support(self, claim: str, sources: Sequence[Chunk]) -> tuple[float, str | None, str | None]:
        """Return ``(score in [0,1], supporting_quote, note)`` for ``claim``."""

    # ------------------------------------------------------------------ #
    # shared pipeline
    # ------------------------------------------------------------------ #

    def verify(self, answer: Answer, sources: Mapping[str, Chunk] | None = None) -> Answer:
        """Annotate every sentence with a verdict, and the answer with faithfulness.

        ``sources`` maps chunk id to chunk. When omitted it is reconstructed from
        the answer's own retrieval result, so verification works on an answer
        alone — which is what lets it run as an independent stage, and lets the
        evaluation harness verify stored answers after the fact.

        Mutates and returns ``answer``, so verification composes with generation
        without a second representation of the same data.
        """
        started = time.perf_counter()

        if answer.refused:
            # A refusal has nothing to verify, and scoring it as unfaithful
            # would punish the system for behaving correctly.
            answer.verified = True
            answer.faithfulness = None
            for sentence in answer.sentences:
                sentence.verdict = SupportVerdict.NOT_A_CLAIM
            answer.timings_ms["verification_ms"] = _ms_since(started)
            return answer

        by_marker = {c.marker: c.chunk_id for c in answer.citations}
        lookup = dict(sources) if sources else _sources_from_retrieval(answer)

        supported = 0
        claims = 0
        for sentence in answer.sentences:
            if not is_claim(sentence):
                sentence.verdict = SupportVerdict.NOT_A_CLAIM
                continue

            claims += 1
            if not sentence.citation_markers:
                sentence.verdict = SupportVerdict.UNCITED
                sentence.support_score = 0.0
                sentence.verification_note = "the sentence makes a claim but cites no source"
                continue

            cited = self._resolve_sources(sentence, by_marker, lookup)
            if not cited:
                sentence.verdict = SupportVerdict.UNSUPPORTED
                sentence.support_score = 0.0
                sentence.verification_note = "cited source could not be resolved"
                continue

            claim = strip_markers(sentence.text)
            try:
                score, quote, note = self.support(claim, cited)
            except Exception as exc:
                log.warning("verifier %s failed on a sentence (%s)", self.name, exc)
                sentence.verification_note = f"verification unavailable: {exc}"
                continue

            sentence.support_score = round(float(score), 4)
            sentence.supporting_quote = quote
            if note:
                sentence.verification_note = note
            sentence.verdict = self._verdict(score)
            if sentence.verdict is SupportVerdict.SUPPORTED:
                supported += 1

        answer.verified = True
        answer.faithfulness = round(supported / claims, 4) if claims else None
        answer.timings_ms["verification_ms"] = _ms_since(started)
        return answer

    # ------------------------------------------------------------------ #

    def _verdict(self, score: float) -> SupportVerdict:
        if score >= self.threshold:
            return SupportVerdict.SUPPORTED
        if score >= self.threshold - self.partial_margin:
            return SupportVerdict.PARTIAL
        return SupportVerdict.UNSUPPORTED

    def _resolve_sources(
        self,
        sentence: AnswerSentence,
        by_marker: dict[int, str],
        lookup: Mapping[str, Chunk],
    ) -> list[Chunk]:
        out: list[Chunk] = []
        for marker in sentence.citation_markers:
            chunk = lookup.get(by_marker.get(marker, ""))
            if chunk is not None:
                out.append(chunk)
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r}, threshold={self.threshold})"


def _sources_from_retrieval(answer: Answer) -> dict[str, Chunk]:
    """Rebuild the chunk lookup from the answer's own retrieval result."""
    if answer.retrieval is None:
        return {}
    return {scored.chunk.id: scored.chunk for scored in answer.retrieval.results}


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)
