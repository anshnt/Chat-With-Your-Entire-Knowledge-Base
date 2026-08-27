"""Offline extractive generator — the default.

It does not write prose. It **selects** the sentences from the retrieved chunks
that best answer the question, and cites each one to the chunk it came from. That
is a deliberate choice rather than a limitation: an extractive answer is
*trivially* faithful, because every sentence is verbatim from a source. There is
no mechanism by which it can hallucinate.

Which makes it the right default for a system whose point is verified citations:

* it needs no key, no network and no model, so the demo, the tests and CI all
  exercise the real end-to-end path including citation resolution;
* it is deterministic, so citation-verification tests can assert exact verdicts;
* it gives the evaluation harness a genuine floor to measure an LLM against —
  "the LLM adds 12 points of answer quality" is only meaningful against a
  baseline that actually tries.

Sentence selection is MMR over the candidate sentences: relevance to the query,
minus redundancy against what is already selected. Without the redundancy term
the answer becomes the same fact restated from four overlapping chunks, which is
the characteristic failure of naive extractive summarisation.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from kb.chunking.base import split_sentences
from kb.generate.base import Generator
from kb.models import Chunk
from kb.retrieval.lexical import extract_terms

_WORD_RE = re.compile(r"[a-z0-9_]+")

#: Sentences shorter than this are usually fragments — headings, list markers,
#: "See below." — and make a poor answer even when they score well.
MIN_SENTENCE_CHARS = 30
MAX_SENTENCE_CHARS = 600

#: How much a sentence's own chunk's retrieval rank contributes. Retrieval
#: already did work; ignoring it entirely throws that away. It only ever breaks
#: ties *among sentences that already clear the relevance floor* — letting it
#: rescue an irrelevant sentence is how an extractive answer ends up padded with
#: whatever happened to be retrieved.
RANK_PRIOR_WEIGHT = 0.25
MMR_LAMBDA = 0.72

#: Minimum share of the query's IDF mass a sentence must cover to be used.
#: Without this floor, "what is the capital of France?" against a retrieval
#: corpus produces a confident, fully-cited answer about reranking — the single
#: most damaging thing a grounded system can do.
MIN_RELEVANCE = 0.25


class ExtractiveGenerator(Generator):
    """Answers by selecting and citing the most relevant retrieved sentences."""

    name = "extractive"
    model = "extractive-v1"
    supports_streaming = False

    def __init__(
        self,
        *,
        token_budget: int = 6000,
        max_chunks: int | None = None,
        max_sentences: int = 4,
    ) -> None:
        super().__init__(token_budget=token_budget, max_chunks=max_chunks)
        self.max_sentences = max_sentences

    # ------------------------------------------------------------------ #

    def _generate_text(self, query: str, chunks: Sequence[Chunk]) -> str:
        terms = extract_terms(query)
        pool = self._candidate_sentences(chunks)
        if not pool:
            return _no_answer_text()

        idf = _idf(terms, [tokens for _, _, tokens in pool])
        scored: list[tuple[float, int, str, list[str]]] = []
        for marker, sentence, tokens in pool:
            relevance = self._relevance(terms, tokens, idf)
            if relevance < MIN_RELEVANCE:
                continue
            rank_prior = RANK_PRIOR_WEIGHT * _rank_prior(marker, len(chunks))
            scored.append((relevance + rank_prior, marker, sentence, tokens))

        if not scored:
            # Nothing in the retrieved context addresses the query. Saying so is
            # the correct answer, and the only honest one.
            return _no_answer_text()

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = self._select_diverse(scored)
        # Present in source order so the answer reads as a coherent passage
        # rather than a relevance-ranked list.
        selected.sort(key=lambda item: (item[1], item[3]))
        return " ".join(
            f"{sentence} [{marker}]" for _, marker, sentence, _, _ in _reindex(selected)
        )

    # ------------------------------------------------------------------ #

    def _candidate_sentences(self, chunks: Sequence[Chunk]) -> list[tuple[int, str, list[str]]]:
        """``(marker, sentence, tokens)`` for every usable sentence in context."""
        out: list[tuple[int, str, list[str]]] = []
        for marker, chunk in enumerate(chunks, start=1):
            body = _strip_heading_prefix(chunk)
            # Split on blank lines before splitting on sentences. A list has no
            # sentence terminator before it, so a single split_sentences pass
            # swallows the whole list into the sentence that introduces it —
            # producing an answer like "This is a design decision: - item one; -
            # item two", which reads as broken.
            for block in _paragraphs(body):
                for sentence in split_sentences(block):
                    cleaned = " ".join(sentence.split())
                    if not MIN_SENTENCE_CHARS <= len(cleaned) <= MAX_SENTENCE_CHARS:
                        continue
                    if _is_boilerplate(cleaned):
                        continue
                    out.append((marker, cleaned, _WORD_RE.findall(cleaned.lower())))
        return out

    def _relevance(
        self, terms: Sequence[str], tokens: Sequence[str], idf: dict[str, float]
    ) -> float:
        """IDF-weighted share of the query's information present in the sentence."""
        if not terms:
            return 0.0
        present = set(tokens)
        # Only terms present somewhere in the candidate pool carry weight.
        total = sum(idf.values()) or 1.0
        matched = sum(
            weight
            for term, weight in idf.items()
            if term in present or (len(term) >= 5 and any(t.startswith(term) for t in present))
        )
        return matched / total

    def _select_diverse(
        self, scored: list[tuple[float, int, str, list[str]]]
    ) -> list[tuple[float, int, str, list[str]]]:
        """MMR over sentences: relevance minus redundancy against the selection."""
        selected: list[tuple[float, int, str, list[str]]] = []
        remaining = list(scored)
        while remaining and len(selected) < self.max_sentences:
            best_index, best_value = 0, -math.inf
            for index, (relevance, _, _, tokens) in enumerate(remaining):
                redundancy = max((_jaccard(tokens, chosen[3]) for chosen in selected), default=0.0)
                value = MMR_LAMBDA * relevance - (1.0 - MMR_LAMBDA) * redundancy
                if value > best_value:
                    best_index, best_value = index, value
            candidate = remaining.pop(best_index)
            # A sentence that is nearly a duplicate of one already chosen adds
            # length without adding an answer.
            if any(_jaccard(candidate[3], chosen[3]) > 0.8 for chosen in selected):
                continue
            selected.append(candidate)
        return selected


def _reindex(
    selected: list[tuple[float, int, str, list[str]]],
) -> list[tuple[float, int, str, list[str], int]]:
    """Attach the sentence's position so the sort is stable and explicit."""
    return [
        (score, marker, sentence, tokens, i)
        for i, (score, marker, sentence, tokens) in enumerate(selected)
    ]


def _idf(terms: Sequence[str], token_lists: Sequence[Sequence[str]]) -> dict[str, float]:
    """IDF over the candidate sentences, so weighting adapts to this context.

    A term appearing in *no* candidate is dropped, not given the maximal weight
    the formula would otherwise assign it. IDF measures how well a term
    discriminates *within this pool*, and a term present nowhere discriminates
    nothing — while a large weight on it drags every candidate below the
    relevance floor and makes the generator refuse a question it could answer.

    Concretely: asking "what is the hashing embedder" of a passage saying
    "HashingEmbedder is signed feature hashing…" matches on `hashing` but not on
    `embedder`. Weighting the absent term maximally scored that 0.17 and refused;
    dropping it scores 1.0 and answers.
    """
    n = len(token_lists) or 1
    frequency: dict[str, int] = dict.fromkeys(terms, 0)
    for tokens in token_lists:
        present = set(tokens)
        for term in terms:
            if term in present or (len(term) >= 5 and any(t.startswith(term) for t in present)):
                frequency[term] += 1
    return {
        term: math.log(1.0 + (n - frequency[term] + 0.5) / (frequency[term] + 0.5))
        for term in terms
        if frequency[term] > 0
    }


def _rank_prior(marker: int, n_chunks: int) -> float:
    """1.0 for the top-ranked chunk, decaying linearly."""
    if n_chunks <= 1:
        return 1.0
    return max(0.0, 1.0 - (marker - 1) / n_chunks)


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def _strip_heading_prefix(chunk: Chunk) -> str:
    """Drop the ``A › B › C`` prefix the chunker added.

    It is valuable for retrieval and useless in an answer — quoting it back would
    read as a fragment.
    """
    text = chunk.text
    if chunk.heading_context and text.startswith(chunk.heading_context):
        return text[len(chunk.heading_context) :].lstrip("\n ")
    return text


_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s")


def _paragraphs(text: str) -> list[str]:
    """Split into paragraphs, dropping list blocks and fenced code.

    A list item is not a sentence, and a paragraph that introduces one
    ("…for two reasons:") is not an answer on its own — so both are skipped
    rather than stitched together.
    """
    blocks: list[str] = []
    in_fence = False
    for raw_block in re.split(r"\n\s*\n", text):
        block = raw_block.strip()
        if not block:
            continue
        fence_lines = sum(1 for line in block.splitlines() if line.lstrip().startswith("```"))
        if fence_lines % 2 == 1:
            in_fence = not in_fence
            continue
        if in_fence or block.startswith("```"):
            continue
        lines = [line for line in block.splitlines() if line.strip()]
        # A block that is mostly list items is a list, however it wrapped.
        list_lines = sum(1 for line in lines if _LIST_ITEM_RE.match(line))
        if lines and list_lines / len(lines) > 0.4:
            continue
        blocks.append(block)
    return blocks


def _is_boilerplate(sentence: str) -> bool:
    """Filter out fragments that score well but read as noise."""
    if sentence.startswith(("|", "```", "- ", "* ", "#")):
        return True
    if sentence.count("|") >= 3:  # a table row
        return True
    # A flattened list that survived paragraph splitting: "…reasons: - a - b".
    if sentence.count(" - ") >= 2 or ": -" in sentence:
        return True
    # A sentence that only introduces something is not an answer by itself.
    return sentence.endswith(":")


def _no_answer_text() -> str:
    return (
        "The retrieved sources do not contain an answer to this question. "
        "Try rephrasing it, or ingest a source that covers it."
    )
