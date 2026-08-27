"""Offline cross-feature reranker.

This is the default because it needs no keys, no network and no model download,
and it is far from a stub: it computes the query-passage interaction features
that a first-stage retriever structurally cannot, because BM25 scores terms
independently and a bi-encoder never sees the pair together.

Five features, combined linearly:

1. **IDF-weighted term coverage.** How much of the query's *information* the
   passage covers, not how many words. Matching a rare term is worth more than
   matching a common one, and IDF is computed over the candidate set itself, so
   it adapts per query with no corpus statistics to maintain.
2. **Proximity.** The width of the smallest window containing all matched terms.
   "fusion … 400 words … ranks" and "fusion of ranks" score identically under
   BM25 and very differently here.
3. **Exact phrase match.** A contiguous run of ≥2 query terms is strong evidence,
   and it is the single clearest signal a bag-of-words model discards.
4. **Position of first match.** Passages that answer immediately beat passages
   that mention the topic in passing near the end.
5. **Heading match.** A hit in the heading path means the whole section is about
   the query, not just one sentence of it.

Weights are chosen for interpretability, not fitted — this reranker exists to be
a strong, explainable baseline that the evaluation harness can measure a hosted
cross-encoder *against*. `kb eval` is what decides whether the upgrade is worth
its latency on your corpus.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from kb.models import ChunkKind, ScoredChunk
from kb.rerank.base import Reranker
from kb.retrieval.lexical import extract_terms

_WORD_RE = re.compile(r"[a-z0-9_]+(?:'[a-z]+)?")

# Feature weights. Coverage dominates: a passage that does not cover the query
# cannot be the best answer regardless of how tidily its few matches cluster.
W_COVERAGE = 1.00
W_PROXIMITY = 0.35
W_PHRASE = 0.30
W_POSITION = 0.15
W_HEADING = 0.20

#: Beyond this many tokens, proximity stops being informative.
_PROXIMITY_HORIZON = 120


class LexicalReranker(Reranker):
    """Deterministic query-passage feature reranker. Always available."""

    name = "lexical-rerank"

    def __init__(
        self,
        *,
        code_penalty: float = 0.05,
        length_penalty: float = 0.0,
    ) -> None:
        # Code chunks match identifier-shaped query terms very easily; a small
        # penalty stops a config file outranking the prose that explains it,
        # unless the query is itself code-shaped (handled below).
        self.code_penalty = code_penalty
        self.length_penalty = length_penalty

    # ------------------------------------------------------------------ #

    def score(self, query: str, candidates: Sequence[ScoredChunk]) -> list[float]:
        terms = extract_terms(query)
        if not terms or not candidates:
            return [c.score for c in candidates]

        tokenised = [_tokenize(c.chunk.text) for c in candidates]
        idf = _candidate_idf(terms, tokenised)
        query_is_codey = _looks_like_code_query(query)

        scores: list[float] = []
        for candidate, tokens in zip(candidates, tokenised, strict=True):
            positions = _term_positions(terms, tokens)
            coverage = self._coverage(terms, positions, idf)
            proximity = _proximity(positions)
            phrase = _phrase_bonus(terms, tokens)
            position = _first_match_position(positions, len(tokens))
            heading = self._heading_bonus(terms, candidate)

            score = (
                W_COVERAGE * coverage
                + W_PROXIMITY * proximity
                + W_PHRASE * phrase
                + W_POSITION * position
                + W_HEADING * heading
            )

            if candidate.chunk.kind is ChunkKind.CODE and not query_is_codey:
                score -= self.code_penalty
            if self.length_penalty:
                score -= self.length_penalty * math.log1p(len(tokens) / 200.0)

            scores.append(score)
        return scores

    # ------------------------------------------------------------------ #

    def _coverage(
        self, terms: Sequence[str], positions: dict[str, list[int]], idf: dict[str, float]
    ) -> float:
        total = sum(idf[t] for t in terms) or 1.0
        matched = sum(idf[t] for t in terms if positions.get(t))
        return matched / total

    def _heading_bonus(self, terms: Sequence[str], candidate: ScoredChunk) -> float:
        haystack = (f"{candidate.chunk.heading_context} {candidate.chunk.document_title}").lower()
        if not haystack.strip():
            return 0.0
        hits = sum(1 for term in terms if term in haystack)
        return hits / len(terms)


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _candidate_idf(terms: Sequence[str], tokenised: Sequence[list[str]]) -> dict[str, float]:
    """IDF over the candidate set.

    Computing it here rather than over the corpus means the weighting reflects
    what actually distinguishes *these* candidates — which is the only question
    at rerank time — and needs no maintained statistics.
    """
    n = len(tokenised) or 1
    document_frequency: Counter[str] = Counter()
    for tokens in tokenised:
        present = set(tokens)
        for term in terms:
            if term in present or any(tok.startswith(term) for tok in present):
                document_frequency[term] += 1
    return {
        term: math.log(
            1.0 + (n - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)
        )
        for term in terms
    }


def _term_positions(terms: Sequence[str], tokens: Sequence[str]) -> dict[str, list[int]]:
    """Token indices where each query term occurs, allowing prefix matches."""
    positions: dict[str, list[int]] = {}
    for index, token in enumerate(tokens):
        for term in terms:
            if token == term or (len(term) >= 5 and token.startswith(term)):
                positions.setdefault(term, []).append(index)
    return positions


def _proximity(positions: dict[str, list[int]]) -> float:
    """1.0 when all matched terms are adjacent, decaying with window width."""
    matched = [p for p in positions.values() if p]
    if len(matched) < 2:
        return 1.0 if matched else 0.0
    span = _min_window(matched)
    if span is None:
        return 0.0
    ideal = len(matched)
    excess = max(0, span - ideal)
    return max(0.0, 1.0 - excess / _PROXIMITY_HORIZON)


def _min_window(position_lists: list[list[int]]) -> int | None:
    """Width of the narrowest window containing one position from each list.

    Classic k-way merge: advance the pointer sitting on the smallest position,
    since that is the only move that can shrink the window.
    """
    pointers = [0] * len(position_lists)
    best: int | None = None
    while True:
        current = [position_lists[i][pointers[i]] for i in range(len(position_lists))]
        lo, hi = min(current), max(current)
        width = hi - lo + 1
        if best is None or width < best:
            best = width
        smallest = current.index(lo)
        pointers[smallest] += 1
        if pointers[smallest] >= len(position_lists[smallest]):
            return best


def _phrase_bonus(terms: Sequence[str], tokens: Sequence[str]) -> float:
    """Length of the longest contiguous run of query terms, normalised."""
    if len(terms) < 2:
        return 0.0
    term_set = set(terms)
    longest = current = 0
    for token in tokens:
        if token in term_set:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    if longest < 2:
        return 0.0
    return min(1.0, longest / len(terms))


def _first_match_position(positions: dict[str, list[int]], n_tokens: int) -> float:
    """1.0 when the first match is at the start, decaying to 0 at the end."""
    firsts = [p[0] for p in positions.values() if p]
    if not firsts or n_tokens <= 0:
        return 0.0
    return max(0.0, 1.0 - min(firsts) / max(n_tokens, 1))


def _looks_like_code_query(query: str) -> bool:
    """True when the query is itself code-shaped, so code chunks are wanted."""
    return bool(
        re.search(r"[(){}\[\];]|::|->|=>|\bdef\b|\bclass\b|\bimport\b|_[a-z]|[a-z][A-Z]", query)
    )
