"""Lexical retrieval over SQLite FTS5 (Okapi BM25).

Two things matter here beyond "call MATCH":

**Query sanitisation.** FTS5's MATCH grammar treats ``"``, ``*``, ``:``, ``^``,
``AND/OR/NOT``, and parentheses as operators. A raw user question like
``What is "hybrid search"? (RRF)`` is a syntax error, not a query. Every term is
therefore extracted and re-quoted rather than escaped in place.

**Recall under an exact-match engine.** BM25 needs the terms to actually occur.
A strict AND over every term returns nothing for most natural-language
questions, and a plain OR ranks a document matching one stopword alongside one
matching everything. The compromise used here: OR over terms, with a prefix
variant for the longer terms so ``retriev`` reaches ``retrieval`` and
``retrieving`` beyond what the porter stemmer catches, and stopwords dropped so
they neither dilute the query nor eliminate it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from kb.models import Chunk, ScoredChunk, SourceType
from kb.store import SQLiteStore

_TERM_RE = re.compile(r"[A-Za-z0-9_]+(?:[.\-'][A-Za-z0-9_]+)*")

# Deliberately small: an aggressive stoplist strips meaningful words from short
# queries ("who is on call", "how to do X"). These are the words that carry no
# retrieval signal in essentially any corpus.
STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "for",
        "from",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "ours",
        "she",
        "should",
        "so",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "too",
        "us",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
    ]
)

_MIN_PREFIX_LEN = 5


def extract_terms(query: str, *, keep_stopwords: bool = False) -> list[str]:
    """Pull safe, deduplicated search terms out of a free-text query."""
    seen: dict[str, None] = {}
    for match in _TERM_RE.finditer(query):
        term = match.group(0).strip("'-.").lower()
        if len(term) < 2:
            continue
        if not keep_stopwords and term in STOPWORDS:
            continue
        seen.setdefault(term, None)
    if not seen and not keep_stopwords:
        # Query was entirely stopwords ("what is it") — better to search them
        # than to return nothing.
        return extract_terms(query, keep_stopwords=True)
    return list(seen)


def build_match_query(query: str, *, prefix_expansion: bool = True) -> str:
    """Compile a free-text query into a valid FTS5 MATCH expression.

    Phrases the caller quoted are preserved as phrase matches, since an explicit
    quote is a strong signal that adjacency is intended.
    """
    phrases = [p.strip() for p in re.findall(r'"([^"]+)"', query) if p.strip()]
    residual = re.sub(r'"[^"]*"', " ", query)
    terms = extract_terms(residual)

    clauses: list[str] = []
    for phrase in phrases:
        safe = " ".join(extract_terms(phrase, keep_stopwords=True))
        if safe:
            clauses.append(f'"{safe}"')
    for term in terms:
        if prefix_expansion and len(term) >= _MIN_PREFIX_LEN:
            clauses.append(f'("{term}" OR "{term}"*)')
        else:
            clauses.append(f'"{term}"')

    if not clauses:
        return ""
    return " OR ".join(clauses)


class LexicalRetriever:
    """BM25 retrieval with the score normalisation the fusion layer expects."""

    name = "lexical"

    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def search(
        self,
        query: str,
        *,
        collection: str = "default",
        limit: int = 50,
        source_types: Sequence[SourceType] | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        match_query = build_match_query(query)
        if not match_query:
            return []
        hits = self.store.search_lexical(
            match_query,
            collection=collection,
            limit=limit,
            source_types=[s.value for s in source_types] if source_types else None,
            document_ids=list(document_ids) if document_ids else None,
        )
        if not hits:
            return []
        chunks = {c.id: c for c in self.store.get_chunks([cid for cid, _ in hits])}
        out: list[ScoredChunk] = []
        for rank, (chunk_id, score) in enumerate(hits, start=1):
            chunk = chunks.get(chunk_id)
            if chunk is None:
                continue
            out.append(
                ScoredChunk(
                    chunk=chunk,
                    score=score,
                    lexical_score=score,
                    lexical_rank=rank,
                    retrievers=[self.name],
                )
            )
        return out


def keyword_coverage(query: str, chunk: Chunk) -> float:
    """Fraction of the query's content terms present in ``chunk``.

    Used as a tie-breaker and by the offline reranker. It is the cheapest useful
    proxy for "does this chunk actually talk about what was asked".
    """
    terms = set(extract_terms(query))
    if not terms:
        return 0.0
    haystack = f"{chunk.heading_context}\n{chunk.text}".lower()
    present = sum(1 for term in terms if term in haystack)
    return present / len(terms)
