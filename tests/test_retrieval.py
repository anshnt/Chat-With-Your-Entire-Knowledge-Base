"""Retrieval tests: query compilation, fusion algebra, MMR, and the pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from kb.chunking.base import tokenize_words
from kb.models import (
    Chunk,
    FusionMethod,
    RetrievalRequest,
    RetrievalStrategy,
    ScoredChunk,
    SourceType,
    TextLocator,
)
from kb.retrieval.fusion import max_fusion, reciprocal_rank_fusion, weighted_fusion
from kb.retrieval.lexical import build_match_query, extract_terms, keyword_coverage
from kb.retrieval.mmr import mmr_rerank


def scored(
    chunk_id: str,
    score: float,
    *,
    lexical=None,
    dense=None,
    lex_rank=None,
    dense_rank=None,
    text="text",
):
    chunk = Chunk(
        id=chunk_id,
        document_id="doc_1",
        ordinal=0,
        text=text,
        locator=TextLocator(line_start=1, line_end=1),
        document_title="Doc",
        source_type=SourceType.MARKDOWN,
    )
    return ScoredChunk(
        chunk=chunk,
        score=score,
        lexical_score=lexical,
        dense_score=dense,
        lexical_rank=lex_rank,
        dense_rank=dense_rank,
        retrievers=["lexical"] if lexical is not None else ["dense"],
    )


class TestExtractTerms:
    def test_drops_stopwords_and_lowercases(self) -> None:
        assert extract_terms("What is the Reciprocal Rank Fusion?") == [
            "reciprocal",
            "rank",
            "fusion",
        ]

    def test_deduplicates_preserving_order(self) -> None:
        assert extract_terms("fusion rank fusion") == ["fusion", "rank"]

    def test_keeps_identifiers_and_versions(self) -> None:
        assert extract_terms("error E1234 in v2.1-beta") == ["error", "e1234", "v2.1-beta"]

    def test_all_stopword_query_falls_back_to_stopwords(self) -> None:
        # Better to search "what is it" than to return nothing at all.
        assert extract_terms("what is it") == ["what", "is", "it"]

    def test_single_characters_are_dropped(self) -> None:
        assert extract_terms("a b retrieval") == ["retrieval"]


class TestBuildMatchQuery:
    def test_terms_are_or_combined(self) -> None:
        query = build_match_query("rank fusion")
        assert " OR " in query
        assert '"rank"' in query

    def test_long_terms_get_prefix_expansion(self) -> None:
        assert '("fusion" OR "fusion"*)' in build_match_query("fusion")

    def test_short_terms_do_not(self) -> None:
        query = build_match_query("bm25")
        assert '"bm25"' in query
        assert '"bm25"*' not in query

    def test_quoted_phrases_stay_phrases(self) -> None:
        query = build_match_query('"hybrid search" ranking')
        assert '"hybrid search"' in query

    @pytest.mark.parametrize(
        "hostile",
        [
            'What is "hybrid search"? (RRF)',
            "NEAR(a b) AND NOT c",
            "col:value ^caret *star",
            "a OR OR b",
            '")))"',
            "",
        ],
    )
    def test_operator_characters_never_reach_fts(self, hostile: str) -> None:
        """FTS5 syntax must never leak from user input — it would be an error, not a query."""
        query = build_match_query(hostile)
        # Every quote must be balanced and no bare operator characters remain.
        assert query.count('"') % 2 == 0
        for char in "^:*()":
            if char == "*":
                continue  # only ever emitted by our own prefix expansion
            assert char not in query.replace('("', "").replace('")', "") or char in "()"


class TestKeywordCoverage:
    def _chunk(self, text: str) -> Chunk:
        return Chunk(
            document_id="d",
            ordinal=0,
            text=text,
            locator=TextLocator(line_start=1, line_end=1),
        )

    def test_full_coverage(self) -> None:
        assert keyword_coverage("rank fusion", self._chunk("rank fusion works")) == 1.0

    def test_partial_coverage(self) -> None:
        assert keyword_coverage("rank fusion", self._chunk("only rank here")) == 0.5

    def test_no_terms_scores_zero(self) -> None:
        assert keyword_coverage("", self._chunk("anything")) == 0.0


class TestReciprocalRankFusion:
    def test_agreement_beats_a_single_top_hit(self) -> None:
        """The chunk both retrievers like should beat one retriever's favourite."""
        lexical = [
            scored("a", 10.0, lexical=10.0, lex_rank=1),
            scored("b", 9.0, lexical=9.0, lex_rank=2),
        ]
        dense = [
            scored("c", 0.9, dense=0.9, dense_rank=1),
            scored("b", 0.8, dense=0.8, dense_rank=2),
        ]
        fused = reciprocal_rank_fusion([lexical, dense], k=60)
        assert fused[0].chunk.id == "b"

    def test_ranks_not_scores_decide(self) -> None:
        """A huge BM25 score must not dominate; only its position counts.

        RRF re-derives ranks from list position rather than trusting the stored
        rank, so a filtered or truncated list still fuses correctly.
        """
        lexical = [
            scored("filler", 10000.0, lexical=10000.0, lex_rank=1),
            scored("a", 9999.0, lexical=9999.0, lex_rank=2),
        ]
        dense = [scored("b", 0.01, dense=0.01, dense_rank=1)]
        fused = reciprocal_rank_fusion([lexical, dense], k=60, weights=[0.5, 0.5])
        # "b" is rank 1 of its list and beats "a" at rank 2, despite the
        # six-orders-of-magnitude score difference.
        ranking = [r.chunk.id for r in fused]
        assert ranking.index("b") < ranking.index("a")

    def test_provenance_is_merged(self) -> None:
        lexical = [scored("b", 9.0, lexical=9.0, lex_rank=1)]
        dense = [scored("b", 0.8, dense=0.8, dense_rank=3)]
        fused = reciprocal_rank_fusion([lexical, dense], k=60)
        assert fused[0].lexical_score == 9.0
        assert fused[0].dense_score == 0.8
        assert set(fused[0].retrievers) == {"lexical", "dense"}

    def test_weights_shift_the_balance(self) -> None:
        lexical = [scored("a", 1.0, lexical=1.0, lex_rank=1)]
        dense = [scored("b", 1.0, dense=1.0, dense_rank=1)]
        assert reciprocal_rank_fusion([lexical, dense], weights=[0.9, 0.1])[0].chunk.id == "a"
        assert reciprocal_rank_fusion([lexical, dense], weights=[0.1, 0.9])[0].chunk.id == "b"

    def test_smaller_k_sharpens_top_ranks(self) -> None:
        lexical = [scored(str(i), 1.0, lexical=1.0, lex_rank=i) for i in range(1, 6)]
        gap_small_k = reciprocal_rank_fusion([lexical], k=1)
        gap_large_k = reciprocal_rank_fusion([lexical], k=1000)
        spread_small = gap_small_k[0].score - gap_small_k[1].score
        spread_large = gap_large_k[0].score - gap_large_k[1].score
        assert spread_small > spread_large

    def test_mismatched_weights_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="weights must match"):
            reciprocal_rank_fusion([[], []], weights=[1.0])

    def test_empty_input(self) -> None:
        assert reciprocal_rank_fusion([[], []]) == []


class TestWeightedFusion:
    def test_normalises_incomparable_scales(self) -> None:
        lexical = [
            scored("a", 100.0, lexical=100.0, lex_rank=1),
            scored("b", 50.0, lexical=50.0, lex_rank=2),
        ]
        dense = [
            scored("b", 0.9, dense=0.9, dense_rank=1),
            scored("a", 0.1, dense=0.1, dense_rank=2),
        ]
        fused = weighted_fusion(lexical, dense, lexical_weight=0.5, dense_weight=0.5)
        # a: 1.0*0.5 + 0.0*0.5 = 0.5 ; b: 0.0*0.5 + 1.0*0.5 = 0.5 -> tie broken by id
        assert {r.chunk.id for r in fused} == {"a", "b"}
        assert pytest.approx(fused[0].score, abs=1e-6) == 0.5

    def test_all_equal_scores_do_not_divide_by_zero(self) -> None:
        lexical = [
            scored("a", 5.0, lexical=5.0, lex_rank=1),
            scored("b", 5.0, lexical=5.0, lex_rank=2),
        ]
        fused = weighted_fusion(lexical, [], lexical_weight=1.0, dense_weight=0.0)
        assert all(r.score == 1.0 for r in fused)

    def test_dense_weight_dominates_when_set(self) -> None:
        lexical = [scored("a", 10.0, lexical=10.0, lex_rank=1)]
        dense = [scored("b", 0.9, dense=0.9, dense_rank=1)]
        fused = weighted_fusion(lexical, dense, lexical_weight=0.1, dense_weight=0.9)
        assert fused[0].chunk.id == "b"


class TestMaxFusion:
    def test_takes_the_better_normalised_score(self) -> None:
        lexical = [
            scored("a", 10.0, lexical=10.0, lex_rank=1),
            scored("b", 1.0, lexical=1.0, lex_rank=2),
        ]
        dense = [
            scored("b", 0.9, dense=0.9, dense_rank=1),
            scored("c", 0.1, dense=0.1, dense_rank=2),
        ]
        fused = max_fusion(lexical, dense)
        by_id = {r.chunk.id: r.score for r in fused}
        assert by_id["a"] == pytest.approx(1.0)
        assert by_id["b"] == pytest.approx(1.0)
        assert by_id["c"] == pytest.approx(0.0)


class TestMMR:
    def test_removes_near_duplicates(self) -> None:
        results = [
            scored("a", 1.0, text="reciprocal rank fusion combines ranked lists"),
            scored("a2", 0.99, text="reciprocal rank fusion combines ranked lists"),
            scored("b", 0.5, text="cosine similarity over normalised dense vectors"),
        ]
        picked = mmr_rerank(results, top_k=2, lambda_=0.5)
        ids = [r.chunk.id for r in picked]
        assert ids[0] == "a"
        assert ids[1] == "b", "the duplicate should lose to the diverse chunk"

    def test_lambda_one_is_pure_relevance(self) -> None:
        results = [
            scored("a", 1.0, text="same text here"),
            scored("a2", 0.99, text="same text here"),
            scored("b", 0.5, text="totally different content"),
        ]
        picked = mmr_rerank(results, top_k=2, lambda_=1.0)
        assert [r.chunk.id for r in picked] == ["a", "a2"]

    def test_uses_vectors_when_available(self) -> None:
        results = [scored("a", 1.0), scored("b", 0.9), scored("c", 0.8)]
        vectors = {
            "a": np.array([1.0, 0.0], dtype="float32"),
            "b": np.array([1.0, 0.0], dtype="float32"),  # identical to a
            "c": np.array([0.0, 1.0], dtype="float32"),  # orthogonal
        }
        picked = mmr_rerank(results, top_k=2, lambda_=0.5, vectors=vectors)
        assert [r.chunk.id for r in picked] == ["a", "c"]

    def test_empty_input(self) -> None:
        assert mmr_rerank([], top_k=5) == []

    def test_top_k_larger_than_candidates(self) -> None:
        results = [scored("a", 1.0), scored("b", 0.5)]
        assert len(mmr_rerank(results, top_k=10, lambda_=0.7)) == 2


class TestRetrievalRequest:
    def test_blank_query_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="blank"):
            RetrievalRequest(query="   ")

    def test_query_is_stripped(self) -> None:
        assert RetrievalRequest(query="  fusion  ").query == "fusion"

    def test_defaults(self) -> None:
        request = RetrievalRequest(query="q")
        assert request.strategy is RetrievalStrategy.HYBRID
        assert request.fusion is FusionMethod.RRF
        assert request.candidate_k > request.top_k


def test_tokenize_words_handles_contractions_and_code() -> None:
    assert tokenize_words("don't split_this CamelCase 42") == [
        "don't",
        "split",
        "this",
        "camelcase",
        "42",
    ]
