"""Reranking tests.

The interesting properties are behavioural, not numeric: does the reranker prefer
the passage that answers the question over the one that merely shares its
vocabulary, does it degrade safely when it fails, and does it never lose a
candidate.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from kb.config import RerankProvider, Settings
from kb.knowledge_base import KnowledgeBase
from kb.models import Chunk, ChunkKind, ScoredChunk, SourceType, TextLocator
from kb.rerank import build_reranker
from kb.rerank.base import NoOpReranker, Reranker
from kb.rerank.lexical import (
    LexicalReranker,
    _min_window,
    _phrase_bonus,
    _proximity,
    _term_positions,
)
from kb.rerank.llm import parse_ordering


def candidate(
    chunk_id: str,
    text: str,
    *,
    score: float = 0.5,
    heading: str = "",
    kind: ChunkKind = ChunkKind.PROSE,
    title: str = "Doc",
) -> ScoredChunk:
    chunk = Chunk(
        id=chunk_id,
        document_id="doc_1",
        ordinal=0,
        text=text,
        kind=kind,
        locator=TextLocator(line_start=1, line_end=1),
        document_title=title,
        source_type=SourceType.MARKDOWN,
        heading_context=heading,
    )
    return ScoredChunk(chunk=chunk, score=score, fusion_score=score, retrievers=["lexical"])


# --------------------------------------------------------------------------- #
# feature functions
# --------------------------------------------------------------------------- #


class TestMinWindow:
    def test_adjacent_terms(self) -> None:
        assert _min_window([[0], [1]]) == 2

    def test_picks_the_narrowest_window(self) -> None:
        # 'a' at 0 and 50, 'b' at 51 -> narrowest window is 50..51
        assert _min_window([[0, 50], [51]]) == 2

    def test_single_list(self) -> None:
        assert _min_window([[5]]) == 1

    def test_three_way(self) -> None:
        assert _min_window([[0, 10], [11], [12]]) == 3


class TestProximity:
    def test_adjacent_terms_score_one(self) -> None:
        positions = {"a": [0], "b": [1]}
        assert _proximity(positions) == pytest.approx(1.0)

    def test_distant_terms_score_lower(self) -> None:
        near = _proximity({"a": [0], "b": [1]})
        far = _proximity({"a": [0], "b": [80]})
        assert far < near

    def test_no_matches_scores_zero(self) -> None:
        assert _proximity({}) == 0.0

    def test_single_matched_term_scores_one(self) -> None:
        assert _proximity({"a": [5]}) == 1.0


class TestPhraseBonus:
    def test_contiguous_run_is_rewarded(self) -> None:
        assert _phrase_bonus(["rank", "fusion"], ["reciprocal", "rank", "fusion"]) > 0

    def test_scattered_terms_get_nothing(self) -> None:
        assert _phrase_bonus(["rank", "fusion"], ["rank", "x", "y", "fusion"]) == 0.0

    def test_single_term_query_has_no_phrase(self) -> None:
        assert _phrase_bonus(["rank"], ["rank", "rank"]) == 0.0


class TestTermPositions:
    def test_exact_matches(self) -> None:
        positions = _term_positions(["rank"], ["the", "rank", "of", "rank"])
        assert positions == {"rank": [1, 3]}

    def test_long_terms_match_by_prefix(self) -> None:
        positions = _term_positions(["fusion"], ["fusions", "later"])
        assert positions["fusion"] == [0]

    def test_short_terms_do_not_prefix_match(self) -> None:
        assert _term_positions(["ran"], ["random"]) == {}


# --------------------------------------------------------------------------- #
# lexical reranker
# --------------------------------------------------------------------------- #


class TestLexicalReranker:
    def test_prefers_the_answer_over_topical_noise(self) -> None:
        """The whole point: vocabulary overlap is not an answer."""
        reranker = LexicalReranker()
        candidates = [
            candidate("noise", "Ranking, ranking, ranking. Constants are discussed at length."),
            candidate("answer", "The RRF damping constant defaults to 60."),
        ]
        ranked = reranker.rerank("what does the RRF damping constant default to", candidates)
        assert ranked[0].chunk.id == "answer"

    def test_proximity_breaks_a_coverage_tie(self) -> None:
        reranker = LexicalReranker()
        filler = " ".join(["padding"] * 60)
        candidates = [
            candidate("scattered", f"fusion {filler} ranks"),
            candidate("adjacent", "fusion of ranks is the mechanism"),
        ]
        ranked = reranker.rerank("fusion ranks", candidates)
        assert ranked[0].chunk.id == "adjacent"

    def test_heading_match_is_rewarded(self) -> None:
        reranker = LexicalReranker()
        body = "This section explains the mechanism in general terms."
        candidates = [
            candidate("plain", body),
            candidate("headed", body, heading="Retrieval › Fusion"),
        ]
        ranked = reranker.rerank("fusion", candidates)
        assert ranked[0].chunk.id == "headed"

    def test_early_answers_beat_buried_ones(self) -> None:
        reranker = LexicalReranker()
        preamble = " ".join(["unrelated"] * 80)
        candidates = [
            candidate("buried", f"{preamble} the damping constant is mentioned"),
            candidate("early", "The damping constant is mentioned right away here."),
        ]
        ranked = reranker.rerank("damping constant", candidates)
        assert ranked[0].chunk.id == "early"

    def test_code_chunks_are_penalised_for_prose_queries(self) -> None:
        reranker = LexicalReranker(code_penalty=0.3)
        text = "the fusion constant value"
        candidates = [
            candidate("code", text, kind=ChunkKind.CODE),
            candidate("prose", text, kind=ChunkKind.PROSE),
        ]
        ranked = reranker.rerank("what is the fusion constant value", candidates)
        assert ranked[0].chunk.id == "prose"

    def test_code_queries_do_not_penalise_code(self) -> None:
        reranker = LexicalReranker(code_penalty=0.3)
        text = "def fuse(rankings): return merged"
        candidates = [candidate("code", text, kind=ChunkKind.CODE)]
        scores = reranker.score("def fuse(rankings)", candidates)
        # No penalty applied, so the score is the raw feature sum.
        assert scores[0] > 0

    def test_rare_terms_outweigh_common_ones(self) -> None:
        """IDF is computed over the candidate set, so it adapts per query."""
        reranker = LexicalReranker()
        candidates = [
            candidate("common_only", "ranking ranking ranking discussed here"),
            candidate("rare_only", "the reciprocal mechanism is described"),
            candidate("also_common", "ranking is everywhere in this passage"),
            candidate("common_too", "ranking appears here as well"),
        ]
        scores = dict(
            zip(
                [c.chunk.id for c in candidates],
                reranker.score("reciprocal ranking", candidates),
                strict=True,
            )
        )
        assert scores["rare_only"] > scores["common_only"]

    def test_scores_are_discriminative(self) -> None:
        """RRF scores are compressed by design; rerank scores must separate."""
        reranker = LexicalReranker()
        candidates = [
            candidate("answer", "The RRF damping constant defaults to 60.", score=0.0164),
            candidate("noise", "Entirely unrelated prose about deployment.", score=0.0161),
        ]
        ranked = reranker.rerank("RRF damping constant default", candidates)
        spread = (ranked[0].rerank_score or 0) - (ranked[1].rerank_score or 0)
        assert spread > 0.5

    def test_provenance_is_preserved(self) -> None:
        reranker = LexicalReranker()
        ranked = reranker.rerank("fusion", [candidate("a", "fusion of ranks", score=0.42)])
        item = ranked[0]
        assert item.fusion_score == pytest.approx(0.42)
        assert item.rerank_score is not None
        assert item.score == item.rerank_score
        assert "lexical-rerank" in item.retrievers

    def test_top_n_truncates(self) -> None:
        reranker = LexicalReranker()
        candidates = [candidate(str(i), f"text {i} about fusion") for i in range(10)]
        assert len(reranker.rerank("fusion", candidates, top_n=3)) == 3

    def test_no_candidate_is_lost_without_top_n(self) -> None:
        reranker = LexicalReranker()
        candidates = [candidate(str(i), f"passage {i}") for i in range(7)]
        assert len(reranker.rerank("anything", candidates)) == 7

    def test_empty_candidates(self) -> None:
        assert LexicalReranker().rerank("q", []) == []

    def test_query_with_no_usable_terms_keeps_fused_order(self) -> None:
        reranker = LexicalReranker()
        candidates = [candidate("a", "x", score=0.9), candidate("b", "y", score=0.1)]
        ranked = reranker.rerank("!!! ???", candidates)
        assert [r.chunk.id for r in ranked] == ["a", "b"]


# --------------------------------------------------------------------------- #
# base-class behaviour
# --------------------------------------------------------------------------- #


class _BrokenReranker(Reranker):
    name = "broken"

    def score(self, query: str, candidates: Sequence[ScoredChunk]) -> list[float]:
        raise RuntimeError("model exploded")


class _MiscountingReranker(Reranker):
    name = "miscounting"

    def score(self, query: str, candidates: Sequence[ScoredChunk]) -> list[float]:
        return [1.0]  # wrong length on purpose


class TestFailureModes:
    def test_a_failing_reranker_degrades_to_the_fused_order(self) -> None:
        """A worse ranking beats a failed query."""
        candidates = [candidate("a", "x", score=0.9), candidate("b", "y", score=0.1)]
        ranked = _BrokenReranker().rerank("q", candidates)
        assert [r.chunk.id for r in ranked] == ["a", "b"]

    def test_a_length_mismatch_degrades_safely(self) -> None:
        candidates = [candidate("a", "x", score=0.9), candidate("b", "y", score=0.1)]
        ranked = _MiscountingReranker().rerank("q", candidates)
        assert [r.chunk.id for r in ranked] == ["a", "b"]

    def test_ties_fall_back_to_the_fused_order(self) -> None:
        """A reranker that cannot separate two passages must not shuffle them."""

        class Flat(Reranker):
            name = "flat"

            def score(self, query: str, candidates: Sequence[ScoredChunk]) -> list[float]:
                return [1.0] * len(candidates)

        candidates = [candidate("low", "x", score=0.1), candidate("high", "y", score=0.9)]
        ranked = Flat().rerank("q", candidates)
        assert [r.chunk.id for r in ranked] == ["high", "low"]

    def test_noop_reranker_is_a_true_passthrough(self) -> None:
        candidates = [candidate("a", "x", score=0.1), candidate("b", "y", score=0.9)]
        ranked = NoOpReranker().rerank("q", candidates)
        assert ranked == candidates


# --------------------------------------------------------------------------- #
# listwise LLM parsing
# --------------------------------------------------------------------------- #


class TestParseOrdering:
    def test_clean_reply(self) -> None:
        assert parse_ordering("3, 1, 2", 3) == [2, 0, 1]

    def test_prose_around_the_answer_is_tolerated(self) -> None:
        assert parse_ordering("Sure! The best order is 2, 1, 3.", 3) == [1, 0, 2]

    def test_duplicates_are_dropped(self) -> None:
        assert parse_ordering("1, 1, 2", 3) == [0, 1, 2]

    def test_out_of_range_indices_are_dropped(self) -> None:
        assert parse_ordering("9, 1", 2) == [0, 1]

    def test_omitted_candidates_are_appended_in_order(self) -> None:
        assert parse_ordering("3", 4) == [2, 0, 1, 3]

    def test_garbage_falls_back_to_the_original_order(self) -> None:
        assert parse_ordering("I cannot help with that.", 3) == [0, 1, 2]

    def test_empty_reply(self) -> None:
        assert parse_ordering("", 2) == [0, 1]

    def test_result_is_always_a_permutation(self) -> None:
        for reply in ("2,2,2", "", "abc", "1 2 3 4 5 6 7"):
            assert sorted(parse_ordering(reply, 4)) == [0, 1, 2, 3]


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #


class TestBuildReranker:
    def test_lexical_is_the_default(self) -> None:
        assert isinstance(build_reranker(Settings()), LexicalReranker)

    def test_disabled_returns_none(self) -> None:
        assert build_reranker(Settings(rerank_enabled=False)) is None

    def test_provider_none_returns_none(self) -> None:
        assert build_reranker(Settings(rerank_provider=RerankProvider.NONE)) is None

    def test_missing_extra_falls_back_to_lexical(self) -> None:
        """A missing optional dependency must not break retrieval."""
        reranker = build_reranker(Settings(rerank_provider=RerankProvider.CROSS_ENCODER))
        assert isinstance(reranker, LexicalReranker)

    def test_missing_api_key_falls_back_to_lexical(self, monkeypatch) -> None:
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        reranker = build_reranker(Settings(rerank_provider=RerankProvider.COHERE))
        assert isinstance(reranker, LexicalReranker)


# --------------------------------------------------------------------------- #
# pipeline integration
# --------------------------------------------------------------------------- #


@pytest.fixture
def rerank_kb(tmp_path: Path) -> KnowledgeBase:
    """A corpus built so that fusion alone gets the ordering wrong."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text(
        "# Guide\n\n"
        "## Ranking chatter\n\n"
        "Ranking, ranking, ranking. Constants come up constantly in ranking "
        "discussions, and the word default appears here as well.\n\n"
        "## Fusion constant\n\n"
        "The RRF damping constant defaults to 60.\n\n"
        "## Deployment\n\n"
        "Notes on topology, monitoring and unrelated infrastructure concerns.\n"
    )
    settings = Settings(
        data_dir=tmp_path / "data",
        embedding_dim=256,
        chunk_size=700,
        min_chunk_size=40,
        rerank_enabled=True,
        rerank_provider=RerankProvider.LEXICAL,
    )
    instance = KnowledgeBase(settings)
    instance.ingest(str(docs))
    return instance


class TestPipelineIntegration:
    def test_reranker_is_attached_when_enabled(self, rerank_kb: KnowledgeBase) -> None:
        assert isinstance(rerank_kb.reranker, LexicalReranker)

    def test_rerank_flag_is_reported(self, rerank_kb: KnowledgeBase) -> None:
        assert rerank_kb.search("damping constant", rerank=True).reranked is True
        assert rerank_kb.search("damping constant", rerank=False).reranked is False

    def test_rerank_stage_is_timed(self, rerank_kb: KnowledgeBase) -> None:
        result = rerank_kb.search("damping constant", rerank=True)
        assert "rerank_ms" in result.timings_ms

    def test_reranking_fixes_the_ordering(self, rerank_kb: KnowledgeBase) -> None:
        query = "what does the RRF damping constant default to"
        reranked = rerank_kb.search(query, top_k=3, rerank=True)
        assert "defaults to 60" in reranked.results[0].chunk.text

    def test_rerank_scores_appear_on_results(self, rerank_kb: KnowledgeBase) -> None:
        result = rerank_kb.search("damping constant", top_k=3, rerank=True)
        assert all(r.rerank_score is not None for r in result.results)
        assert all(r.fusion_score is not None for r in result.results)
        assert "rerank=" in result.results[0].explain()

    def test_rerank_pool_is_bounded_by_top_n(self, rerank_kb: KnowledgeBase) -> None:
        result = rerank_kb.search("ranking", top_k=2, rerank=True, rerank_top_n=1)
        assert len(result.results) <= 2

    def test_no_reranker_means_the_flag_is_inert(self, tmp_path: Path) -> None:
        settings = Settings(data_dir=tmp_path / "off", embedding_dim=128, rerank_enabled=False)
        instance = KnowledgeBase(settings)
        instance.ingest_text("# Doc\n\nSome content about fusion.", title="Doc")
        result = instance.search("fusion", rerank=True)
        assert result.reranked is False
        assert result.results
        instance.close()
