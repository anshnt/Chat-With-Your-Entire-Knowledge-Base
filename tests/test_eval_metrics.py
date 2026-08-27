"""Metric tests.

Written against the definitions, with particular attention to the edge cases
where evaluation harnesses quietly lie: empty relevance sets, more relevant
documents than the cutoff, and graded relevance.
"""

from __future__ import annotations

import math

import pytest

from kb.eval.metrics import (
    average_precision,
    dcg_at_k,
    first_relevant_rank,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED = ["a", "b", "c", "d", "e"]


class TestRecall:
    def test_all_relevant_retrieved(self) -> None:
        assert recall_at_k(RANKED, {"a": 2, "b": 2}, 5) == 1.0

    def test_half_retrieved(self) -> None:
        assert recall_at_k(RANKED, {"a": 2, "z": 2}, 5) == 0.5

    def test_none_retrieved(self) -> None:
        assert recall_at_k(RANKED, {"z": 2}, 5) == 0.0

    def test_cutoff_caps_the_denominator(self) -> None:
        """20 relevant chunks cannot all fit in the top 3.

        Dividing by 20 would cap the score at 0.15 no matter how good the
        retriever is, making the metric a property of the golden set.
        """
        relevance = {f"doc{i}": 2 for i in range(20)}
        relevance.update({"a": 2, "b": 2, "c": 2})
        assert recall_at_k(RANKED, relevance, 3) == 1.0

    def test_no_relevant_documents_is_nan(self) -> None:
        """Excluded, not zero: scoring it zero makes runs incomparable."""
        assert math.isnan(recall_at_k(RANKED, {}, 5))

    def test_grade_zero_does_not_count_as_relevant(self) -> None:
        assert math.isnan(recall_at_k(RANKED, {"a": 0}, 5))


class TestPrecision:
    def test_divides_by_k_not_by_results_returned(self) -> None:
        """Two relevant results out of two returned is not precision 1.0 at k=8.

        It is a failure to fill the context window.
        """
        assert precision_at_k(["a", "b"], {"a": 2, "b": 2}, 8) == pytest.approx(0.25)

    def test_full_precision(self) -> None:
        assert precision_at_k(RANKED, dict.fromkeys(RANKED, 2), 5) == 1.0

    def test_no_relevant_is_nan(self) -> None:
        assert math.isnan(precision_at_k(RANKED, {}, 5))


class TestHitRate:
    def test_any_hit_scores_one(self) -> None:
        assert hit_rate_at_k(RANKED, {"e": 2}, 5) == 1.0

    def test_hit_outside_the_cutoff_scores_zero(self) -> None:
        assert hit_rate_at_k(RANKED, {"e": 2}, 3) == 0.0

    def test_no_relevant_is_nan(self) -> None:
        assert math.isnan(hit_rate_at_k(RANKED, {}, 5))


class TestReciprocalRank:
    @pytest.mark.parametrize(
        ("relevant", "expected"),
        [("a", 1.0), ("b", 0.5), ("c", 1 / 3), ("e", 0.2)],
    )
    def test_inverse_of_the_first_hit(self, relevant: str, expected: float) -> None:
        assert reciprocal_rank(RANKED, {relevant: 2}) == pytest.approx(expected)

    def test_only_the_first_hit_matters(self) -> None:
        assert reciprocal_rank(RANKED, {"a": 2, "b": 2, "c": 2}) == 1.0

    def test_no_hit_scores_zero(self) -> None:
        assert reciprocal_rank(RANKED, {"z": 2}) == 0.0

    def test_no_relevant_is_nan(self) -> None:
        assert math.isnan(reciprocal_rank(RANKED, {}))


class TestAveragePrecision:
    def test_all_relevant_first_scores_one(self) -> None:
        assert average_precision(RANKED, {"a": 2, "b": 2}) == 1.0

    def test_rewards_finding_all_of_them_early(self) -> None:
        """Unlike MRR, MAP distinguishes one good hit from several."""
        one_hit = average_precision(RANKED, {"a": 2, "e": 2})
        both_early = average_precision(RANKED, {"a": 2, "b": 2})
        assert both_early > one_hit

    def test_no_hit_scores_zero(self) -> None:
        assert average_precision(RANKED, {"z": 2}) == 0.0


class TestDCGAndNDCG:
    def test_dcg_uses_exponential_gain(self) -> None:
        # grade 2 at rank 1: (2^2 - 1) / log2(2) = 3
        assert dcg_at_k(["a"], {"a": 2}, 1) == pytest.approx(3.0)

    def test_dcg_discounts_by_rank(self) -> None:
        first = dcg_at_k(["a", "b"], {"a": 2}, 2)
        second = dcg_at_k(["b", "a"], {"a": 2}, 2)
        assert first > second

    def test_perfect_ranking_scores_one(self) -> None:
        assert ndcg_at_k(RANKED, {"a": 2, "b": 1}, 5) == pytest.approx(1.0)

    def test_reversed_ranking_scores_less(self) -> None:
        assert ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 2) < 1.0

    def test_graded_relevance_is_respected(self) -> None:
        """Putting the "directly answers" chunk first must beat putting the
        merely-related one first."""
        good = ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 2)
        worse = ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 2)
        assert good > worse

    def test_ceiling_is_achievable_at_k(self) -> None:
        """With 5 relevant chunks and k=2, retrieving the best 2 must score 1.0.

        Normalising against an unbounded perfect ranking would move the ceiling
        with the golden set.
        """
        relevance = {"a": 2, "b": 2, "c": 2, "d": 2, "e": 2}
        assert ndcg_at_k(["a", "b"], relevance, 2) == pytest.approx(1.0)

    def test_no_relevant_is_nan(self) -> None:
        assert math.isnan(ndcg_at_k(RANKED, {}, 5))

    def test_all_zero_grades_is_nan(self) -> None:
        assert math.isnan(ndcg_at_k(RANKED, {"a": 0, "b": 0}, 5))


class TestFirstRelevantRank:
    def test_reports_the_position(self) -> None:
        assert first_relevant_rank(RANKED, {"c": 2}) == 3

    def test_none_when_not_retrieved(self) -> None:
        assert first_relevant_rank(RANKED, {"z": 2}) is None

    def test_empty_ranking(self) -> None:
        assert first_relevant_rank([], {"a": 2}) is None
