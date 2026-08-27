"""Retrieval evaluation: golden sets, metrics, sweeps and reports."""

from kb.eval.dataset import (
    GoldenQuery,
    GoldenSet,
    ResolvedQuery,
    golden_set_from_pairs,
    resolve_golden_set,
)
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
from kb.eval.report import (
    json_report,
    markdown_report,
    metric_bar_chart,
    rank_distribution_chart,
    write_report,
)
from kb.eval.runner import EvalRun, EvalRunner, MetricSummary, QueryResult
from kb.eval.synthesize import (
    generate_golden_set,
    generate_golden_set_with_llm,
    mine_golden_set,
)

__all__ = [
    "EvalRun",
    "EvalRunner",
    "GoldenQuery",
    "GoldenSet",
    "MetricSummary",
    "QueryResult",
    "ResolvedQuery",
    "average_precision",
    "dcg_at_k",
    "first_relevant_rank",
    "generate_golden_set",
    "generate_golden_set_with_llm",
    "golden_set_from_pairs",
    "hit_rate_at_k",
    "json_report",
    "markdown_report",
    "metric_bar_chart",
    "mine_golden_set",
    "ndcg_at_k",
    "precision_at_k",
    "rank_distribution_chart",
    "recall_at_k",
    "reciprocal_rank",
    "resolve_golden_set",
    "write_report",
]
