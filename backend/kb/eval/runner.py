"""The evaluation runner.

Runs a golden set through the *real* retrieval pipeline — the same
:class:`~kb.knowledge_base.KnowledgeBase` that serves a live query — and reports
aggregate metrics plus per-query detail.

Two design points do most of the work:

**Queries with no resolvable expectation are excluded, not scored zero.** They
are counted and reported separately. Scoring them zero would make the headline
number depend on how stale the golden set is, which is the most expensive kind of
false alarm: it looks exactly like a retrieval regression.

**Sweeps share one runner.** Comparing ``hybrid`` against ``dense`` means running
the same questions through the same corpus with one setting changed, so the
comparison is attributable. ``compare()`` returns a table with per-configuration
deltas, which is the artefact that actually answers "is this change an
improvement".
"""

from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from kb.eval.dataset import GoldenSet, ResolvedQuery, resolve_golden_set
from kb.eval.metrics import (
    average_precision,
    first_relevant_rank,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from kb.models import RetrievalRequest, SupportVerdict

log = logging.getLogger(__name__)

DEFAULT_CUTOFFS = (1, 3, 5, 10)


class QueryResult(BaseModel):
    """Per-query outcome, including enough context to diagnose a failure."""

    query_id: str
    query: str
    n_relevant: int = 0
    n_retrieved: int = 0
    first_relevant_rank: int | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    retrieved_ids: list[str] = Field(default_factory=list)
    top_hit: str = Field(default="", description="Citation label of the top result")
    latency_ms: float = 0.0
    excluded: bool = False
    exclusion_reason: str = ""
    # End-to-end answer signals, when answers were generated.
    faithfulness: float | None = None
    flagged_claims: int | None = None
    refused: bool | None = None
    answer: str = ""


class MetricSummary(BaseModel):
    """Mean and spread for one metric."""

    mean: float
    median: float
    stdev: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    n: int = 0

    @classmethod
    def of(cls, values: Sequence[float]) -> MetricSummary:
        usable = [v for v in values if v == v]  # drop NaN
        if not usable:
            return cls(mean=0.0, median=0.0, n=0)
        return cls(
            mean=round(statistics.fmean(usable), 4),
            median=round(statistics.median(usable), 4),
            stdev=round(statistics.stdev(usable), 4) if len(usable) > 1 else 0.0,
            minimum=round(min(usable), 4),
            maximum=round(max(usable), 4),
            n=len(usable),
        )


class EvalRun(BaseModel):
    """The result of evaluating one configuration against one golden set."""

    label: str = "default"
    golden_set: str = ""
    collection: str = "default"
    config: dict[str, Any] = Field(default_factory=dict)
    n_queries: int = 0
    n_scored: int = 0
    n_excluded: int = 0
    metrics: dict[str, MetricSummary] = Field(default_factory=dict)
    results: list[QueryResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    def metric(self, name: str) -> float:
        """Mean of ``name``, or 0.0 if it was not computed."""
        summary = self.metrics.get(name)
        return summary.mean if summary else 0.0

    def worst(self, metric: str = "mrr", limit: int = 10) -> list[QueryResult]:
        """The queries this configuration handles worst.

        More useful than any aggregate: it is the list of things to go and fix.
        """
        scored = [r for r in self.results if not r.excluded]
        return sorted(scored, key=lambda r: r.metrics.get(metric, 0.0))[:limit]

    def failures(self, metric: str = "hit_rate@10") -> list[QueryResult]:
        """Queries where retrieval found nothing relevant at all."""
        return [r for r in self.results if not r.excluded and r.metrics.get(metric, 0.0) == 0.0]


class EvalRunner:
    """Evaluates retrieval (and optionally answers) against a golden set."""

    def __init__(self, knowledge_base: Any, *, cutoffs: Sequence[int] = DEFAULT_CUTOFFS) -> None:
        self.kb = knowledge_base
        self.cutoffs = tuple(sorted(set(cutoffs)))

    # ------------------------------------------------------------------ #

    def run(
        self,
        golden: GoldenSet,
        *,
        label: str = "default",
        collection: str | None = None,
        overrides: Mapping[str, Any] | None = None,
        with_answers: bool = False,
    ) -> EvalRun:
        """Evaluate one configuration.

        ``overrides`` are retrieval settings applied to every query, which is how
        a sweep isolates a single variable.
        """
        started = time.perf_counter()
        target = collection or golden.collection
        resolved, warnings = resolve_golden_set(golden, self.kb.store, collection=target)

        # Retrieve deeper than the largest cutoff so MRR and MAP see the whole
        # ranking rather than a truncation artefact.
        max_cutoff = max(self.cutoffs)
        settings_override = dict(overrides or {})
        top_k = int(settings_override.pop("top_k", max(max_cutoff, 10)))

        results: list[QueryResult] = []
        latencies: list[float] = []

        for item in resolved:
            result = self._run_one(
                item,
                collection=target,
                top_k=top_k,
                overrides=settings_override,
                with_answers=with_answers,
            )
            results.append(result)
            if not result.excluded:
                latencies.append(result.latency_ms)

        scored = [r for r in results if not r.excluded]
        return EvalRun(
            label=label,
            golden_set=golden.name,
            collection=target,
            config={"top_k": top_k, **settings_override},
            n_queries=len(results),
            n_scored=len(scored),
            n_excluded=len(results) - len(scored),
            metrics=self._aggregate(scored, with_answers=with_answers),
            results=results,
            warnings=warnings,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            mean_latency_ms=round(statistics.fmean(latencies), 3) if latencies else 0.0,
            p95_latency_ms=_percentile(latencies, 95),
        )

    def compare(
        self,
        golden: GoldenSet,
        configurations: Mapping[str, Mapping[str, Any]],
        *,
        collection: str | None = None,
        with_answers: bool = False,
    ) -> list[EvalRun]:
        """Run several configurations over the same questions and corpus.

        Same golden set, same corpus, one variable changed — which is what makes
        the difference attributable rather than suggestive.
        """
        return [
            self.run(
                golden,
                label=label,
                collection=collection,
                overrides=overrides,
                with_answers=with_answers,
            )
            for label, overrides in configurations.items()
        ]

    # ------------------------------------------------------------------ #

    def _run_one(
        self,
        item: ResolvedQuery,
        *,
        collection: str,
        top_k: int,
        overrides: Mapping[str, Any],
        with_answers: bool,
    ) -> QueryResult:
        golden_query = item.query

        if not item.is_usable:
            return QueryResult(
                query_id=golden_query.id,
                query=golden_query.query,
                excluded=True,
                exclusion_reason=("no expected source resolved to a chunk in this collection"),
            )

        request_kwargs: dict[str, Any] = dict(overrides)
        request_kwargs["collection"] = collection
        request_kwargs["top_k"] = top_k
        if golden_query.source_types:
            request_kwargs["source_types"] = golden_query.source_types

        started = time.perf_counter()
        answer = None
        if with_answers:
            answer = self.kb.ask(golden_query.query, **request_kwargs)
            retrieval = answer.retrieval
            ranked_ids = [s.chunk.id for s in retrieval.results] if retrieval else []
            top_hit = (
                retrieval.results[0].chunk.citation_label()
                if retrieval and retrieval.results
                else ""
            )
        else:
            retrieval = self.kb.search(golden_query.query, **request_kwargs)
            ranked_ids = [s.chunk.id for s in retrieval.results]
            top_hit = retrieval.results[0].chunk.citation_label() if retrieval.results else ""
        latency = (time.perf_counter() - started) * 1000

        result = QueryResult(
            query_id=golden_query.id,
            query=golden_query.query,
            n_relevant=len(item.relevance),
            n_retrieved=len(ranked_ids),
            first_relevant_rank=first_relevant_rank(ranked_ids, item.relevance),
            metrics=self._metrics_for(ranked_ids, item.relevance),
            retrieved_ids=ranked_ids[: max(self.cutoffs)],
            top_hit=top_hit,
            latency_ms=round(latency, 3),
        )

        if answer is not None:
            result.faithfulness = answer.faithfulness
            result.flagged_claims = len(answer.flagged_sentences())
            result.refused = answer.refused
            result.answer = answer.text

        return result

    def _metrics_for(
        self, ranked_ids: Sequence[str], relevance: Mapping[str, int]
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for k in self.cutoffs:
            metrics[f"recall@{k}"] = recall_at_k(ranked_ids, relevance, k)
            metrics[f"precision@{k}"] = precision_at_k(ranked_ids, relevance, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, relevance, k)
            metrics[f"hit_rate@{k}"] = hit_rate_at_k(ranked_ids, relevance, k)
        metrics["mrr"] = reciprocal_rank(ranked_ids, relevance)
        metrics["map"] = average_precision(ranked_ids, relevance)
        return metrics

    def _aggregate(
        self, results: Sequence[QueryResult], *, with_answers: bool
    ) -> dict[str, MetricSummary]:
        if not results:
            return {}
        names = sorted(
            {name for result in results for name in result.metrics},
            key=_metric_sort_key,
        )
        summaries = {
            name: MetricSummary.of([r.metrics.get(name, float("nan")) for r in results])
            for name in names
        }
        if with_answers:
            faithfulness = [r.faithfulness for r in results if r.faithfulness is not None]
            if faithfulness:
                summaries["faithfulness"] = MetricSummary.of(faithfulness)
            refusals = [1.0 if r.refused else 0.0 for r in results if r.refused is not None]
            if refusals:
                summaries["refusal_rate"] = MetricSummary.of(refusals)
            flagged = [float(r.flagged_claims) for r in results if r.flagged_claims is not None]
            if flagged:
                summaries["flagged_claims"] = MetricSummary.of(flagged)
        return summaries


def _metric_sort_key(name: str) -> tuple[int, str, int]:
    """Group metrics by family, then by cutoff, so tables read consistently."""
    order = {"hit_rate": 0, "recall": 1, "precision": 2, "ndcg": 3, "mrr": 4, "map": 5}
    if "@" in name:
        family, cutoff = name.split("@", 1)
        return (order.get(family, 9), family, int(cutoff))
    return (order.get(name, 9), name, 0)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentile / 100.0) * (len(ordered) - 1)))
    return round(ordered[index], 3)


def request_for(query: str, **kwargs: Any) -> RetrievalRequest:
    """Build a retrieval request — exposed for tests and ad-hoc scripts."""
    return RetrievalRequest(query=query, **kwargs)


__all__ = [
    "DEFAULT_CUTOFFS",
    "EvalRun",
    "EvalRunner",
    "MetricSummary",
    "QueryResult",
    "SupportVerdict",
    "request_for",
]
