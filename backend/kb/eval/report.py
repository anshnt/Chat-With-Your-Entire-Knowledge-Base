"""Evaluation reports.

Three outputs, each for a different reader:

* **Markdown** — commits into the repo, so a retrieval change arrives in a PR
  with its numbers attached and the diff shows what moved.
* **JSON** — machine-readable, for CI thresholds and trend tracking.
* **SVG charts** — committed alongside the Markdown so a README can show the
  result rather than assert it. Hand-authored (no plotting dependency) with
  colours that read on both light and dark backgrounds, since that is where they
  get looked at.

The comparison table is the point of the whole module: same questions, same
corpus, one variable changed, deltas against the baseline. That is what turns
"hybrid beats dense" from a claim into a measurement.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from kb.eval.runner import EvalRun

#: Metrics shown in the headline table, in this order.
HEADLINE = ("hit_rate@5", "recall@5", "ndcg@5", "mrr", "map")

# Chart palette. Mid-tone hues chosen to hold contrast against both a white and
# a dark page, because a committed SVG has no control over its background.
_SERIES_COLOURS = ("#3b82f6", "#f59e0b", "#10b981", "#a855f7", "#ef4444", "#14b8a6")
_AXIS = "#8b93a1"
_TEXT = "#6b7280"


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #


def markdown_report(runs: Sequence[EvalRun], *, title: str = "Retrieval evaluation") -> str:
    """Render one or more runs as Markdown."""
    if not runs:
        return f"# {title}\n\nNo runs.\n"

    lines: list[str] = [f"# {title}", ""]
    first = runs[0]
    lines += [
        f"Golden set **{first.golden_set}** · collection **{first.collection}** · "
        f"{first.n_scored} of {first.n_queries} queries scored",
        "",
    ]
    if first.n_excluded:
        lines += [
            f"> {first.n_excluded} query(ies) excluded: no expected source resolved to a "
            "chunk in this collection. Excluded rather than scored zero, so a stale "
            "golden set does not read as a retrieval regression.",
            "",
        ]

    lines += _headline_table(runs)

    if len(runs) > 1:
        lines += _delta_table(runs)

    lines += _latency_table(runs)

    for run in runs:
        lines += _run_detail(run, include_heading=len(runs) > 1)

    if first.warnings:
        lines += ["## Warnings", ""]
        lines += [f"- {w}" for w in first.warnings[:40]]
        if len(first.warnings) > 40:
            lines.append(f"- … and {len(first.warnings) - 40} more")
        lines.append("")

    return "\n".join(lines)


def _headline_table(runs: Sequence[EvalRun]) -> list[str]:
    metrics = [m for m in HEADLINE if any(m in r.metrics for r in runs)]
    extra = [m for m in ("faithfulness", "refusal_rate") if any(m in run.metrics for run in runs)]
    metrics += extra
    header = "| configuration | " + " | ".join(metrics) + " |"
    divider = "|---" * (len(metrics) + 1) + "|"
    rows = [
        "| `" + run.label + "` | " + " | ".join(f"{run.metric(m):.3f}" for m in metrics) + " |"
        for run in runs
    ]
    return ["## Results", "", header, divider, *rows, ""]


def _delta_table(runs: Sequence[EvalRun]) -> list[str]:
    """Deltas against the first run, which is treated as the baseline."""
    baseline = runs[0]
    metrics = [m for m in HEADLINE if m in baseline.metrics]
    header = "| configuration | " + " | ".join(f"Δ {m}" for m in metrics) + " |"
    divider = "|---" * (len(metrics) + 1) + "|"
    rows: list[str] = []
    for run in runs[1:]:
        cells = []
        for metric in metrics:
            delta = run.metric(metric) - baseline.metric(metric)
            sign = "+" if delta > 0 else ""
            cells.append(f"{sign}{delta:.3f}")
        rows.append("| `" + run.label + "` | " + " | ".join(cells) + " |")
    return [
        f"### Change vs `{baseline.label}`",
        "",
        header,
        divider,
        *rows,
        "",
    ]


def _latency_table(runs: Sequence[EvalRun]) -> list[str]:
    lines = [
        "### Latency",
        "",
        "| configuration | mean | p95 |",
        "|---|---|---|",
    ]
    for run in runs:
        lines.append(
            f"| `{run.label}` | {run.mean_latency_ms:.1f} ms | {run.p95_latency_ms:.1f} ms |"
        )
    lines.append("")
    return lines


def _run_detail(run: EvalRun, *, include_heading: bool) -> list[str]:
    lines: list[str] = []
    if include_heading:
        lines += [f"## `{run.label}`", ""]
        if run.config:
            settings = ", ".join(f"{k}={v}" for k, v in sorted(run.config.items()))
            lines += [f"Config: `{settings}`", ""]

    failures = run.failures()
    if failures:
        lines += [
            f"### Retrieval found nothing relevant ({len(failures)} queries)",
            "",
            "These are the cases where generation quality is irrelevant — the answer "
            "was not in the context.",
            "",
            "| query | top result |",
            "|---|---|",
        ]
        for result in failures[:15]:
            lines.append(f"| {_escape(result.query)} | {_escape(result.top_hit) or '—'} |")
        if len(failures) > 15:
            lines.append(f"| … and {len(failures) - 15} more | |")
        lines.append("")

    worst = [r for r in run.worst("mrr", limit=10) if r not in failures]
    if worst:
        lines += [
            "### Weakest rankings",
            "",
            "| query | first relevant rank | MRR | top result |",
            "|---|---|---|---|",
        ]
        for result in worst:
            rank = result.first_relevant_rank or "—"
            lines.append(
                f"| {_escape(result.query)} | {rank} | {result.metrics.get('mrr', 0):.3f} | "
                f"{_escape(result.top_hit)} |"
            )
        lines.append("")
    return lines


def _escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


# --------------------------------------------------------------------------- #
# json
# --------------------------------------------------------------------------- #


def json_report(runs: Sequence[EvalRun], *, include_queries: bool = False) -> str:
    """Machine-readable report, for CI thresholds and trend tracking."""
    payload = {
        "runs": [
            {
                "label": run.label,
                "golden_set": run.golden_set,
                "collection": run.collection,
                "config": run.config,
                "n_queries": run.n_queries,
                "n_scored": run.n_scored,
                "n_excluded": run.n_excluded,
                "mean_latency_ms": run.mean_latency_ms,
                "p95_latency_ms": run.p95_latency_ms,
                "metrics": {name: summary.model_dump() for name, summary in run.metrics.items()},
                **({"queries": [r.model_dump() for r in run.results]} if include_queries else {}),
            }
            for run in runs
        ]
    }
    return json.dumps(payload, indent=2)


# --------------------------------------------------------------------------- #
# svg charts
# --------------------------------------------------------------------------- #


def metric_bar_chart(
    runs: Sequence[EvalRun],
    *,
    metrics: Sequence[str] = HEADLINE,
    width: int = 720,
    height: int = 320,
    title: str = "Retrieval quality by configuration",
) -> str:
    """Grouped bar chart of metrics across configurations, as standalone SVG.

    Hand-authored rather than plotted so the repo needs no charting dependency
    and the output is deterministic — a chart that changes pixels on every run
    makes for a useless diff.
    """
    metrics = [m for m in metrics if any(m in r.metrics for r in runs)]
    if not runs or not metrics:
        return _empty_chart(width, height, "no data")

    pad_left, pad_right, pad_top, pad_bottom = 48, 16, 46, 52
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    group_w = plot_w / len(metrics)
    bar_w = min(28.0, (group_w * 0.72) / len(runs))
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="{_xml(title)}" font-family="system-ui, -apple-system, sans-serif">',
        f"<title>{_xml(title)}</title>",
        f'<text x="{pad_left}" y="22" font-size="14" font-weight="600" fill="{_TEXT}">'
        f"{_xml(title)}</text>",
    ]

    # Gridlines and y-axis labels at 0, 0.25, 0.5, 0.75, 1.0.
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = pad_top + plot_h * (1 - fraction)
        parts.append(
            f'<line x1="{pad_left}" y1="{y:.1f}" x2="{width - pad_right}" y2="{y:.1f}" '
            f'stroke="{_AXIS}" stroke-opacity="0.25" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_left - 8}" y="{y + 4:.1f}" font-size="10" text-anchor="end" '
            f'fill="{_TEXT}" fill-opacity="0.8">{fraction:g}</text>'
        )

    for m_index, metric in enumerate(metrics):
        group_x = pad_left + group_w * m_index
        for r_index, run in enumerate(runs):
            value = max(0.0, min(1.0, run.metric(metric)))
            bar_h = plot_h * value
            x = group_x + (group_w - bar_w * len(runs)) / 2 + bar_w * r_index
            y = pad_top + plot_h - bar_h
            colour = _SERIES_COLOURS[r_index % len(_SERIES_COLOURS)]
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
                f'fill="{colour}" rx="2"><title>{_xml(run.label)} · {_xml(metric)}: '
                f"{value:.3f}</title></rect>"
            )
            if bar_w >= 18:
                parts.append(
                    f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" font-size="9" '
                    f'text-anchor="middle" fill="{_TEXT}">{value:.2f}</text>'
                )
        parts.append(
            f'<text x="{group_x + group_w / 2:.1f}" y="{pad_top + plot_h + 16}" '
            f'font-size="10" text-anchor="middle" fill="{_TEXT}">{_xml(metric)}</text>'
        )

    parts.append(_legend(runs, pad_left, height - 14))
    parts.append("</svg>")
    return "\n".join(parts)


def rank_distribution_chart(
    run: EvalRun,
    *,
    width: int = 720,
    height: int = 260,
    max_rank: int = 10,
    title: str = "Rank of the first relevant chunk",
) -> str:
    """Histogram of where the first relevant chunk landed.

    More diagnostic than any aggregate: a long tail at rank 8-10 says the
    reranker is the problem, while a spike at "not found" says retrieval is.
    """
    scored = [r for r in run.results if not r.excluded]
    if not scored:
        return _empty_chart(width, height, "no scored queries")

    buckets = [0] * (max_rank + 1)  # index 0 .. max_rank-1 are ranks, last is "not found"
    for result in scored:
        rank = result.first_relevant_rank
        if rank is None or rank > max_rank:
            buckets[max_rank] += 1
        else:
            buckets[rank - 1] += 1

    pad_left, pad_right, pad_top, pad_bottom = 40, 16, 42, 42
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    peak = max(buckets) or 1
    bar_w = plot_w / len(buckets)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{_xml(title)}" '
        f'font-family="system-ui, -apple-system, sans-serif">',
        f"<title>{_xml(title)}</title>",
        f'<text x="{pad_left}" y="22" font-size="14" font-weight="600" fill="{_TEXT}">'
        f"{_xml(title)}</text>",
        f'<line x1="{pad_left}" y1="{pad_top + plot_h}" x2="{width - pad_right}" '
        f'y2="{pad_top + plot_h}" stroke="{_AXIS}" stroke-opacity="0.4"/>',
    ]

    for index, count in enumerate(buckets):
        bar_h = plot_h * (count / peak)
        x = pad_left + bar_w * index + bar_w * 0.15
        y = pad_top + plot_h - bar_h
        # The "not found" bucket is the failure case, so it is coloured as one.
        colour = _SERIES_COLOURS[4] if index == max_rank else _SERIES_COLOURS[0]
        label = f">{max_rank}" if index == max_rank else str(index + 1)
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.7:.1f}" height="{bar_h:.1f}" '
            f'fill="{colour}" rx="2"><title>rank {label}: {count} queries</title></rect>'
        )
        if count:
            parts.append(
                f'<text x="{x + bar_w * 0.35:.1f}" y="{y - 4:.1f}" font-size="9" '
                f'text-anchor="middle" fill="{_TEXT}">{count}</text>'
            )
        parts.append(
            f'<text x="{x + bar_w * 0.35:.1f}" y="{pad_top + plot_h + 15}" font-size="10" '
            f'text-anchor="middle" fill="{_TEXT}">{label}</text>'
        )

    parts.append(
        f'<text x="{width / 2}" y="{height - 8}" font-size="10" text-anchor="middle" '
        f'fill="{_TEXT}" fill-opacity="0.8">rank of first relevant chunk '
        f"(rightmost bar = not retrieved)</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def _legend(runs: Sequence[EvalRun], x: float, y: float) -> str:
    parts: list[str] = []
    cursor = x
    for index, run in enumerate(runs):
        colour = _SERIES_COLOURS[index % len(_SERIES_COLOURS)]
        parts.append(
            f'<rect x="{cursor:.1f}" y="{y - 9:.1f}" width="10" height="10" fill="{colour}" rx="2"/>'
        )
        parts.append(
            f'<text x="{cursor + 15:.1f}" y="{y:.1f}" font-size="10" fill="{_TEXT}">'
            f"{_xml(run.label)}</text>"
        )
        cursor += 26 + 6.2 * len(run.label)
    return "".join(parts)


def _empty_chart(width: int, height: int, message: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{_xml(message)}">'
        f'<text x="{width / 2}" y="{height / 2}" font-size="12" text-anchor="middle" '
        f'fill="{_TEXT}" font-family="system-ui, sans-serif">{_xml(message)}</text></svg>'
    )


def _xml(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #


def write_report(
    runs: Sequence[EvalRun],
    directory: str | Path,
    *,
    stem: str = "evaluation",
    include_queries: bool = False,
    charts: bool = True,
) -> dict[str, Path]:
    """Write Markdown, JSON and (optionally) SVG charts. Returns the paths."""
    out_dir = Path(directory).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    markdown_path = out_dir / f"{stem}.md"
    markdown_path.write_text(markdown_report(runs), encoding="utf-8")
    written["markdown"] = markdown_path

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json_report(runs, include_queries=include_queries), encoding="utf-8")
    written["json"] = json_path

    if charts and runs:
        # Name the golden set in the chart title: a committed chart is read far
        # from the command that produced it, and "which set is this?" is the
        # first question anyone asks of it.
        suffix = f" — {runs[0].golden_set}" if runs[0].golden_set else ""

        chart_path = out_dir / f"{stem}-metrics.svg"
        chart_path.write_text(
            metric_bar_chart(runs, title=f"Retrieval quality{suffix}"), encoding="utf-8"
        )
        written["metrics_chart"] = chart_path

        ranks_path = out_dir / f"{stem}-ranks.svg"
        ranks_path.write_text(
            rank_distribution_chart(runs[0], title=f"Rank of the first relevant chunk{suffix}"),
            encoding="utf-8",
        )
        written["rank_chart"] = ranks_path

    return written
