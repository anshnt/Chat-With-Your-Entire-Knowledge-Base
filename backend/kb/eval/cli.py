"""``kb eval`` — the evaluation sub-commands.

Kept in the eval package rather than in ``kb/cli.py`` so the evaluation layer
owns its own surface, and so the main CLI stays readable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from kb.errors import KBError
from kb.eval.dataset import GoldenSet
from kb.eval.report import HEADLINE, markdown_report, write_report
from kb.eval.runner import EvalRunner
from kb.eval.synthesize import generate_golden_set, mine_golden_set
from kb.models import RetrievalStrategy

app = typer.Typer(
    name="eval",
    help="Measure retrieval quality against a golden set.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)

#: Named sweeps, so the interesting comparison is one flag rather than a script.
PRESET_SWEEPS: dict[str, dict[str, dict[str, Any]]] = {
    "strategies": {
        "lexical": {"strategy": RetrievalStrategy.LEXICAL, "rerank": False},
        "dense": {"strategy": RetrievalStrategy.DENSE, "rerank": False},
        "hybrid": {"strategy": RetrievalStrategy.HYBRID, "rerank": False},
    },
    "fusion": {
        "rrf": {"fusion": "rrf", "rerank": False},
        "weighted": {"fusion": "weighted", "rerank": False},
        "max": {"fusion": "max", "rerank": False},
    },
    "rerank": {
        "fusion only": {"rerank": False},
        "reranked": {"rerank": True},
    },
    "mmr": {
        "no mmr": {"use_mmr": False},
        "mmr 0.7": {"use_mmr": True, "mmr_lambda": 0.7},
        "mmr 0.5": {"use_mmr": True, "mmr_lambda": 0.5},
    },
    "full": {
        "lexical": {"strategy": RetrievalStrategy.LEXICAL, "rerank": False},
        "dense": {"strategy": RetrievalStrategy.DENSE, "rerank": False},
        "hybrid": {"strategy": RetrievalStrategy.HYBRID, "rerank": False},
        "hybrid+rerank": {"strategy": RetrievalStrategy.HYBRID, "rerank": True},
    },
}


def _open(state: dict[str, Any]):
    from kb.knowledge_base import KnowledgeBase

    return KnowledgeBase(state["settings"]())


def _fail(exc: Exception) -> None:
    message = exc.message if isinstance(exc, KBError) else str(exc)
    err_console.print(f"[bold red]error:[/] {message}")
    raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #


@app.command("generate")
def generate(
    ctx: typer.Context,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to write the golden set")
    ] = Path("eval/golden.yaml"),
    collection: Annotated[str, typer.Option("--collection", "-c")] = "default",
    limit: Annotated[int, typer.Option("--limit", "-n", help="Maximum questions")] = 100,
    per_document: Annotated[
        int, typer.Option("--per-document", help="Cap questions per document")
    ] = 2,
    use_llm: Annotated[
        bool, typer.Option("--llm", help="Generate with a language model (needs a key)")
    ] = False,
) -> None:
    """Generate a golden set from the corpus.

    Labels are correct by construction: each question is derived from a specific
    chunk, so that chunk answers it. Expectations are recorded as text snippets
    rather than chunk ids, so the set survives re-chunking.

    Synthetic questions are phrased in the corpus's own vocabulary and therefore
    over-reward lexical retrieval. Use them to compare configurations and catch
    regressions — not as an absolute quality measure.
    """
    knowledge_base = _open(ctx.obj)
    chunks = [c for batch in knowledge_base.store.iter_chunks(collection) for c in batch]
    if not chunks:
        _fail(KBError(f"collection {collection!r} is empty — ingest something first"))
        return

    if use_llm:
        from kb.eval.synthesize import generate_golden_set_with_llm

        try:
            from kb.llm import AnthropicClient

            client = AnthropicClient(api_key=knowledge_base.settings.anthropic_api_key)
        except Exception as exc:
            _fail(exc)
            return
        golden = generate_golden_set_with_llm(chunks, client, collection=collection, limit=limit)
    else:
        golden = generate_golden_set(
            chunks, collection=collection, per_document=per_document, limit=limit
        )

    if not golden.queries:
        _fail(KBError("no questions could be generated from this corpus"))
        return

    golden.save(output)
    console.print(
        f"wrote [bold green]{len(golden)}[/] questions to [bold]{output}[/]\n"
        f"[dim]review them before trusting the numbers — delete any that are not "
        f"answerable from their expected source[/]"
    )
    for query in golden.queries[:8]:
        console.print(f"  [cyan]?[/] {query.query}")
    if len(golden.queries) > 8:
        console.print(f"  [dim]… and {len(golden.queries) - 8} more[/]")


@app.command("mine")
def mine(
    ctx: typer.Context,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("eval/mined.yaml"),
    collection: Annotated[str, typer.Option("--collection", "-c")] = "default",
    limit: Annotated[int, typer.Option("--limit", "-n")] = 100,
) -> None:
    """Seed a golden set from real logged queries, for you to label.

    Real traffic is the only ground truth for what people actually ask, so the
    labelling is left to you: each query arrives with a ``must_contain``
    placeholder to fill in.
    """
    knowledge_base = _open(ctx.obj)
    queries = knowledge_base.store.recent_queries(collection, limit=limit)
    if not queries:
        _fail(KBError("no queries logged yet — run some searches first"))
        return
    golden = mine_golden_set(queries, collection=collection)
    golden.save(output)
    console.print(
        f"wrote [bold green]{len(golden)}[/] logged queries to [bold]{output}[/]\n"
        f"[dim]fill in must_contain for each, then delete the rest[/]"
    )


@app.command("run")
def run(
    ctx: typer.Context,
    golden_path: Annotated[Path, typer.Argument(help="Golden set (YAML, JSON or JSONL)")],
    collection: Annotated[str | None, typer.Option("--collection", "-c")] = None,
    sweep: Annotated[
        str | None,
        typer.Option("--sweep", help=f"Preset comparison: {', '.join(PRESET_SWEEPS)}"),
    ] = None,
    strategy: Annotated[RetrievalStrategy | None, typer.Option("--strategy", "-s")] = None,
    rerank: Annotated[bool | None, typer.Option("--rerank/--no-rerank")] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 10,
    with_answers: Annotated[
        bool,
        typer.Option("--answers", help="Also generate and verify answers (slower)"),
    ] = False,
    report_dir: Annotated[
        Path | None,
        typer.Option("--report", help="Write markdown, JSON and SVG charts here"),
    ] = None,
    tag: Annotated[str | None, typer.Option("--tag", help="Only queries carrying this tag")] = None,
    fail_under: Annotated[
        float | None,
        typer.Option("--fail-under", help="Exit non-zero if the headline metric is below this"),
    ] = None,
    metric: Annotated[
        str, typer.Option("--metric", help="Metric used by --fail-under and for ordering")
    ] = "ndcg@5",
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
) -> None:
    """Evaluate retrieval against a golden set.

    With ``--sweep`` the same questions run through several configurations over
    the same corpus, so the difference between them is attributable rather than
    suggestive.
    """
    try:
        golden = GoldenSet.load(golden_path)
    except KBError as exc:
        _fail(exc)
        return
    if tag:
        golden = golden.tagged(tag)
        if not golden.queries:
            _fail(KBError(f"no queries tagged {tag!r}"))
            return

    knowledge_base = _open(ctx.obj)
    runner = EvalRunner(knowledge_base)

    if sweep:
        if sweep not in PRESET_SWEEPS:
            _fail(KBError(f"unknown sweep {sweep!r}; choose from {', '.join(PRESET_SWEEPS)}"))
            return
        configurations = {
            label: {**overrides, "top_k": top_k}
            for label, overrides in PRESET_SWEEPS[sweep].items()
        }
        runs = runner.compare(
            golden, configurations, collection=collection, with_answers=with_answers
        )
    else:
        overrides: dict[str, Any] = {"top_k": top_k}
        if strategy is not None:
            overrides["strategy"] = strategy
        if rerank is not None:
            overrides["rerank"] = rerank
        runs = [
            runner.run(
                golden,
                label=strategy.value if strategy else "default",
                collection=collection,
                overrides=overrides,
                with_answers=with_answers,
            )
        ]

    if as_json:
        from kb.eval.report import json_report

        console.print_json(json_report(runs))
    else:
        _print_runs(runs, with_answers=with_answers)

    if report_dir:
        written = write_report(runs, report_dir, include_queries=True)
        console.print()
        for kind, path in written.items():
            console.print(f"[dim]{kind}:[/] {path}")

    if fail_under is not None:
        best = max(r.metric(metric) for r in runs)
        if best < fail_under:
            err_console.print(
                f"[bold red]{metric} {best:.3f} is below the {fail_under:.3f} threshold[/]"
            )
            raise typer.Exit(code=1)
        console.print(f"[green]{metric} {best:.3f} meets the {fail_under:.3f} threshold[/]")


@app.command("report")
def report(
    golden_path: Annotated[Path, typer.Argument(help="A previously written JSON report")],
) -> None:
    """Re-render a stored JSON report as Markdown."""
    payload = json.loads(Path(golden_path).read_text(encoding="utf-8"))
    from kb.eval.runner import EvalRun

    runs = [EvalRun.model_validate(item) for item in payload.get("runs", [])]
    console.print(markdown_report(runs))


# --------------------------------------------------------------------------- #


def _print_runs(runs: list[Any], *, with_answers: bool) -> None:
    first = runs[0]
    console.print(
        f"golden set [bold]{first.golden_set}[/] · collection [bold]{first.collection}[/] · "
        f"[bold]{first.n_scored}[/] of {first.n_queries} queries scored"
    )
    if first.n_excluded:
        console.print(
            f"[yellow]{first.n_excluded} excluded[/] [dim](no expected source resolved to a "
            f"chunk — excluded rather than scored zero, so a stale golden set does not "
            f"read as a regression)[/]"
        )
    console.print()

    metrics = [m for m in HEADLINE if any(m in r.metrics for r in runs)]
    if with_answers:
        metrics += [
            m for m in ("faithfulness", "refusal_rate") if any(m in r.metrics for r in runs)
        ]

    table = Table(box=None)
    table.add_column("configuration", style="bold cyan")
    for name in metrics:
        table.add_column(name, justify="right")
    table.add_column("mean ms", justify="right", style="dim")

    baseline = runs[0]
    for index, item in enumerate(runs):
        cells = []
        for name in metrics:
            value = item.metric(name)
            if index == 0 or len(runs) == 1:
                cells.append(f"{value:.3f}")
            else:
                delta = value - baseline.metric(name)
                colour = "green" if delta > 0.0005 else "red" if delta < -0.0005 else "dim"
                sign = "+" if delta > 0 else ""
                cells.append(f"{value:.3f} [{colour}]{sign}{delta:.3f}[/]")
        table.add_row(item.label, *cells, f"{item.mean_latency_ms:.1f}")
    console.print(table)

    failures = baseline.failures()
    if failures:
        console.print(
            f"\n[bold yellow]{len(failures)} query(ies) retrieved nothing relevant[/] "
            f"[dim](generation quality is irrelevant for these)[/]"
        )
        for result in failures[:8]:
            console.print(f"  [yellow]![/] {result.query}")
            console.print(f"    [dim]top result: {result.top_hit or '—'}[/]")

    weak = [r for r in baseline.worst("mrr", limit=5) if r not in failures]
    if weak:
        console.print("\n[bold]weakest rankings[/]")
        for result in weak:
            rank = result.first_relevant_rank or "—"
            console.print(f"  [dim]rank {rank}[/] {result.query}  [dim]→ {result.top_hit}[/]")
