"""Command-line interface.

``kb`` is the primary way to drive the system, and it deliberately exposes the
retrieval knobs (strategy, fusion, weights, MMR) as flags. Being able to run

    kb search "..." --strategy lexical
    kb search "..." --strategy dense
    kb search "..." --fusion weighted

back to back, and see the score breakdown change, is how you develop an
intuition for what hybrid retrieval is actually doing.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from kb import __version__
from kb.config import EmbeddingProvider, Settings, get_settings
from kb.errors import KBError
from kb.knowledge_base import KnowledgeBase
from kb.models import FusionMethod, RetrievalStrategy, SourceType

app = typer.Typer(
    name="kb",
    help="Chat with your entire knowledge base: hybrid retrieval with verified citations.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# shared option plumbing
# --------------------------------------------------------------------------- #

_overrides: dict[str, object] = {}
_state: dict[str, object] = {}


@app.callback()
def main_callback(
    db: Annotated[
        Path | None, typer.Option("--db", help="Path to the SQLite database file")
    ] = None,
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Directory for database, uploads and cache")
    ] = None,
    embedding_provider: Annotated[
        EmbeddingProvider | None,
        typer.Option("--embedder", help="Embedding provider"),
    ] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress progress output")] = False,
) -> None:
    """Global options, applied before any subcommand runs."""
    overrides: dict[str, object] = {}
    if data_dir is not None:
        overrides["data_dir"] = data_dir
    if db is not None:
        overrides["data_dir"] = db.parent if db.suffix else db
        if db.suffix:
            overrides["db_filename"] = db.name
    if embedding_provider is not None:
        overrides["embedding_provider"] = embedding_provider
        overrides["embedding_model"] = ""
    _overrides.clear()
    _overrides.update(overrides)
    _state["quiet"] = quiet


def _settings() -> Settings:
    base = get_settings()
    return base.model_copy(update=dict(_overrides)) if _overrides else base


def _open() -> KnowledgeBase:
    return KnowledgeBase(_settings())


def _fail(exc: KBError | Exception) -> None:
    message = exc.message if isinstance(exc, KBError) else str(exc)
    err_console.print(f"[bold red]error:[/] {message}")
    raise typer.Exit(code=1)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"kb {__version__}")


@app.command()
def ingest(
    sources: Annotated[
        list[str], typer.Argument(help="Files, directories, globs or URLs to ingest")
    ],
    collection: Annotated[str, typer.Option("--collection", "-c")] = "default",
    title: Annotated[
        str | None, typer.Option("--title", help="Override the document title")
    ] = None,
    no_embed: Annotated[
        bool, typer.Option("--no-embed", help="Skip embedding (BM25 search still works)")
    ] = False,
) -> None:
    """Ingest sources into a collection.

    Directories are walked and globs expanded, skipping the usual noise
    (``node_modules``, ``.git``, build output). Re-ingesting unchanged content is
    a no-op, so this is safe to run on a schedule.
    """
    knowledge_base = _open()
    options: dict[str, object] = {}
    if title:
        options["title"] = title
    try:
        report = knowledge_base.ingest_many(
            list(sources), collection=collection, embed=not no_embed, **options
        )
    except KBError as exc:
        _fail(exc)
        return

    table = Table(box=None, pad_edge=False)
    table.add_column("", style="dim")
    table.add_column("")
    table.add_row("documents", f"[bold green]{report.documents_created}[/]")
    table.add_row("chunks", f"[bold]{report.chunks_created}[/]")
    if report.duplicates_skipped:
        table.add_row("unchanged", f"[yellow]{report.duplicates_skipped}[/]")
    if report.documents_skipped:
        table.add_row("skipped", f"[yellow]{report.documents_skipped}[/]")
    table.add_row("elapsed", f"{report.elapsed_ms:.0f} ms")
    console.print(table)

    if report.errors:
        console.print(f"\n[bold yellow]{len(report.errors)} source(s) failed:[/]")
        for error in report.errors[:20]:
            console.print(f"  [dim]{error.get('source', '?')}[/] — {error.get('error', '')}")
        if len(report.errors) > 20:
            console.print(f"  [dim]… and {len(report.errors) - 20} more[/]")

    if report.documents:
        console.print()
        for document in report.documents[:15]:
            console.print(
                f"  [green]+[/] {document.title} "
                f"[dim]({document.source_type.value}, {document.n_chunks} chunks)[/]"
            )
        if len(report.documents) > 15:
            console.print(f"  [dim]… and {len(report.documents) - 15} more[/]")


@app.command(name="add-text")
def add_text(
    title: Annotated[str, typer.Argument(help="Document title")],
    text: Annotated[
        str | None, typer.Option("--text", help="Body text; omit to read from stdin")
    ] = None,
    collection: Annotated[str, typer.Option("--collection", "-c")] = "default",
) -> None:
    """Ingest text from an argument or a pipe: ``cat notes.md | kb add-text Notes``."""
    body = text if text is not None else sys.stdin.read()
    if not body.strip():
        _fail(KBError("no text provided"))
        return
    knowledge_base = _open()
    report = knowledge_base.ingest_text(body, title=title, collection=collection)
    console.print(
        f"[green]+[/] {title} [dim]({report.chunks_created} chunks, {report.elapsed_ms:.0f} ms)[/]"
    )


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What to search for")],
    collection: Annotated[str, typer.Option("--collection", "-c")] = "default",
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Results to return")] = 8,
    candidate_k: Annotated[
        int | None, typer.Option("--candidate-k", help="Candidates per retriever before fusion")
    ] = None,
    strategy: Annotated[RetrievalStrategy | None, typer.Option("--strategy", "-s")] = None,
    fusion: Annotated[FusionMethod | None, typer.Option("--fusion", "-f")] = None,
    mmr: Annotated[bool, typer.Option("--mmr/--no-mmr", help="Diversify results")] = False,
    rerank: Annotated[
        bool | None, typer.Option("--rerank/--no-rerank", help="Second-stage reranking")
    ] = None,
    source_type: Annotated[
        list[SourceType] | None, typer.Option("--source", help="Restrict to source types")
    ] = None,
    show_text: Annotated[
        bool, typer.Option("--full", help="Print full chunk text instead of a snippet")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
) -> None:
    """Search the knowledge base and print the ranking with score provenance."""
    knowledge_base = _open()
    try:
        result = knowledge_base.search(
            query,
            collection=collection,
            top_k=top_k,
            candidate_k=candidate_k,
            strategy=strategy,
            fusion=fusion,
            use_mmr=mmr or None,
            rerank=rerank,
            source_types=list(source_type) if source_type else None,
        )
    except KBError as exc:
        _fail(exc)
        return
    except ValueError as exc:
        _fail(exc)
        return

    if as_json:
        console.print_json(
            json.dumps(
                {
                    "query": result.query,
                    "strategy": result.strategy.value,
                    "fusion": result.fusion.value if result.fusion else None,
                    "reranked": result.reranked,
                    "timings_ms": result.timings_ms,
                    "results": [
                        {
                            "chunk_id": r.chunk.id,
                            "score": r.score,
                            "lexical_score": r.lexical_score,
                            "dense_score": r.dense_score,
                            "rerank_score": r.rerank_score,
                            "citation": r.chunk.citation_label(),
                            "deep_link": r.chunk.deep_link(),
                            "text": r.chunk.text,
                        }
                        for r in result.results
                    ],
                }
            )
        )
        return

    if not result.results:
        console.print("[yellow]no results[/]")
        console.print(
            f"[dim]lexical candidates: {result.lexical_candidates}, "
            f"dense candidates: {result.dense_candidates}[/]"
        )
        return

    header = (
        f"[bold]{len(result.results)}[/] results  "
        f"[dim]strategy={result.strategy.value}"
        + (f" fusion={result.fusion.value}" if result.fusion else "")
        + (" reranked" if result.reranked else "")
        + f" · lexical={result.lexical_candidates} dense={result.dense_candidates}"
        f" fused={result.fused_candidates} · {result.total_ms()}ms[/]"
    )
    console.print(header)
    console.print()

    for index, scored in enumerate(result.results, start=1):
        chunk = scored.chunk
        title = Text()
        title.append(f"{index}. ", style="bold cyan")
        title.append(f"{scored.score:.4f}  ", style="bold")
        title.append(chunk.citation_label(), style="bold white")
        body = chunk.text if show_text else _snippet(chunk.text)
        link = chunk.deep_link()
        subtitle = f"[dim]{scored.explain()}[/]"
        if link:
            subtitle += f"\n[dim blue]{link}[/]"
        console.print(
            Panel(
                body,
                title=title,
                subtitle=subtitle,
                title_align="left",
                subtitle_align="left",
                border_style="dim",
            )
        )


@app.command()
def ask(
    query: Annotated[str, typer.Argument(help="Your question")],
    collection: Annotated[str, typer.Option("--collection", "-c")] = "default",
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Chunks of context")] = 8,
    strategy: Annotated[RetrievalStrategy | None, typer.Option("--strategy", "-s")] = None,
    rerank: Annotated[bool | None, typer.Option("--rerank/--no-rerank")] = None,
    mmr: Annotated[bool, typer.Option("--mmr/--no-mmr", help="Diversify context")] = False,
    source_type: Annotated[
        list[SourceType] | None, typer.Option("--source", help="Restrict to source types")
    ] = None,
    show_sources: Annotated[
        bool, typer.Option("--sources/--no-sources", help="Print the cited sources")
    ] = True,
    show_context: Annotated[
        bool, typer.Option("--context", help="Print the retrieval that fed the answer")
    ] = False,
    verify: Annotated[
        bool | None,
        typer.Option("--verify/--no-verify", help="Check each claim against its cited source"),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output")] = False,
) -> None:
    """Ask a question and get an answer with citations.

    Citations print with their deep links, so a claim can be checked against the
    exact page, line range or timestamp it came from.
    """
    knowledge_base = _open()
    try:
        answer = knowledge_base.ask(
            query,
            collection=collection,
            top_k=top_k,
            strategy=strategy,
            rerank=rerank,
            use_mmr=mmr or None,
            source_types=list(source_type) if source_type else None,
            verify=verify,
        )
    except KBError as exc:
        _fail(exc)
        return

    if as_json:
        console.print_json(answer.model_dump_json(exclude={"retrieval"}))
        return

    console.print()
    console.print(_highlight_markers(answer.text))
    console.print()

    if show_sources and answer.citations:
        table = Table(box=None, show_header=False, pad_edge=False)
        table.add_column("", style="bold cyan", width=5)
        table.add_column("")
        for citation in answer.citations:
            label = f"[bold]{citation.document_title}[/]"
            if citation.label:
                label += f" [dim]— {citation.label}[/]"
            if citation.deep_link:
                label += f"\n[dim blue]{citation.deep_link}[/]"
            table.add_row(f"[{citation.marker}]", label)
        console.print(table)
        console.print()

    if answer.refused:
        console.print("[yellow]the sources did not cover this question[/]")

    flagged = answer.flagged_sentences()
    if flagged:
        console.print("[bold yellow]claims to check before trusting:[/]")
        for sentence in flagged:
            verdict = sentence.verdict.value if sentence.verdict else "unverified"
            console.print(f"  [yellow]![/] [bold]{verdict}[/] — {_snippet(sentence.text, 130)}")
            if sentence.verification_note:
                console.print(f"    [dim]{sentence.verification_note}[/]")
        console.print()

    footer = (
        f"[dim]{answer.generator}"
        + (f" ({answer.model})" if answer.model and answer.model != answer.generator else "")
        + f" · {answer.context_chunks} chunks, ~{answer.context_tokens} tokens"
        f" · {answer.total_ms()}ms[/]"
    )
    if answer.faithfulness is not None:
        flagged_count = len(answer.flagged_sentences())
        colour = "green" if flagged_count == 0 else "yellow"
        footer += (
            f"\n[{colour}]faithfulness {answer.faithfulness:.0%}"
            + (f" · {flagged_count} claim(s) flagged" if flagged_count else "")
            + "[/]"
        )
    console.print(footer)

    if show_context and answer.retrieval:
        console.print("\n[bold]retrieved context[/]")
        for index, scored in enumerate(answer.retrieval.results, start=1):
            console.print(
                f"  [cyan]{index}.[/] {scored.chunk.citation_label()} [dim]{scored.explain()}[/]"
            )


@app.command()
def stats(
    collection: Annotated[str, typer.Option("--collection", "-c")] = "default",
) -> None:
    """Show corpus statistics."""
    knowledge_base = _open()
    summary = knowledge_base.stats(collection)

    table = Table(title=f"collection: {collection}", box=None, title_justify="left")
    table.add_column("", style="dim")
    table.add_column("", justify="right")
    table.add_row("documents", f"{summary.n_documents:,}")
    table.add_row("chunks", f"{summary.n_chunks:,}")
    table.add_row("embedded", f"{summary.n_embedded:,}")
    if summary.n_chunks:
        coverage = summary.n_embedded / summary.n_chunks
        table.add_row("coverage", f"{coverage:.0%}")
    table.add_row("tokens (est.)", f"{summary.total_tokens:,}")
    table.add_row("embedding model", summary.embedding_model or "—")
    table.add_row("embedding dim", str(summary.embedding_dim or "—"))
    console.print(table)

    if summary.by_source_type:
        console.print()
        source_table = Table(box=None)
        source_table.add_column("source", style="cyan")
        source_table.add_column("documents", justify="right")
        for source, count in summary.by_source_type.items():
            source_table.add_row(source, str(count))
        console.print(source_table)

    others = [c for c in knowledge_base.collections() if c != collection]
    if others:
        console.print(f"\n[dim]other collections: {', '.join(others)}[/]")


@app.command()
def documents(
    collection: Annotated[str, typer.Option("--collection", "-c")] = "default",
    limit: Annotated[int, typer.Option("--limit", "-n")] = 30,
    source_type: Annotated[SourceType | None, typer.Option("--source")] = None,
) -> None:
    """List documents in a collection."""
    knowledge_base = _open()
    rows = knowledge_base.documents(collection, limit=limit, source_type=source_type)
    if not rows:
        console.print("[yellow]no documents[/]")
        return
    table = Table(box=None)
    table.add_column("title", style="bold", max_width=48, overflow="ellipsis")
    table.add_column("source", style="cyan")
    table.add_column("chunks", justify="right")
    table.add_column("id", style="dim")
    for document in rows:
        table.add_row(
            document.title, document.source_type.value, str(document.n_chunks), document.id
        )
    console.print(table)
    total = knowledge_base.store.count_documents(collection)
    if total > len(rows):
        console.print(f"[dim]showing {len(rows)} of {total}[/]")


@app.command()
def chunks(
    document_id: Annotated[str, typer.Argument(help="Document id")],
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
) -> None:
    """Inspect how a document was chunked.

    Chunk boundaries determine retrieval quality more than almost anything else,
    so they should be easy to look at.
    """
    knowledge_base = _open()
    try:
        document = knowledge_base.document(document_id)
    except KBError as exc:
        _fail(exc)
        return
    console.print(f"[bold]{document.title}[/] [dim]{document.uri}[/]\n")
    for chunk in knowledge_base.document_chunks(document_id)[:limit]:
        console.print(
            f"[cyan]#{chunk.ordinal}[/] [dim]{chunk.locator.label()} · "
            f"{chunk.kind.value} · ~{chunk.token_estimate} tok[/]"
        )
        console.print(f"  {_snippet(chunk.text, 220)}\n")


@app.command()
def embed(
    collection: Annotated[str, typer.Option("--collection", "-c")] = "default",
    rebuild: Annotated[
        bool, typer.Option("--rebuild", help="Discard existing vectors and re-embed everything")
    ] = False,
) -> None:
    """Embed chunks that have no vector yet, or rebuild the whole index."""
    knowledge_base = _open()
    count = (
        knowledge_base.reembed(collection=collection)
        if rebuild
        else knowledge_base.embed_pending(collection=collection)
    )
    console.print(f"embedded [bold green]{count}[/] chunks with {knowledge_base.embedder.model}")


@app.command()
def heatmap(
    collection: Annotated[str, typer.Option("--collection", "-c")] = "default",
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
) -> None:
    """Show which chunks are actually being retrieved.

    A chunk that never surfaces is either redundant or unreachable. Both are
    worth knowing; neither is visible without logging retrievals.
    """
    knowledge_base = _open()
    rows = knowledge_base.heatmap(collection, limit=limit)
    if not rows:
        console.print("[yellow]no retrievals logged yet — run some searches first[/]")
        return
    table = Table(box=None)
    table.add_column("hits", justify="right", style="bold")
    table.add_column("avg rank", justify="right")
    table.add_column("document", max_width=40, overflow="ellipsis")
    table.add_column("chunk", style="dim")
    for row in rows:
        table.add_row(
            str(row["hits"]),
            f"{row['avg_rank']:.1f}",
            row["document_title"],
            row["chunk_id"],
        )
    console.print(table)


@app.command()
def delete(
    target: Annotated[
        str, typer.Argument(help="Document id, or a collection name with --collection")
    ],
    is_collection: Annotated[
        bool, typer.Option("--collection", help="Treat the target as a collection name")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
) -> None:
    """Delete a document, or an entire collection."""
    knowledge_base = _open()
    if is_collection:
        if not yes and not typer.confirm(f"Delete the entire collection {target!r}?"):
            raise typer.Abort
        removed = knowledge_base.delete_collection(target)
        console.print(f"deleted collection {target!r} ([bold]{removed}[/] documents)")
        return
    try:
        document = knowledge_base.document(target)
    except KBError as exc:
        _fail(exc)
        return
    if not yes and not typer.confirm(f"Delete {document.title!r}?"):
        raise typer.Abort
    knowledge_base.delete_document(target)
    console.print(f"deleted {document.title!r}")


@app.command()
def serve(
    host: Annotated[str | None, typer.Option("--host")] = None,
    port: Annotated[int | None, typer.Option("--port", "-p")] = None,
    reload: Annotated[bool, typer.Option("--reload", help="Auto-reload on code changes")] = False,
) -> None:
    """Run the HTTP API."""
    import uvicorn

    settings = _settings()
    console.print(
        f"[bold]kb {__version__}[/] serving on "
        f"http://{host or settings.api_host}:{port or settings.api_port}  "
        f"[dim](docs at /docs)[/]"
    )
    uvicorn.run(
        "kb.api.app:create_app",
        factory=True,
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


@app.command()
def config() -> None:
    """Print the effective configuration.

    Useful for confirming which provider and retrieval settings are actually in
    force, since every one of them can come from the environment.
    """
    settings = _settings()
    payload = settings.model_dump(mode="json")
    for key in list(payload):
        if key.endswith("_api_key"):
            payload[key] = "<set>" if payload[key] else ""
    console.print_json(json.dumps(payload, indent=2, default=str))


_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _highlight_markers(text: str) -> str:
    """Make citation markers visually distinct in the terminal."""
    return _MARKER_RE.sub(lambda m: f"[bold cyan]\\[{m.group(1)}][/]", text)


def _snippet(text: str, width: int = 300) -> str:
    body = " ".join(text.split())
    return body if len(body) <= width else f"{body[:width].rstrip()}…"


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
