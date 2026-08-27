"""CLI tests via typer's runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kb.cli import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "kbdata"


@pytest.fixture
def base_args(data_dir: Path) -> list[str]:
    return ["--data-dir", str(data_dir)]


@pytest.fixture
def ingested(runner: CliRunner, base_args: list[str], corpus_dir: Path) -> list[str]:
    result = runner.invoke(app, [*base_args, "ingest", str(corpus_dir)])
    assert result.exit_code == 0, result.output
    return base_args


def test_version(runner: CliRunner) -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "kb" in result.output


def test_help_lists_commands(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("ingest", "search", "stats", "serve", "embed"):
        assert command in result.output


class TestIngest:
    def test_reports_counts(
        self, runner: CliRunner, base_args: list[str], corpus_dir: Path
    ) -> None:
        result = runner.invoke(app, [*base_args, "ingest", str(corpus_dir)])
        assert result.exit_code == 0, result.output
        assert "documents" in result.output
        assert "chunks" in result.output

    def test_second_run_reports_unchanged(
        self, runner: CliRunner, base_args: list[str], corpus_dir: Path
    ) -> None:
        runner.invoke(app, [*base_args, "ingest", str(corpus_dir)])
        result = runner.invoke(app, [*base_args, "ingest", str(corpus_dir)])
        assert result.exit_code == 0
        assert "unchanged" in result.output

    def test_failure_is_reported_without_crashing(
        self, runner: CliRunner, base_args: list[str]
    ) -> None:
        result = runner.invoke(app, [*base_args, "ingest", "/nope/missing.md"])
        assert result.exit_code == 0
        assert "failed" in result.output

    def test_add_text_from_stdin(self, runner: CliRunner, base_args: list[str]) -> None:
        result = runner.invoke(
            app,
            [*base_args, "add-text", "Piped Doc"],
            input="# Piped\n\nContent about reranking.\n",
        )
        assert result.exit_code == 0
        assert "Piped Doc" in result.output


class TestSearch:
    def test_prints_scores_and_links(self, runner: CliRunner, ingested: list[str]) -> None:
        result = runner.invoke(app, [*ingested, "search", "reciprocal rank fusion", "-k", "2"])
        assert result.exit_code == 0, result.output
        assert "strategy=hybrid" in result.output
        assert "bm25=" in result.output or "dense=" in result.output

    def test_json_output_is_parseable(self, runner: CliRunner, ingested: list[str]) -> None:
        result = runner.invoke(app, [*ingested, "search", "dense vectors", "-k", "2", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["strategy"] == "hybrid"
        assert payload["results"]
        assert payload["results"][0]["citation"]
        assert payload["results"][0]["deep_link"]

    @pytest.mark.parametrize("strategy", ["lexical", "dense", "hybrid"])
    def test_strategy_flag(self, runner: CliRunner, ingested: list[str], strategy: str) -> None:
        result = runner.invoke(
            app, [*ingested, "search", "retrieval", "-s", strategy, "-k", "2", "--json"]
        )
        assert result.exit_code == 0
        assert json.loads(result.output)["strategy"] == strategy

    @pytest.mark.parametrize("fusion", ["rrf", "weighted", "max"])
    def test_fusion_flag(self, runner: CliRunner, ingested: list[str], fusion: str) -> None:
        result = runner.invoke(
            app, [*ingested, "search", "retrieval", "-f", fusion, "-k", "2", "--json"]
        )
        assert result.exit_code == 0
        assert json.loads(result.output)["fusion"] == fusion

    def test_mmr_flag(self, runner: CliRunner, ingested: list[str]) -> None:
        result = runner.invoke(
            app, [*ingested, "search", "retrieval", "--mmr", "-k", "2", "--json"]
        )
        assert result.exit_code == 0
        assert "mmr_ms" in json.loads(result.output)["timings_ms"]

    def test_no_results_message(self, runner: CliRunner, ingested: list[str]) -> None:
        result = runner.invoke(app, [*ingested, "search", "zzqxv nothingmatches", "-s", "lexical"])
        assert result.exit_code == 0
        assert "no results" in result.output


class TestInspection:
    def test_stats(self, runner: CliRunner, ingested: list[str]) -> None:
        result = runner.invoke(app, [*ingested, "stats"])
        assert result.exit_code == 0
        assert "documents" in result.output
        assert "coverage" in result.output

    def test_documents_listing(self, runner: CliRunner, ingested: list[str]) -> None:
        result = runner.invoke(app, [*ingested, "documents"])
        assert result.exit_code == 0
        assert "Retrieval" in result.output

    def test_chunks_inspection(self, runner: CliRunner, ingested: list[str]) -> None:
        listing = runner.invoke(app, [*ingested, "documents", "--limit", "1"])
        document_id = next(token for token in listing.output.split() if token.startswith("doc_"))
        result = runner.invoke(app, [*ingested, "chunks", document_id])
        assert result.exit_code == 0
        assert "#0" in result.output

    def test_chunks_unknown_document_exits_nonzero(
        self, runner: CliRunner, ingested: list[str]
    ) -> None:
        result = runner.invoke(app, [*ingested, "chunks", "doc_nope"])
        assert result.exit_code == 1
        assert "error" in result.output

    def test_heatmap_is_empty_before_searching(
        self, runner: CliRunner, ingested: list[str]
    ) -> None:
        result = runner.invoke(app, [*ingested, "heatmap"])
        assert result.exit_code == 0
        assert "no retrievals logged" in result.output

    def test_heatmap_after_searching(self, runner: CliRunner, ingested: list[str]) -> None:
        runner.invoke(app, [*ingested, "search", "fusion", "-k", "2"])
        result = runner.invoke(app, [*ingested, "heatmap"])
        assert result.exit_code == 0
        assert "hits" in result.output

    def test_config_prints_json_and_masks_keys(
        self, runner: CliRunner, base_args: list[str]
    ) -> None:
        result = runner.invoke(app, [*base_args, "config"])
        assert result.exit_code == 0
        assert "embedding_provider" in result.output
        # The raw key value must never be printed.
        assert "sk-" not in result.output


class TestEmbedAndDelete:
    def test_embed_backfill(
        self, runner: CliRunner, base_args: list[str], corpus_dir: Path
    ) -> None:
        runner.invoke(app, [*base_args, "ingest", str(corpus_dir), "--no-embed"])
        result = runner.invoke(app, [*base_args, "embed"])
        assert result.exit_code == 0
        assert "embedded" in result.output

    def test_delete_document_requires_confirmation(
        self, runner: CliRunner, ingested: list[str]
    ) -> None:
        listing = runner.invoke(app, [*ingested, "documents", "--limit", "1"])
        document_id = next(t for t in listing.output.split() if t.startswith("doc_"))
        aborted = runner.invoke(app, [*ingested, "delete", document_id], input="n\n")
        assert aborted.exit_code != 0

        confirmed = runner.invoke(app, [*ingested, "delete", document_id, "-y"])
        assert confirmed.exit_code == 0
        assert "deleted" in confirmed.output

    def test_delete_collection(self, runner: CliRunner, ingested: list[str]) -> None:
        result = runner.invoke(app, [*ingested, "delete", "default", "--collection", "-y"])
        assert result.exit_code == 0
        assert "deleted collection" in result.output
