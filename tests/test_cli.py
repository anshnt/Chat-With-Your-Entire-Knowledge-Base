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


class TestAsk:
    def test_prints_answer_and_sources(self, runner: CliRunner, ingested: list[str]) -> None:
        result = runner.invoke(
            app, [*ingested, "ask", "what does the damping constant default to?"]
        )
        assert result.exit_code == 0, result.output
        assert "60" in result.output
        assert "[1]" in result.output
        assert "extractive" in result.output

    def test_json_output(self, runner: CliRunner, ingested: list[str]) -> None:
        result = runner.invoke(app, [*ingested, "ask", "reciprocal rank fusion", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["text"]
        assert payload["generator"] == "extractive"
        assert "citations" in payload

    def test_sources_can_be_hidden(self, runner: CliRunner, ingested: list[str]) -> None:
        result = runner.invoke(app, [*ingested, "ask", "reciprocal rank fusion", "--no-sources"])
        assert result.exit_code == 0
        assert "extractive" in result.output

    def test_context_flag_shows_the_retrieval(self, runner: CliRunner, ingested: list[str]) -> None:
        result = runner.invoke(app, [*ingested, "ask", "fusion", "--context"])
        assert result.exit_code == 0
        assert "retrieved context" in result.output

    def test_off_corpus_question_says_so(self, runner: CliRunner, ingested: list[str]) -> None:
        result = runner.invoke(app, [*ingested, "ask", "who won the 1998 world cup?"])
        assert result.exit_code == 0
        assert "did not cover" in result.output


class TestAskVerification:
    def test_faithfulness_is_reported(self, runner: CliRunner, ingested: list[str]) -> None:
        result = runner.invoke(
            app, [*ingested, "ask", "what does the damping constant k default to?"]
        )
        assert result.exit_code == 0
        assert "faithfulness" in result.output

    def test_verification_can_be_skipped(self, runner: CliRunner, ingested: list[str]) -> None:
        result = runner.invoke(app, [*ingested, "ask", "fusion", "--no-verify"])
        assert result.exit_code == 0
        assert "faithfulness" not in result.output

    def test_json_output_carries_verdicts(self, runner: CliRunner, ingested: list[str]) -> None:
        result = runner.invoke(app, [*ingested, "ask", "fusion", "--json"])
        payload = json.loads(result.output)
        assert payload["verified"] is True
        assert payload["sentences"]


class TestEvalCommands:
    def test_generate_writes_a_golden_set(
        self, runner: CliRunner, ingested: list[str], tmp_path: Path
    ) -> None:
        output = tmp_path / "golden.yaml"
        result = runner.invoke(
            app, [*ingested, "eval", "generate", "-o", str(output), "--per-document", "5"]
        )
        assert result.exit_code == 0, result.output
        assert output.is_file()
        assert "questions to" in result.output

    def test_generate_on_an_empty_collection_fails_clearly(
        self, runner: CliRunner, base_args: list[str], tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app, [*base_args, "eval", "generate", "-o", str(tmp_path / "g.yaml")]
        )
        assert result.exit_code == 1
        assert "empty" in result.output

    def test_run_reports_metrics(
        self, runner: CliRunner, ingested: list[str], tmp_path: Path
    ) -> None:
        output = tmp_path / "golden.yaml"
        runner.invoke(
            app, [*ingested, "eval", "generate", "-o", str(output), "--per-document", "5"]
        )
        result = runner.invoke(app, [*ingested, "eval", "run", str(output)])
        assert result.exit_code == 0, result.output
        assert "ndcg@5" in result.output

    def test_sweep_compares_configurations(
        self, runner: CliRunner, ingested: list[str], tmp_path: Path
    ) -> None:
        output = tmp_path / "golden.yaml"
        runner.invoke(
            app, [*ingested, "eval", "generate", "-o", str(output), "--per-document", "5"]
        )
        result = runner.invoke(
            app, [*ingested, "eval", "run", str(output), "--sweep", "strategies"]
        )
        assert result.exit_code == 0, result.output
        for label in ("lexical", "dense", "hybrid"):
            assert label in result.output

    def test_unknown_sweep_fails(
        self, runner: CliRunner, ingested: list[str], tmp_path: Path
    ) -> None:
        output = tmp_path / "golden.yaml"
        runner.invoke(
            app, [*ingested, "eval", "generate", "-o", str(output), "--per-document", "5"]
        )
        result = runner.invoke(app, [*ingested, "eval", "run", str(output), "--sweep", "nonsense"])
        assert result.exit_code == 1
        assert "unknown sweep" in result.output

    def test_report_artifacts_are_written(
        self, runner: CliRunner, ingested: list[str], tmp_path: Path
    ) -> None:
        output = tmp_path / "golden.yaml"
        runner.invoke(
            app, [*ingested, "eval", "generate", "-o", str(output), "--per-document", "5"]
        )
        report_dir = tmp_path / "report"
        result = runner.invoke(
            app, [*ingested, "eval", "run", str(output), "--report", str(report_dir)]
        )
        assert result.exit_code == 0
        assert (report_dir / "evaluation.md").is_file()
        assert (report_dir / "evaluation.json").is_file()
        assert (report_dir / "evaluation-metrics.svg").is_file()

    def test_fail_under_gates_on_the_metric(
        self, runner: CliRunner, ingested: list[str], tmp_path: Path
    ) -> None:
        """A retrieval regression should fail a build, not be found in production."""
        output = tmp_path / "golden.yaml"
        runner.invoke(
            app, [*ingested, "eval", "generate", "-o", str(output), "--per-document", "5"]
        )
        passing = runner.invoke(app, [*ingested, "eval", "run", str(output), "--fail-under", "0.0"])
        assert passing.exit_code == 0
        assert "meets the" in passing.output

        failing = runner.invoke(
            app, [*ingested, "eval", "run", str(output), "--fail-under", "1.01"]
        )
        assert failing.exit_code == 1
        assert "below the" in failing.output

    def test_mine_requires_logged_queries(
        self, runner: CliRunner, ingested: list[str], tmp_path: Path
    ) -> None:
        result = runner.invoke(app, [*ingested, "eval", "mine", "-o", str(tmp_path / "mined.yaml")])
        assert result.exit_code == 1
        assert "no queries logged" in result.output

    def test_mine_seeds_from_logged_queries(
        self, runner: CliRunner, ingested: list[str], tmp_path: Path
    ) -> None:
        runner.invoke(app, [*ingested, "search", "reciprocal rank fusion"])
        output = tmp_path / "mined.yaml"
        result = runner.invoke(app, [*ingested, "eval", "mine", "-o", str(output)])
        assert result.exit_code == 0
        assert output.is_file()
        assert "fill in must_contain" in result.output


class TestShippedGoldenSet:
    def test_the_committed_paraphrase_set_is_valid(self) -> None:
        """The set shipped in the repo must load and be well-formed."""
        from kb.eval import GoldenSet

        path = Path(__file__).resolve().parents[1] / "eval" / "golden-paraphrase.yaml"
        if not path.is_file():  # pragma: no cover - present in the repo
            pytest.skip("paraphrase golden set not present")
        golden = GoldenSet.load(path)
        assert len(golden) >= 15
        assert all(q.must_contain for q in golden.queries)
        assert len({q.id for q in golden.queries}) == len(golden)


class TestConnectorsCommand:
    def test_lists_connectors_in_precedence_order(
        self, runner: CliRunner, base_args: list[str]
    ) -> None:
        result = runner.invoke(app, [*base_args, "connectors"])
        assert result.exit_code == 0
        for name in ("notion", "markdown", "pdf", "youtube", "github", "web"):
            assert name in result.output
        # The web connector claims any URL, so it must come last.
        assert result.output.index("youtube") < result.output.index("any http(s) URL")

    def test_ingest_accepts_the_connector_options(
        self, runner: CliRunner, base_args: list[str], tmp_path: Path
    ) -> None:
        checkout = tmp_path / "repo"
        (checkout / "src").mkdir(parents=True)
        (checkout / "src" / "app.py").write_text("def handler():\n    return 1\n")
        result = runner.invoke(
            app,
            [*base_args, "ingest", "anshnt/demo", "--local", str(checkout), "--ref", "main"],
        )
        assert result.exit_code == 0, result.output
        assert "documents" in result.output


class TestMapCommand:
    def test_prints_the_cluster_table(self, runner: CliRunner, ingested: list[str]) -> None:
        result = runner.invoke(app, [*ingested, "map"])
        assert result.exit_code == 0, result.output
        assert "clusters" in result.output
        assert "distinctive terms" in result.output

    def test_reports_pca_variance_as_a_caveat(self, runner: CliRunner, ingested: list[str]) -> None:
        result = runner.invoke(app, [*ingested, "map", "--method", "pca"])
        assert result.exit_code == 0
        assert "variance" in result.output

    def test_flags_never_retrieved_clusters(self, runner: CliRunner, ingested: list[str]) -> None:
        result = runner.invoke(app, [*ingested, "map"])
        assert result.exit_code == 0
        assert "never been retrieved" in result.output

    def test_writes_an_svg(self, runner: CliRunner, ingested: list[str], tmp_path: Path) -> None:
        output = tmp_path / "nested" / "map.svg"
        result = runner.invoke(app, [*ingested, "map", "-o", str(output)])
        assert result.exit_code == 0, result.output
        assert output.is_file()
        import xml.dom.minidom

        xml.dom.minidom.parse(str(output))

    def test_empty_collection(self, runner: CliRunner, base_args: list[str]) -> None:
        result = runner.invoke(app, [*base_args, "map"])
        assert result.exit_code == 0
        assert "nothing to plot" in result.output
