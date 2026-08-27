"""Evaluation harness tests: golden sets, resolution, runner, and reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kb.config import Settings
from kb.errors import EvaluationError
from kb.eval import (
    EvalRunner,
    GoldenQuery,
    GoldenSet,
    generate_golden_set,
    golden_set_from_pairs,
    json_report,
    markdown_report,
    metric_bar_chart,
    mine_golden_set,
    rank_distribution_chart,
    resolve_golden_set,
    write_report,
)
from kb.eval.dataset import normalize_for_match
from kb.eval.synthesize import (
    question_from_sentence,
    snippet_for,
    strip_inline_markdown,
)
from kb.knowledge_base import KnowledgeBase

CORPUS = {
    "retrieval.md": """# Retrieval

## Hybrid search

Hybrid search combines BM25 lexical matching with dense vector retrieval over
the same corpus. Lexical matching finds exact identifiers and rare terms.

## Reciprocal rank fusion

Reciprocal Rank Fusion combines ranked lists using ranks rather than raw scores.
The damping constant k defaults to 60 in the standard formulation.

## Reranking

A cross-encoder reranker scores each query-document pair jointly and is more
accurate than comparing independent embeddings.
""",
    "storage.md": """# Storage

## SQLite

The store keeps documents, chunks, the full text index and the dense vectors in
a single SQLite file. Vectors are little-endian float32 blobs.

## Filtering

Filtering is applied before the top k selection so a selective filter does not
silently reduce recall.
""",
}


@pytest.fixture
def eval_kb(tmp_path: Path) -> KnowledgeBase:
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, body in CORPUS.items():
        (docs / name).write_text(body, encoding="utf-8")
    settings = Settings(
        data_dir=tmp_path / "data", embedding_dim=384, chunk_size=600, min_chunk_size=50
    )
    instance = KnowledgeBase(settings)
    report = instance.ingest(str(docs))
    assert report.chunks_created > 0
    return instance


@pytest.fixture
def golden() -> GoldenSet:
    return GoldenSet(
        name="test",
        queries=[
            GoldenQuery(
                query="what does the damping constant default to?",
                must_contain=["damping constant k defaults to 60"],
            ),
            GoldenQuery(
                query="how are lexical and dense results combined?",
                must_contain=["combines ranked lists using ranks"],
            ),
            GoldenQuery(
                query="how are the vectors stored on disk?",
                must_contain=["little-endian float32 blobs"],
            ),
        ],
    )


# --------------------------------------------------------------------------- #
# golden sets
# --------------------------------------------------------------------------- #


class TestGoldenSet:
    def test_ids_are_assigned(self, golden: GoldenSet) -> None:
        assert [q.id for q in golden.queries] == ["q001", "q002", "q003"]

    def test_explicit_ids_are_kept(self) -> None:
        payload = GoldenSet(queries=[GoldenQuery(id="custom", query="q", must_contain=["x"])])
        assert payload.queries[0].id == "custom"

    def test_duplicate_ids_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            GoldenSet(
                queries=[
                    GoldenQuery(id="a", query="one", must_contain=["x"]),
                    GoldenQuery(id="a", query="two", must_contain=["y"]),
                ]
            )

    def test_a_query_with_no_expectation_is_rejected(self) -> None:
        """Silently accepting one would produce a permanent zero."""
        with pytest.raises(ValueError, match="no expected sources"):
            GoldenQuery(query="unanswerable")

    def test_yaml_round_trip(self, golden: GoldenSet, tmp_path: Path) -> None:
        path = tmp_path / "golden.yaml"
        golden.save(path)
        reloaded = GoldenSet.load(path)
        assert len(reloaded) == len(golden)
        assert reloaded.queries[0].query == golden.queries[0].query

    def test_json_round_trip(self, golden: GoldenSet, tmp_path: Path) -> None:
        path = tmp_path / "golden.json"
        golden.save(path)
        assert len(GoldenSet.load(path)) == 3

    def test_jsonl_loading(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.jsonl"
        path.write_text(
            "\n".join(json.dumps({"query": f"q{i}", "must_contain": ["x"]}) for i in range(3))
        )
        assert len(GoldenSet.load(path)) == 3

    def test_bare_list_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "golden.yaml"
        path.write_text("- query: a question\n  must_contain: ['snippet']\n")
        assert len(GoldenSet.load(path)) == 1

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(EvaluationError, match="not found"):
            GoldenSet.load(tmp_path / "nope.yaml")

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("queries: [unclosed\n")
        with pytest.raises(EvaluationError, match="could not parse"):
            GoldenSet.load(path)

    def test_tag_filtering(self) -> None:
        payload = GoldenSet(
            queries=[
                GoldenQuery(query="a", must_contain=["x"], tags=["easy"]),
                GoldenQuery(query="b", must_contain=["y"], tags=["hard"]),
            ]
        )
        assert len(payload.tagged("easy")) == 1
        assert payload.all_tags() == ["easy", "hard"]

    def test_from_pairs(self) -> None:
        payload = golden_set_from_pairs([("q1", "s1"), ("q2", "s2")])
        assert len(payload) == 2
        assert payload.queries[0].must_contain == ["s1"]


class TestResolution:
    def test_snippets_resolve_to_chunks(self, eval_kb: KnowledgeBase, golden: GoldenSet) -> None:
        resolved, warnings = resolve_golden_set(golden, eval_kb.store)
        assert len(resolved) == 3
        assert all(r.is_usable for r in resolved), warnings
        assert not warnings

    def test_matching_ignores_whitespace_differences(self, eval_kb: KnowledgeBase) -> None:
        """Chunk text keeps newlines; a snippet from a sentence is space-joined.

        Matching literally silently drops every snippet crossing a line break.
        """
        payload = GoldenSet(
            queries=[
                GoldenQuery(
                    query="q",
                    # This span crosses a line break in the source file.
                    must_contain=["dense vector retrieval over the same corpus"],
                )
            ]
        )
        resolved, warnings = resolve_golden_set(payload, eval_kb.store)
        assert resolved[0].is_usable, warnings

    def test_unresolvable_snippet_is_warned_not_silent(self, eval_kb: KnowledgeBase) -> None:
        """A drifted golden set must not look like a retrieval regression."""
        payload = GoldenSet(queries=[GoldenQuery(query="q", must_contain=["text that is nowhere"])])
        resolved, warnings = resolve_golden_set(payload, eval_kb.store)
        assert not resolved[0].is_usable
        assert any("unresolved" in w for w in warnings)
        assert any("excluded from scoring" in w for w in warnings)

    def test_document_title_resolution(self, eval_kb: KnowledgeBase) -> None:
        payload = GoldenSet(queries=[GoldenQuery(query="q", document_titles=["Retrieval"])])
        resolved, _ = resolve_golden_set(payload, eval_kb.store)
        assert resolved[0].is_usable
        assert len(resolved[0].relevance) > 1

    def test_unknown_title_is_reported(self, eval_kb: KnowledgeBase) -> None:
        payload = GoldenSet(queries=[GoldenQuery(query="q", document_titles=["No Such Document"])])
        _, warnings = resolve_golden_set(payload, eval_kb.store)
        assert any("document_title" in w for w in warnings)

    def test_grades_are_carried_through(self, eval_kb: KnowledgeBase) -> None:
        snippet = "damping constant k defaults to 60"
        payload = GoldenSet(
            queries=[GoldenQuery(query="q", must_contain=[snippet], grades={snippet: 1})]
        )
        resolved, _ = resolve_golden_set(payload, eval_kb.store)
        assert set(resolved[0].relevance.values()) == {1}

    def test_default_grade_is_two(self, eval_kb: KnowledgeBase, golden: GoldenSet) -> None:
        resolved, _ = resolve_golden_set(golden, eval_kb.store)
        assert set(resolved[0].relevance.values()) == {2}


def test_normalize_for_match() -> None:
    assert normalize_for_match("  Two   lines\nhere ") == "two lines here"


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #


class TestSnippetFor:
    def test_takes_from_the_middle(self) -> None:
        """Openings are formulaic and repeat, so a leading snippet over-matches."""
        sentence = " ".join(f"w{i}" for i in range(20))
        snippet = snippet_for(sentence, words=4)
        assert not snippet.startswith("w0")

    def test_short_sentence_is_used_whole(self) -> None:
        assert snippet_for("three words here", words=8) == "three words here"


class TestStripInlineMarkdown:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("the `code` value", "the code value"),
            ("**bold** text", "bold text"),
            ("a [link](http://x) here", "a link here"),
            ("~~struck~~ out", "struck out"),
        ],
    )
    def test_formatting_is_removed(self, raw: str, expected: str) -> None:
        assert strip_inline_markdown(raw) == expected


class TestQuestionFromSentence:
    def test_defaults_to_pattern(self) -> None:
        question = question_from_sentence("The damping constant k defaults to 60.")
        assert question is not None
        assert question.startswith("what does")
        assert question.endswith("?")

    def test_is_pattern(self) -> None:
        assert question_from_sentence("Recall at k is a retrieval metric.") is not None

    @pytest.mark.parametrize(
        "sentence",
        [
            # A clause boundary means the matched verb is not the main verb.
            "A reranker scores pairs jointly and is more accurate than a bi-encoder.",
            "Citations cannot be resolved until the text is complete.",
            # Non-referential subjects are unanswerable however good retrieval is.
            "There is no fallback path available.",
            "It is configurable in the settings file.",
        ],
    )
    def test_degenerate_sentences_are_rejected(self, sentence: str) -> None:
        assert question_from_sentence(sentence) is None

    def test_unmatched_sentence_returns_none(self) -> None:
        assert question_from_sentence("Ingest the corpus first.") is None


class TestGenerateGoldenSet:
    def test_produces_usable_questions(self, eval_kb: KnowledgeBase) -> None:
        chunks = [c for batch in eval_kb.store.iter_chunks() for c in batch]
        payload = generate_golden_set(chunks, per_document=5, limit=20)
        assert payload.queries
        for query in payload.queries:
            assert query.query.endswith("?")
            assert query.must_contain

    def test_labels_are_correct_by_construction(self, eval_kb: KnowledgeBase) -> None:
        """Each question came from a chunk, so that chunk must answer it."""
        chunks = [c for batch in eval_kb.store.iter_chunks() for c in batch]
        payload = generate_golden_set(chunks, per_document=5, limit=20)
        resolved, warnings = resolve_golden_set(payload, eval_kb.store)
        assert all(r.is_usable for r in resolved), warnings

    def test_per_document_cap_is_respected(self, eval_kb: KnowledgeBase) -> None:
        """One long document must not dominate the set."""
        chunks = [c for batch in eval_kb.store.iter_chunks() for c in batch]
        payload = generate_golden_set(chunks, per_document=1, limit=50)
        by_document: dict[str, int] = {}
        for query in payload.queries:
            # Recover the document from the note, which records the source sentence.
            by_document[query.notes] = by_document.get(query.notes, 0) + 1
        documents = {c.document_id for c in chunks}
        assert len(payload.queries) <= len(documents)

    def test_limit_is_respected(self, eval_kb: KnowledgeBase) -> None:
        chunks = [c for batch in eval_kb.store.iter_chunks() for c in batch]
        assert len(generate_golden_set(chunks, per_document=10, limit=2).queries) <= 2

    def test_questions_are_deduplicated(self, eval_kb: KnowledgeBase) -> None:
        chunks = [c for batch in eval_kb.store.iter_chunks() for c in batch]
        payload = generate_golden_set(chunks, per_document=10, limit=50)
        questions = [q.query.lower() for q in payload.queries]
        assert len(questions) == len(set(questions))

    def test_deterministic(self, eval_kb: KnowledgeBase) -> None:
        chunks = [c for batch in eval_kb.store.iter_chunks() for c in batch]
        first = generate_golden_set(chunks, per_document=5, limit=20)
        second = generate_golden_set(chunks, per_document=5, limit=20)
        assert [q.query for q in first.queries] == [q.query for q in second.queries]

    def test_empty_corpus(self) -> None:
        assert generate_golden_set([]).queries == []


def test_mine_golden_set_leaves_labelling_to_a_human() -> None:
    payload = mine_golden_set(["real query one", "real query two"])
    assert len(payload) == 2
    assert all("TODO" in q.must_contain[0] for q in payload.queries)


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #


class TestEvalRunner:
    def test_scores_every_resolvable_query(self, eval_kb: KnowledgeBase, golden: GoldenSet) -> None:
        run = EvalRunner(eval_kb).run(golden)
        assert run.n_queries == 3
        assert run.n_scored == 3
        assert run.n_excluded == 0

    def test_metrics_are_computed(self, eval_kb: KnowledgeBase, golden: GoldenSet) -> None:
        run = EvalRunner(eval_kb).run(golden)
        for name in ("hit_rate@5", "recall@5", "ndcg@5", "mrr", "map"):
            assert name in run.metrics
            assert 0.0 <= run.metric(name) <= 1.0

    def test_retrieval_actually_finds_the_answers(
        self, eval_kb: KnowledgeBase, golden: GoldenSet
    ) -> None:
        """A sanity floor: if this fails the harness is measuring nothing."""
        run = EvalRunner(eval_kb).run(golden)
        assert run.metric("hit_rate@10") > 0.5

    def test_unresolvable_queries_are_excluded_not_zeroed(self, eval_kb: KnowledgeBase) -> None:
        payload = GoldenSet(
            queries=[
                GoldenQuery(query="answerable", must_contain=["damping constant k defaults to 60"]),
                GoldenQuery(query="drifted", must_contain=["text that is nowhere"]),
            ]
        )
        run = EvalRunner(eval_kb).run(payload)
        assert run.n_scored == 1
        assert run.n_excluded == 1
        # The excluded query must not drag the mean down.
        assert run.metric("hit_rate@5") == 1.0

    def test_per_query_detail_is_reported(self, eval_kb: KnowledgeBase, golden: GoldenSet) -> None:
        run = EvalRunner(eval_kb).run(golden)
        result = run.results[0]
        assert result.query_id
        assert result.n_relevant > 0
        assert result.top_hit
        assert result.latency_ms >= 0

    def test_latency_is_summarised(self, eval_kb: KnowledgeBase, golden: GoldenSet) -> None:
        run = EvalRunner(eval_kb).run(golden)
        assert run.mean_latency_ms > 0
        assert run.p95_latency_ms >= run.mean_latency_ms * 0.5

    def test_worst_lists_the_weakest_queries(
        self, eval_kb: KnowledgeBase, golden: GoldenSet
    ) -> None:
        run = EvalRunner(eval_kb).run(golden)
        worst = run.worst("mrr", limit=2)
        assert len(worst) <= 2
        if len(worst) == 2:
            assert worst[0].metrics["mrr"] <= worst[1].metrics["mrr"]

    def test_overrides_reach_retrieval(self, eval_kb: KnowledgeBase, golden: GoldenSet) -> None:
        run = EvalRunner(eval_kb).run(golden, overrides={"strategy": "lexical"})
        assert run.config["strategy"] == "lexical"

    def test_compare_runs_every_configuration(
        self, eval_kb: KnowledgeBase, golden: GoldenSet
    ) -> None:
        runs = EvalRunner(eval_kb).compare(
            golden,
            {
                "lexical": {"strategy": "lexical", "rerank": False},
                "dense": {"strategy": "dense", "rerank": False},
                "hybrid": {"strategy": "hybrid", "rerank": False},
            },
        )
        assert [r.label for r in runs] == ["lexical", "dense", "hybrid"]
        assert all(r.n_scored == 3 for r in runs)

    def test_hybrid_recall_is_never_below_either_part(
        self, eval_kb: KnowledgeBase, golden: GoldenSet
    ) -> None:
        """The claim hybrid retrieval rests on, measured rather than asserted."""
        runs = EvalRunner(eval_kb, cutoffs=(10,)).compare(
            golden,
            {
                "lexical": {"strategy": "lexical", "rerank": False},
                "dense": {"strategy": "dense", "rerank": False},
                "hybrid": {"strategy": "hybrid", "rerank": False},
            },
        )
        by_label = {r.label: r.metric("recall@10") for r in runs}
        assert by_label["hybrid"] >= max(by_label["lexical"], by_label["dense"]) - 1e-9

    def test_with_answers_adds_faithfulness(
        self, eval_kb: KnowledgeBase, golden: GoldenSet
    ) -> None:
        run = EvalRunner(eval_kb).run(golden, with_answers=True)
        assert "faithfulness" in run.metrics or "refusal_rate" in run.metrics
        assert any(r.answer for r in run.results)

    def test_empty_golden_set(self, eval_kb: KnowledgeBase) -> None:
        run = EvalRunner(eval_kb).run(GoldenSet(name="empty"))
        assert run.n_queries == 0
        assert run.metrics == {}


# --------------------------------------------------------------------------- #
# reports
# --------------------------------------------------------------------------- #


class TestReports:
    def test_markdown_contains_the_headline_table(
        self, eval_kb: KnowledgeBase, golden: GoldenSet
    ) -> None:
        run = EvalRunner(eval_kb).run(golden, label="hybrid")
        markdown = markdown_report([run])
        assert "# Retrieval evaluation" in markdown
        assert "ndcg@5" in markdown
        assert "hybrid" in markdown

    def test_markdown_shows_deltas_for_a_sweep(
        self, eval_kb: KnowledgeBase, golden: GoldenSet
    ) -> None:
        runs = EvalRunner(eval_kb).compare(
            golden,
            {"lexical": {"strategy": "lexical"}, "hybrid": {"strategy": "hybrid"}},
        )
        markdown = markdown_report(runs)
        assert "Change vs `lexical`" in markdown

    def test_markdown_reports_exclusions(self, eval_kb: KnowledgeBase) -> None:
        payload = GoldenSet(queries=[GoldenQuery(query="drifted", must_contain=["nowhere at all"])])
        markdown = markdown_report([EvalRunner(eval_kb).run(payload)])
        assert "excluded" in markdown

    def test_pipes_in_a_query_do_not_break_the_table(self, eval_kb: KnowledgeBase) -> None:
        payload = GoldenSet(
            queries=[
                GoldenQuery(
                    query="what about a | pipe?",
                    must_contain=["damping constant k defaults to 60"],
                )
            ]
        )
        markdown = markdown_report([EvalRunner(eval_kb).run(payload)])
        assert "\\|" in markdown

    def test_json_is_parseable(self, eval_kb: KnowledgeBase, golden: GoldenSet) -> None:
        run = EvalRunner(eval_kb).run(golden)
        payload = json.loads(json_report([run]))
        assert payload["runs"][0]["n_scored"] == 3
        assert "ndcg@5" in payload["runs"][0]["metrics"]

    def test_json_can_include_per_query_detail(
        self, eval_kb: KnowledgeBase, golden: GoldenSet
    ) -> None:
        run = EvalRunner(eval_kb).run(golden)
        payload = json.loads(json_report([run], include_queries=True))
        assert len(payload["runs"][0]["queries"]) == 3

    def test_metric_chart_is_valid_svg(self, eval_kb: KnowledgeBase, golden: GoldenSet) -> None:
        import xml.dom.minidom

        runs = EvalRunner(eval_kb).compare(
            golden, {"a": {"strategy": "lexical"}, "b": {"strategy": "hybrid"}}
        )
        svg = metric_bar_chart(runs)
        xml.dom.minidom.parseString(svg)
        assert "<svg" in svg
        assert "ndcg@5" in svg

    def test_rank_chart_is_valid_svg(self, eval_kb: KnowledgeBase, golden: GoldenSet) -> None:
        import xml.dom.minidom

        svg = rank_distribution_chart(EvalRunner(eval_kb).run(golden))
        xml.dom.minidom.parseString(svg)

    def test_charts_escape_xml_in_labels(self, eval_kb: KnowledgeBase, golden: GoldenSet) -> None:
        import xml.dom.minidom

        run = EvalRunner(eval_kb).run(golden, label="a & b <c>")
        svg = metric_bar_chart([run])
        xml.dom.minidom.parseString(svg)
        assert "&amp;" in svg

    def test_charts_handle_no_data(self) -> None:
        import xml.dom.minidom

        xml.dom.minidom.parseString(metric_bar_chart([]))

    def test_write_report_produces_every_artifact(
        self, eval_kb: KnowledgeBase, golden: GoldenSet, tmp_path: Path
    ) -> None:
        run = EvalRunner(eval_kb).run(golden)
        written = write_report([run], tmp_path / "out")
        assert set(written) == {"markdown", "json", "metrics_chart", "rank_chart"}
        for path in written.values():
            assert path.is_file()
            assert path.stat().st_size > 0

    def test_chart_title_names_the_golden_set(
        self, eval_kb: KnowledgeBase, golden: GoldenSet, tmp_path: Path
    ) -> None:
        """A committed chart is read far from the command that produced it."""
        run = EvalRunner(eval_kb).run(golden)
        written = write_report([run], tmp_path / "out")
        assert "test" in written["metrics_chart"].read_text()
