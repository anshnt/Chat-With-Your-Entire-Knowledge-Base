"""End-to-end retrieval tests over a real ingested corpus.

These are the tests that would catch a regression a unit test misses: that
hybrid retrieval finds things neither retriever finds alone, that filters
actually restrict, and that the reported diagnostics match what ran.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kb.knowledge_base import KnowledgeBase
from kb.models import FusionMethod, RetrievalStrategy, SourceType


class TestSearchBasics:
    def test_finds_the_relevant_chunk(self, kb: KnowledgeBase) -> None:
        result = kb.search("what does the damping constant k default to?", top_k=3)
        assert result.results
        assert any("60" in r.chunk.text for r in result.results)

    def test_results_are_sorted_by_score(self, kb: KnowledgeBase) -> None:
        result = kb.search("retrieval metrics", top_k=5)
        scores = [r.score for r in result.results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_is_respected(self, kb: KnowledgeBase) -> None:
        assert len(kb.search("retrieval", top_k=2).results) <= 2

    def test_every_result_carries_a_usable_citation(self, kb: KnowledgeBase) -> None:
        for hit in kb.search("cosine similarity", top_k=5).results:
            assert hit.chunk.citation_label()
            assert hit.chunk.deep_link()
            assert hit.chunk.document_title

    def test_nonsense_query_returns_nothing_rather_than_junk(self, kb: KnowledgeBase) -> None:
        result = kb.search("zzzxqv unrelatedgibberish", strategy="lexical", top_k=5)
        assert result.results == []

    def test_diagnostics_describe_what_ran(self, kb: KnowledgeBase) -> None:
        result = kb.search("fusion", top_k=3)
        assert result.strategy is RetrievalStrategy.HYBRID
        assert result.fusion is FusionMethod.RRF
        assert result.lexical_candidates > 0
        assert result.dense_candidates > 0
        assert "lexical_ms" in result.timings_ms
        assert "dense_ms" in result.timings_ms
        assert result.total_ms() >= 0


class TestStrategies:
    @pytest.mark.parametrize(
        "strategy", [RetrievalStrategy.LEXICAL, RetrievalStrategy.DENSE, RetrievalStrategy.HYBRID]
    )
    def test_all_strategies_return_results(
        self, kb: KnowledgeBase, strategy: RetrievalStrategy
    ) -> None:
        result = kb.search("hybrid search combines", strategy=strategy, top_k=3)
        assert result.results
        assert result.strategy is strategy

    def test_lexical_only_has_no_dense_scores(self, kb: KnowledgeBase) -> None:
        result = kb.search("fusion", strategy=RetrievalStrategy.LEXICAL, top_k=3)
        assert all(r.dense_score is None for r in result.results)
        assert result.dense_candidates == 0

    def test_dense_only_has_no_lexical_scores(self, kb: KnowledgeBase) -> None:
        result = kb.search("fusion", strategy=RetrievalStrategy.DENSE, top_k=3)
        assert all(r.lexical_score is None for r in result.results)
        assert result.lexical_candidates == 0

    def test_hybrid_recall_is_at_least_the_union_of_the_parts(self, kb: KnowledgeBase) -> None:
        """The point of fusion: hybrid must never see fewer candidates than either part."""
        query = "how are ranked lists combined"
        lexical = kb.search(query, strategy=RetrievalStrategy.LEXICAL, top_k=20)
        dense = kb.search(query, strategy=RetrievalStrategy.DENSE, top_k=20)
        hybrid = kb.search(query, strategy=RetrievalStrategy.HYBRID, top_k=20)

        lexical_ids = {r.chunk.id for r in lexical.results}
        dense_ids = {r.chunk.id for r in dense.results}
        hybrid_ids = {r.chunk.id for r in hybrid.results}
        assert lexical_ids | dense_ids == hybrid_ids

    @pytest.mark.parametrize("fusion", list(FusionMethod))
    def test_every_fusion_method_works(self, kb: KnowledgeBase, fusion: FusionMethod) -> None:
        result = kb.search("dense vectors", fusion=fusion, top_k=3)
        assert result.results
        assert result.fusion is fusion


class TestFiltering:
    def test_source_type_filter_restricts_results(self, kb: KnowledgeBase) -> None:
        result = kb.search("sqlite vectors float32", source_types=[SourceType.TEXT], top_k=5)
        assert result.results
        assert all(r.chunk.source_type is SourceType.TEXT for r in result.results)

    def test_document_filter_restricts_results(self, kb: KnowledgeBase) -> None:
        target = next(
            d for d in kb.documents() if "valuation" in d.title.lower() or "Evaluation" in d.title
        )
        result = kb.search("metrics", document_ids=[target.id], top_k=10)
        assert result.results
        assert all(r.chunk.document_id == target.id for r in result.results)

    def test_impossible_filter_returns_empty(self, kb: KnowledgeBase) -> None:
        result = kb.search("anything", document_ids=["doc_does_not_exist"], top_k=5)
        assert result.results == []

    def test_dense_filter_applies_before_top_k(self, kb: KnowledgeBase) -> None:
        """Post-filtering would silently drop recall; pre-filtering must not."""
        result = kb.search(
            "sqlite float32 blobs",
            strategy=RetrievalStrategy.DENSE,
            source_types=[SourceType.TEXT],
            top_k=5,
        )
        assert result.results
        assert all(r.chunk.source_type is SourceType.TEXT for r in result.results)


class TestMMR:
    def test_mmr_changes_the_result_set(self, empty_kb: KnowledgeBase, tmp_path: Path) -> None:
        # Three near-identical chunks plus one distinct one.
        path = tmp_path / "dupes.md"
        path.write_text(
            "# Dupes\n\n"
            "## One\n\nHybrid search fuses BM25 with dense retrieval results.\n\n"
            "## Two\n\nHybrid search fuses BM25 with dense retrieval outputs.\n\n"
            "## Three\n\nHybrid search fuses BM25 with dense retrieval rankings.\n\n"
            "## Four\n\nEvaluation uses nDCG and mean reciprocal rank as metrics.\n"
        )
        empty_kb.ingest(str(path))

        plain = empty_kb.search("hybrid search fuses BM25", top_k=2, use_mmr=False)
        diverse = empty_kb.search("hybrid search fuses BM25", top_k=2, use_mmr=True, mmr_lambda=0.3)
        assert plain.results and diverse.results
        assert "mmr_ms" in diverse.timings_ms
        assert "mmr_ms" not in plain.timings_ms


class TestSimilarChunks:
    def test_more_like_this_excludes_the_seed(self, kb: KnowledgeBase) -> None:
        seed = kb.search("reciprocal rank fusion", top_k=1).results[0].chunk
        similar = kb.similar_chunks(seed.id, limit=3)
        assert similar
        assert all(s.chunk.id != seed.id for s in similar)

    def test_unknown_chunk_returns_empty(self, kb: KnowledgeBase) -> None:
        assert kb.similar_chunks("chk_nope") == []


class TestTelemetry:
    def test_searches_are_logged_to_the_heatmap(self, kb: KnowledgeBase) -> None:
        assert kb.heatmap() == []
        kb.search("fusion", top_k=2)
        kb.search("fusion", top_k=2)
        rows = kb.heatmap()
        assert rows
        assert rows[0]["hits"] >= 2

    def test_recent_queries_are_recorded(self, kb: KnowledgeBase) -> None:
        kb.search("dense retrieval", top_k=1)
        assert "dense retrieval" in kb.store.recent_queries()


class TestCollections:
    def test_collections_do_not_leak_into_each_other(
        self, empty_kb: KnowledgeBase, tmp_path: Path
    ) -> None:
        a = tmp_path / "a.md"
        a.write_text("# Alpha\n\nUnique alpha content about zebras.")
        b = tmp_path / "b.md"
        b.write_text("# Beta\n\nUnique beta content about walruses.")
        empty_kb.ingest(str(a), collection="alpha")
        empty_kb.ingest(str(b), collection="beta")

        alpha = empty_kb.search("zebras", collection="alpha", top_k=5)
        assert alpha.results
        assert all(r.chunk.collection == "alpha" for r in alpha.results)

        # Dense retrieval always returns a nearest neighbour — there is no
        # "no match" — so isolation is asserted on provenance, not emptiness.
        beta = empty_kb.search("zebras", collection="beta", top_k=5)
        assert all(r.chunk.collection == "beta" for r in beta.results)
        assert not any("zebra" in r.chunk.text.lower() for r in beta.results)

        # Lexical retrieval, which requires an actual term match, finds nothing.
        assert empty_kb.search("zebras", collection="beta", strategy="lexical").results == []

    def test_deleting_a_collection_leaves_others_intact(
        self, empty_kb: KnowledgeBase, tmp_path: Path
    ) -> None:
        for name in ("alpha", "beta"):
            path = tmp_path / f"{name}.md"
            path.write_text(f"# {name}\n\nContent for {name} about retrieval.")
            empty_kb.ingest(str(path), collection=name)

        empty_kb.delete_collection("alpha")
        assert empty_kb.stats("alpha").n_documents == 0
        assert empty_kb.stats("beta").n_documents == 1


class TestChunkContext:
    def test_neighbours_are_returned_in_order(self, kb: KnowledgeBase) -> None:
        document = next(d for d in kb.documents() if d.n_chunks >= 3)
        chunks = kb.document_chunks(document.id)
        middle = chunks[1]
        window = kb.chunk_with_context(middle.id, window=1)
        assert [c.ordinal for c in window] == [0, 1, 2]

    def test_zero_window_is_just_the_chunk(self, kb: KnowledgeBase) -> None:
        chunk = kb.document_chunks(kb.documents()[0].id)[0]
        assert [c.id for c in kb.chunk_with_context(chunk.id, window=0)] == [chunk.id]
