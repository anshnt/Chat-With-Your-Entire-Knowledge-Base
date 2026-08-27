"""Visualisation tests.

The properties worth pinning are determinism (a map whose points move on every
reload cannot be compared across runs), graceful degradation to PCA when the
optional projection libraries are absent, and that cluster labels are *contrast*
terms rather than the most frequent words.
"""

from __future__ import annotations

import xml.dom.minidom
from pathlib import Path

import numpy as np
import pytest

from kb.config import Settings
from kb.knowledge_base import KnowledgeBase
from kb.viz import (
    CorpusMapBuilder,
    cluster_colour,
    cluster_corpus,
    distinctive_terms,
    document_graph,
    kmeans,
    normalize_to_unit_square,
    project,
    render_corpus_map,
    suggest_k,
)
from kb.viz.clustering import _term_counts
from kb.viz.projection import _pca


def blobs(n_per: int = 30, dim: int = 8, separation: float = 6.0) -> np.ndarray:
    """Three well-separated gaussian blobs, deterministically generated."""
    rng = np.random.default_rng(7)
    centres = np.zeros((3, dim))
    centres[0, 0] = separation
    centres[1, 1] = separation
    centres[2, 2] = separation
    return np.vstack([c + rng.normal(scale=0.35, size=(n_per, dim)) for c in centres])


# --------------------------------------------------------------------------- #
# projection
# --------------------------------------------------------------------------- #


class TestPCA:
    def test_shape(self) -> None:
        coordinates, explained = _pca(blobs())
        assert coordinates.shape == (90, 2)
        assert 0.0 <= explained <= 1.0

    def test_separated_blobs_stay_separated(self) -> None:
        """The projection must preserve the structure it exists to show."""
        matrix = blobs(n_per=25)
        coordinates, _ = _pca(matrix)
        groups = [coordinates[0:25], coordinates[25:50], coordinates[50:75]]
        centres = [g.mean(axis=0) for g in groups]
        within = max(
            float(np.linalg.norm(g - c, axis=1).mean())
            for g, c in zip(groups, centres, strict=True)
        )
        between = min(
            float(np.linalg.norm(centres[i] - centres[j]))
            for i in range(3)
            for j in range(i + 1, 3)
        )
        assert between > within * 2

    def test_sign_convention_is_fixed(self) -> None:
        """Eigenvector signs are arbitrary; an unfixed convention mirrors the
        plot at random and makes maps incomparable across runs."""
        matrix = blobs()
        first, _ = _pca(matrix.copy())
        second, _ = _pca(matrix.copy())
        assert np.allclose(first, second)

    def test_explained_variance_is_high_for_low_rank_data(self) -> None:
        rng = np.random.default_rng(3)
        basis = rng.normal(size=(2, 10))
        matrix = rng.normal(size=(60, 2)) @ basis
        _, explained = _pca(matrix)
        assert explained > 0.95

    def test_rank_one_input_is_padded_to_two_columns(self) -> None:
        matrix = np.arange(20, dtype="float64").reshape(20, 1)
        coordinates, _ = _pca(matrix)
        assert coordinates.shape == (20, 2)


class TestProject:
    def test_falls_back_to_pca_when_nothing_is_installed(self) -> None:
        """A map that exists beats a perfect map that does not."""
        result = project(blobs(), method="auto")
        assert result.method in ("umap", "tsne", "pca")
        assert result.coordinates.shape == (90, 2)

    def test_explicit_pca(self) -> None:
        result = project(blobs(), method="pca")
        assert result.method == "pca"
        assert result.explained_variance is not None

    def test_unavailable_method_is_noted_not_hidden(self) -> None:
        result = project(blobs(), method="auto")
        if result.method == "pca":
            assert result.notes, "a fallback must say why"

    def test_deterministic(self) -> None:
        first = project(blobs(), method="pca").coordinates
        second = project(blobs(), method="pca").coordinates
        assert np.allclose(first, second)

    def test_empty_corpus(self) -> None:
        result = project(np.zeros((0, 5)))
        assert result.coordinates.shape == (0, 2)
        assert result.notes

    @pytest.mark.parametrize("n", [1, 2])
    def test_tiny_corpus_does_not_crash(self, n: int) -> None:
        result = project(np.random.default_rng(1).normal(size=(n, 6)))
        assert result.coordinates.shape == (n, 2)

    def test_non_2d_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="2D matrix"):
            project(np.zeros((3, 4, 5)))


class TestNormalizeToUnitSquare:
    def test_maps_into_the_unit_square(self) -> None:
        coordinates = normalize_to_unit_square(blobs()[:, :2] * 100)
        assert coordinates.min() >= -1e-9
        assert coordinates.max() <= 1.0 + 1e-9

    def test_aspect_ratio_is_preserved(self) -> None:
        """Per-axis scaling would distort exactly the distances the map shows."""
        coordinates = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 1.0]])
        scaled = normalize_to_unit_square(coordinates)
        x_span = scaled[:, 0].max() - scaled[:, 0].min()
        y_span = scaled[:, 1].max() - scaled[:, 1].min()
        assert x_span == pytest.approx(1.0)
        assert y_span == pytest.approx(0.1)

    def test_identical_points_land_in_the_centre(self) -> None:
        coordinates = normalize_to_unit_square(np.full((5, 2), 3.0))
        assert np.allclose(coordinates, 0.5)

    def test_empty(self) -> None:
        assert normalize_to_unit_square(np.zeros((0, 2))).size == 0


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #


class TestKMeans:
    def test_recovers_separated_blobs(self) -> None:
        matrix = blobs(n_per=30)
        labels, centroids = kmeans(matrix, 3)
        assert centroids.shape == (3, matrix.shape[1])
        for start in (0, 30, 60):
            group = labels[start : start + 30]
            # Every member of a blob should land in the same cluster.
            assert len(set(group.tolist())) == 1

    def test_deterministic(self) -> None:
        matrix = blobs()
        first, _ = kmeans(matrix, 4)
        second, _ = kmeans(matrix, 4)
        assert np.array_equal(first, second)

    def test_no_empty_clusters_and_no_nan_centroids(self) -> None:
        """An empty cluster would produce NaN centroids and poison the map."""
        matrix = blobs(n_per=5)
        _, centroids = kmeans(matrix, 8)
        assert np.isfinite(centroids).all()

    def test_k_of_one(self) -> None:
        labels, centroids = kmeans(blobs(), 1)
        assert set(labels.tolist()) == {0}
        assert centroids.shape[0] == 1

    def test_k_above_sample_count(self) -> None:
        matrix = blobs(n_per=1)
        labels, _ = kmeans(matrix, 50)
        assert len(labels) == matrix.shape[0]


class TestSuggestK:
    @pytest.mark.parametrize(("n", "expected"), [(0, 1), (5, 1), (8, 2), (200, 10)])
    def test_scales_with_corpus_size(self, n: int, expected: int) -> None:
        assert suggest_k(n) == expected

    def test_is_clamped(self) -> None:
        assert suggest_k(100_000) <= 12


class TestDistinctiveTerms:
    def test_prefers_contrast_over_frequency(self) -> None:
        """Most-frequent terms label every cluster "the, and, retrieval"."""
        cluster = [
            "reciprocal rank fusion combines ranked lists",
            "fusion uses ranks rather than raw scores",
        ]
        others = [
            "cross encoder reranking scores pairs jointly",
            "pdf pages are extracted and de-hyphenated",
            "notion exports carry a page id in the filename",
        ]
        corpus = _term_counts(cluster + others)
        terms = distinctive_terms(cluster, corpus, limit=3)
        assert any("fusion" in t or "rank" in t for t in terms)
        assert "reranking" not in terms
        assert "notion" not in terms

    def test_a_term_in_one_chunk_only_does_not_win(self) -> None:
        """A single verbose passage must not name the whole region."""
        cluster = [
            "zzzunique zzzunique zzzunique zzzunique zzzunique",
            "fusion combines ranked lists using ranks",
            "fusion uses ranks not scores",
            "fusion damping constant defaults",
        ]
        corpus = _term_counts(cluster)
        terms = distinctive_terms(cluster, corpus, limit=3)
        assert "zzzunique" not in terms

    def test_stopwords_never_appear(self) -> None:
        cluster = ["the and with this that from which"] * 3
        terms = distinctive_terms(cluster, _term_counts(cluster), limit=5)
        assert all(t not in {"the", "and", "with", "this", "that"} for t in terms)

    def test_empty_cluster(self) -> None:
        assert distinctive_terms([], _term_counts(["anything"])) == []


class TestClusterCorpus:
    def test_clusters_are_labelled_and_sorted_by_size(self) -> None:
        matrix = blobs(n_per=20)
        texts = (
            ["reciprocal rank fusion combines ranked lists using ranks"] * 20
            + ["cross encoder reranking scores each query document pair jointly"] * 20
            + ["pdf pages are extracted de-hyphenated and stripped of headers"] * 20
        )
        clusters = cluster_corpus(matrix, texts, k=3)
        assert len(clusters) == 3
        assert [c.size for c in clusters] == sorted((c.size for c in clusters), reverse=True)
        for cluster in clusters:
            assert cluster.label
            assert cluster.terms
            assert 0.0 <= cluster.coherence <= 1.0

    def test_coherence_is_higher_for_tight_clusters(self) -> None:
        tight = blobs(n_per=20, separation=20.0)
        loose = np.random.default_rng(5).normal(size=(60, 8))
        texts = ["some text about retrieval and ranking"] * 60
        tight_coherence = np.mean([c.coherence for c in cluster_corpus(tight, texts, k=3)])
        loose_coherence = np.mean([c.coherence for c in cluster_corpus(loose, texts, k=3)])
        assert tight_coherence > loose_coherence

    def test_every_chunk_is_assigned(self) -> None:
        matrix = blobs(n_per=10)
        texts = ["text about retrieval"] * matrix.shape[0]
        clusters = cluster_corpus(matrix, texts, k=3)
        assigned = sorted(i for c in clusters for i in c.member_indices)
        assert assigned == list(range(matrix.shape[0]))

    def test_empty_corpus(self) -> None:
        assert cluster_corpus(np.zeros((0, 4)), []) == []


def test_cluster_colour_is_stable_and_wraps() -> None:
    assert cluster_colour(0) == cluster_colour(0)
    assert cluster_colour(0) == cluster_colour(12)
    assert cluster_colour(0) != cluster_colour(1)


# --------------------------------------------------------------------------- #
# corpus map, end to end
# --------------------------------------------------------------------------- #


@pytest.fixture
def viz_kb(tmp_path: Path) -> KnowledgeBase:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "retrieval.md").write_text(
        "# Retrieval\n\n## Hybrid search\n\nHybrid search combines BM25 lexical matching "
        "with dense vector retrieval over the same corpus.\n\n"
        "## Fusion\n\nReciprocal Rank Fusion combines ranked lists using ranks rather "
        "than raw scores. The damping constant defaults to 60.\n\n"
        "## Reranking\n\nA cross-encoder reranker scores each query-document pair "
        "jointly and is more accurate than a bi-encoder.\n"
    )
    (docs / "storage.md").write_text(
        "# Storage\n\n## SQLite\n\nThe store keeps documents, chunks, the full text index "
        "and the dense vectors in a single SQLite file.\n\n"
        "## Vectors\n\nVectors are little-endian float32 blobs stored alongside their "
        "precomputed norm for fast cosine similarity.\n"
    )
    (docs / "connectors.md").write_text(
        "# Connectors\n\n## PDF\n\nPDF pages are extracted, de-hyphenated, and stripped "
        "of running headers detected by frequency across pages.\n\n"
        "## Notion\n\nNotion exports carry a thirty-two character page id in every "
        "filename, which becomes the citation link.\n"
    )
    settings = Settings(
        data_dir=tmp_path / "data", embedding_dim=384, chunk_size=500, min_chunk_size=40
    )
    instance = KnowledgeBase(settings)
    report = instance.ingest(str(docs))
    assert report.chunks_created >= 6, report.errors
    return instance


class TestCorpusMapBuilder:
    def test_builds_a_map(self, viz_kb: KnowledgeBase) -> None:
        result = CorpusMapBuilder(viz_kb).build()
        assert result.n_chunks > 0
        assert result.n_plotted == result.n_chunks
        assert len(result.points) == result.n_plotted
        assert result.clusters
        assert result.method in ("umap", "tsne", "pca")

    def test_coordinates_are_normalised(self, viz_kb: KnowledgeBase) -> None:
        for point in CorpusMapBuilder(viz_kb).build().points:
            assert 0.0 <= point.x <= 1.0
            assert 0.0 <= point.y <= 1.0

    def test_points_carry_display_metadata(self, viz_kb: KnowledgeBase) -> None:
        for point in CorpusMapBuilder(viz_kb).build().points:
            assert point.chunk_id
            assert point.document_title
            assert point.snippet
            assert point.tokens > 0

    def test_retrieval_counts_are_attached(self, viz_kb: KnowledgeBase) -> None:
        """A cluster nothing ever reaches is the actionable finding."""
        builder = CorpusMapBuilder(viz_kb)
        assert builder.build().coverage() == 0.0

        viz_kb.search("reciprocal rank fusion damping constant", top_k=3)
        result = builder.build()
        assert result.coverage() > 0.0
        assert any(point.retrievals > 0 for point in result.points)

    def test_cache_is_invalidated_by_a_write(self, viz_kb: KnowledgeBase) -> None:
        builder = CorpusMapBuilder(viz_kb)
        first = builder.build()
        assert builder.build() is first, "unchanged corpus should hit the cache"

        viz_kb.ingest_text("# New\n\nA new document about evaluation metrics.", title="New")
        second = builder.build()
        assert second is not first
        assert second.n_chunks > first.n_chunks

    def test_sampling_is_reported_not_silent(self, viz_kb: KnowledgeBase) -> None:
        result = CorpusMapBuilder(viz_kb).build(max_points=3)
        assert result.sampled
        assert result.n_plotted <= 3
        assert any("sampled" in note for note in result.notes)

    def test_sampling_is_stable_between_calls(self, viz_kb: KnowledgeBase) -> None:
        """Even spacing, not random: the map must not reshuffle on reload."""
        builder = CorpusMapBuilder(viz_kb)
        first = [p.chunk_id for p in builder.build(max_points=4).points]
        builder._cache.clear()
        second = [p.chunk_id for p in builder.build(max_points=4).points]
        assert first == second

    def test_explicit_cluster_count(self, viz_kb: KnowledgeBase) -> None:
        result = CorpusMapBuilder(viz_kb).build(k=2)
        assert len(result.clusters) <= 2

    def test_empty_collection_is_explained(self, viz_kb: KnowledgeBase) -> None:
        result = CorpusMapBuilder(viz_kb).build("nonexistent")
        assert result.points == []
        assert result.n_chunks == 0
        assert any("no embedded chunks" in note for note in result.notes)


class TestDocumentGraph:
    def test_nodes_cover_the_documents(self, viz_kb: KnowledgeBase) -> None:
        graph = document_graph(viz_kb)
        assert len(graph["nodes"]) == 3
        assert all(node["title"] for node in graph["nodes"])
        assert all(node["n_chunks"] > 0 for node in graph["nodes"])

    def test_edges_are_symmetric_and_deduplicated(self, viz_kb: KnowledgeBase) -> None:
        graph = document_graph(viz_kb, min_similarity=0.0)
        pairs = [tuple(sorted((e["source"], e["target"]))) for e in graph["edges"]]
        assert len(pairs) == len(set(pairs))

    def test_no_self_edges(self, viz_kb: KnowledgeBase) -> None:
        graph = document_graph(viz_kb, min_similarity=0.0)
        assert all(e["source"] != e["target"] for e in graph["edges"])

    def test_edges_are_sorted_by_weight(self, viz_kb: KnowledgeBase) -> None:
        graph = document_graph(viz_kb, min_similarity=0.0)
        weights = [e["weight"] for e in graph["edges"]]
        assert weights == sorted(weights, reverse=True)

    def test_a_high_threshold_prunes_everything(self, viz_kb: KnowledgeBase) -> None:
        graph = document_graph(viz_kb, min_similarity=0.999)
        assert graph["edges"] == []
        assert graph["notes"]

    def test_edges_per_document_are_capped(self, viz_kb: KnowledgeBase) -> None:
        """A dense corpus otherwise renders as a solid disc."""
        graph = document_graph(viz_kb, min_similarity=0.0, max_edges_per_document=1)
        assert len(graph["edges"]) <= len(graph["nodes"])

    def test_empty_collection(self, viz_kb: KnowledgeBase) -> None:
        graph = document_graph(viz_kb, "nonexistent")
        assert graph["nodes"] == []
        assert graph["notes"]


class TestRenderCorpusMap:
    def test_output_is_valid_svg(self, viz_kb: KnowledgeBase) -> None:
        svg = render_corpus_map(CorpusMapBuilder(viz_kb).build())
        xml.dom.minidom.parseString(svg)
        assert "<svg" in svg
        assert "Corpus map" in svg

    def test_never_retrieved_points_are_hollow(self, viz_kb: KnowledgeBase) -> None:
        """Outline rather than colour, so it survives greyscale and colour-blindness."""
        svg = render_corpus_map(CorpusMapBuilder(viz_kb).build())
        assert 'fill="none"' in svg

    def test_retrieved_points_are_filled(self, viz_kb: KnowledgeBase) -> None:
        viz_kb.search("reciprocal rank fusion", top_k=3)
        builder = CorpusMapBuilder(viz_kb)
        svg = render_corpus_map(builder.build())
        assert 'fill-opacity="0.82"' in svg

    def test_subtitle_carries_the_caveats(self, viz_kb: KnowledgeBase) -> None:
        result = CorpusMapBuilder(viz_kb).build(method="pca")
        svg = render_corpus_map(result)
        assert "PCA" in svg
        assert "variance in 2D" in svg
        assert "ever retrieved" in svg

    def test_labels_are_xml_escaped(self, viz_kb: KnowledgeBase) -> None:
        result = CorpusMapBuilder(viz_kb).build()
        result.clusters[0]["label"] = 'a & b <c> "d"'
        xml.dom.minidom.parseString(render_corpus_map(result))

    def test_empty_map_renders_a_message(self, viz_kb: KnowledgeBase) -> None:
        result = CorpusMapBuilder(viz_kb).build("nonexistent")
        svg = render_corpus_map(result)
        xml.dom.minidom.parseString(svg)
        assert "no embedded chunks" in svg
