"""Store tests: BM25 behaviour, vector round-trips, cascades, and telemetry."""

from __future__ import annotations

import numpy as np
import pytest

from kb.errors import NotFoundError
from kb.models import Chunk, Document, SourceType, TextLocator
from kb.store import SQLiteStore, blob_to_vector, vector_to_blob


def make_document(collection: str = "default", **kwargs) -> Document:
    payload = {
        "collection": collection,
        "source_type": SourceType.MARKDOWN,
        "title": "Retrieval",
        "uri": "/tmp/retrieval.md",
        "content_hash": "hash-1",
    }
    payload.update(kwargs)
    return Document(**payload)


def make_chunks(document: Document, texts: list[str]) -> list[Chunk]:
    return [
        Chunk(
            document_id=document.id,
            collection=document.collection,
            ordinal=i,
            text=text,
            locator=TextLocator(line_start=i + 1, line_end=i + 1),
            document_title=document.title,
            source_type=document.source_type,
        )
        for i, text in enumerate(texts)
    ]


class TestVectorSerialisation:
    def test_round_trip_preserves_values(self) -> None:
        vector = np.array([0.1, -0.5, 0.25], dtype="float32")
        assert np.allclose(blob_to_vector(vector_to_blob(vector)), vector)

    def test_accepts_a_plain_list(self) -> None:
        assert blob_to_vector(vector_to_blob([1.0, 2.0])).tolist() == [1.0, 2.0]


class TestDocuments:
    def test_add_and_get(self, store: SQLiteStore) -> None:
        document = make_document()
        chunks = make_chunks(document, ["alpha", "beta"])
        stored = store.add_document(document, chunks)
        assert stored.n_chunks == 2
        assert store.get_document(document.id).title == "Retrieval"
        assert store.count_chunks() == 2

    def test_missing_document_raises(self, store: SQLiteStore) -> None:
        with pytest.raises(NotFoundError):
            store.get_document("doc_missing")

    def test_find_by_hash(self, store: SQLiteStore) -> None:
        document = make_document()
        store.add_document(document, make_chunks(document, ["alpha"]))
        assert store.find_document_by_hash("default", "hash-1") is not None
        assert store.find_document_by_hash("default", "nope") is None
        assert store.find_document_by_hash("other", "hash-1") is None

    def test_delete_cascades_to_chunks_and_vectors(self, store: SQLiteStore) -> None:
        document = make_document()
        chunks = make_chunks(document, ["alpha", "beta"])
        store.add_document(document, chunks)
        store.upsert_embeddings(
            [(c.id, np.ones(4, dtype="float32")) for c in chunks],
            collection="default",
            model="m",
        )
        assert store.collection_stats().n_embedded == 2

        store.delete_document(document.id)
        assert store.count_chunks() == 0
        assert store.collection_stats().n_embedded == 0

    def test_list_documents_filters(self, store: SQLiteStore) -> None:
        md = make_document(title="Markdown doc", content_hash="h1")
        store.add_document(md, make_chunks(md, ["alpha"]))
        pdf = make_document(
            title="PDF doc", content_hash="h2", source_type=SourceType.PDF, uri="/a.pdf"
        )
        store.add_document(pdf, make_chunks(pdf, ["beta"]))

        assert len(store.list_documents()) == 2
        assert len(store.list_documents(source_type=SourceType.PDF)) == 1
        assert len(store.list_documents(search="Markdown")) == 1

    def test_collections_are_isolated(self, store: SQLiteStore) -> None:
        a = make_document("alpha", content_hash="h1")
        b = make_document("beta", content_hash="h2")
        store.add_document(a, make_chunks(a, ["one"]))
        store.add_document(b, make_chunks(b, ["two"]))
        assert store.count_documents("alpha") == 1
        assert store.count_documents("beta") == 1
        assert set(store.list_collections()) >= {"alpha", "beta"}

        removed = store.delete_collection("alpha")
        assert removed == 1
        assert store.count_documents("alpha") == 0
        assert store.count_documents("beta") == 1


class TestLexicalSearch:
    @pytest.fixture
    def populated(self, store: SQLiteStore) -> SQLiteStore:
        document = make_document()
        store.add_document(
            document,
            make_chunks(
                document,
                [
                    "Reciprocal rank fusion combines ranked lists using ranks.",
                    "Dense retrieval uses cosine similarity over normalised vectors.",
                    "BM25 is a lexical ranking function based on term frequency.",
                ],
            ),
        )
        return store

    def test_finds_matching_chunks(self, populated: SQLiteStore) -> None:
        hits = populated.search_lexical('"fusion"', limit=5)
        assert len(hits) == 1

    def test_scores_are_higher_is_better(self, populated: SQLiteStore) -> None:
        hits = populated.search_lexical('"retrieval" OR "cosine"', limit=5)
        scores = [score for _, score in hits]
        assert scores == sorted(scores, reverse=True)

    def test_stemming_matches_word_variants(self, populated: SQLiteStore) -> None:
        # The porter tokenizer means "ranking" reaches "ranked" and "ranks".
        assert populated.search_lexical('"ranking"', limit=5)

    def test_no_match_returns_empty(self, populated: SQLiteStore) -> None:
        assert populated.search_lexical('"kangaroo"', limit=5) == []

    def test_fts_index_is_updated_on_delete(self, populated: SQLiteStore) -> None:
        document = populated.list_documents()[0]
        populated.delete_document(document.id)
        assert populated.search_lexical('"fusion"', limit=5) == []


class TestEmbeddings:
    def test_vector_matrix_is_normalised(self, store: SQLiteStore) -> None:
        document = make_document()
        chunks = make_chunks(document, ["alpha", "beta"])
        store.add_document(document, chunks)
        store.upsert_embeddings(
            [(chunks[0].id, np.array([3.0, 4.0])), (chunks[1].id, np.array([0.0, 2.0]))],
            collection="default",
            model="m",
        )
        matrix, ids = store.vector_matrix("default", model="m")
        assert matrix.shape == (2, 2)
        assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
        assert set(ids) == {chunks[0].id, chunks[1].id}

    def test_matrix_cache_invalidates_on_write(self, store: SQLiteStore) -> None:
        document = make_document()
        chunks = make_chunks(document, ["alpha", "beta"])
        store.add_document(document, chunks)
        store.upsert_embeddings([(chunks[0].id, np.ones(3))], collection="default", model="m")
        first, _ = store.vector_matrix("default", model="m")
        assert first.shape[0] == 1

        store.upsert_embeddings([(chunks[1].id, np.ones(3))], collection="default", model="m")
        second, _ = store.vector_matrix("default", model="m")
        assert second.shape[0] == 2

    def test_unembedded_ids_tracks_per_model(self, store: SQLiteStore) -> None:
        document = make_document()
        chunks = make_chunks(document, ["alpha", "beta"])
        store.add_document(document, chunks)
        assert len(store.unembedded_chunk_ids(model="m")) == 2

        store.upsert_embeddings([(chunks[0].id, np.ones(3))], collection="default", model="m")
        assert len(store.unembedded_chunk_ids(model="m")) == 1
        # A different model has nothing embedded at all.
        assert len(store.unembedded_chunk_ids(model="other")) == 2

    def test_upsert_replaces_rather_than_duplicates(self, store: SQLiteStore) -> None:
        document = make_document()
        chunks = make_chunks(document, ["alpha"])
        store.add_document(document, chunks)
        store.upsert_embeddings([(chunks[0].id, np.ones(3))], collection="default", model="m")
        store.upsert_embeddings([(chunks[0].id, np.zeros(3))], collection="default", model="m")
        assert store.collection_stats().n_embedded == 1

    def test_empty_matrix_for_unknown_model(self, store: SQLiteStore) -> None:
        matrix, ids = store.vector_matrix("default", model="nope")
        assert matrix.size == 0
        assert ids == []


class TestChunkAccess:
    def test_get_chunks_preserves_requested_order(self, store: SQLiteStore) -> None:
        document = make_document()
        chunks = make_chunks(document, ["a", "b", "c"])
        store.add_document(document, chunks)
        wanted = [chunks[2].id, chunks[0].id, chunks[1].id]
        assert [c.id for c in store.get_chunks(wanted)] == wanted

    def test_get_chunks_skips_unknown_ids(self, store: SQLiteStore) -> None:
        document = make_document()
        chunks = make_chunks(document, ["a"])
        store.add_document(document, chunks)
        assert [c.id for c in store.get_chunks([chunks[0].id, "chk_nope"])] == [chunks[0].id]

    def test_neighbours_window(self, store: SQLiteStore) -> None:
        document = make_document()
        chunks = make_chunks(document, ["a", "b", "c", "d", "e"])
        store.add_document(document, chunks)
        window = store.chunk_neighbours(chunks[2].id, window=1)
        assert [c.ordinal for c in window] == [1, 2, 3]

    def test_neighbours_clamps_at_document_edges(self, store: SQLiteStore) -> None:
        document = make_document()
        chunks = make_chunks(document, ["a", "b"])
        store.add_document(document, chunks)
        assert [c.ordinal for c in store.chunk_neighbours(chunks[0].id, window=3)] == [0, 1]

    def test_iter_chunks_batches(self, store: SQLiteStore) -> None:
        document = make_document()
        chunks = make_chunks(document, [f"chunk {i}" for i in range(7)])
        store.add_document(document, chunks)
        batches = list(store.iter_chunks(batch_size=3))
        assert [len(b) for b in batches] == [3, 3, 1]


class TestRetrievalTelemetry:
    def test_heatmap_counts_hits(self, store: SQLiteStore) -> None:
        document = make_document()
        chunks = make_chunks(document, ["a", "b"])
        store.add_document(document, chunks)
        store.log_retrieval("q1", [(chunks[0].id, 0.9), (chunks[1].id, 0.5)])
        store.log_retrieval("q2", [(chunks[0].id, 0.8)])

        rows = store.retrieval_heatmap()
        top = rows[0]
        assert top["chunk_id"] == chunks[0].id
        assert top["hits"] == 2

    def test_recent_queries_are_deduplicated(self, store: SQLiteStore) -> None:
        document = make_document()
        chunks = make_chunks(document, ["a"])
        store.add_document(document, chunks)
        store.log_retrieval("same", [(chunks[0].id, 1.0)])
        store.log_retrieval("same", [(chunks[0].id, 1.0)])
        store.log_retrieval("other", [(chunks[0].id, 1.0)])
        assert sorted(store.recent_queries()) == ["other", "same"]

    def test_logging_nothing_is_a_noop(self, store: SQLiteStore) -> None:
        store.log_retrieval("q", [])
        assert store.retrieval_heatmap() == []
