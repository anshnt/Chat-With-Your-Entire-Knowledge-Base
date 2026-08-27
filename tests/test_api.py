"""API tests via the ASGI test client."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kb.api.app import create_app
from kb.api.deps import set_knowledge_base
from kb.knowledge_base import KnowledgeBase


@pytest.fixture
def client(kb: KnowledgeBase) -> Iterator[TestClient]:
    """A client wired to the pre-populated knowledge base fixture."""
    set_knowledge_base(kb)
    app = create_app(kb.settings)
    with TestClient(app) as test_client:
        yield test_client
    set_knowledge_base(None)


@pytest.fixture
def empty_client(empty_kb: KnowledgeBase) -> Iterator[TestClient]:
    set_knowledge_base(empty_kb)
    app = create_app(empty_kb.settings)
    with TestClient(app) as test_client:
        yield test_client
    set_knowledge_base(None)


class TestHealth:
    def test_reports_the_active_configuration(self, client: TestClient) -> None:
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["embedding_model"] == "hashing-ngram-v1"
        assert body["embedding_dim"] == 256
        assert body["retrieval_strategy"] == "hybrid"
        assert "markdown" in body["connectors"]


class TestSearch:
    def test_returns_hits_with_citations_and_scores(self, client: TestClient) -> None:
        response = client.post("/api/search", json={"query": "reciprocal rank fusion", "top_k": 3})
        assert response.status_code == 200
        body = response.json()
        assert body["hits"]
        hit = body["hits"][0]
        assert hit["citation"]["chunk_id"]
        assert hit["citation"]["deep_link"]
        assert hit["citation"]["label"]
        assert hit["citation"]["locator"]["kind"] in {"text", "pdf"}
        assert hit["scores"]["final"] is not None
        assert set(hit["scores"]["retrievers"]) <= {"lexical", "dense"}

    def test_diagnostics_are_exposed(self, client: TestClient) -> None:
        body = client.post("/api/search", json={"query": "dense vectors"}).json()
        assert body["strategy"] == "hybrid"
        assert body["fusion"] == "rrf"
        assert body["lexical_candidates"] >= 0
        assert body["total_ms"] >= 0

    def test_get_form_works(self, client: TestClient) -> None:
        body = client.get("/api/search", params={"q": "metrics", "top_k": 2}).json()
        assert len(body["hits"]) <= 2

    def test_strategy_override(self, client: TestClient) -> None:
        body = client.post("/api/search", json={"query": "fusion", "strategy": "lexical"}).json()
        assert body["strategy"] == "lexical"
        assert body["dense_candidates"] == 0

    def test_source_type_filter(self, client: TestClient) -> None:
        body = client.post(
            "/api/search", json={"query": "sqlite float32", "source_types": ["text"]}
        ).json()
        assert body["hits"]
        assert all(h["citation"]["source_type"] == "text" for h in body["hits"])

    def test_blank_query_is_rejected(self, client: TestClient) -> None:
        assert client.post("/api/search", json={"query": ""}).status_code == 422

    def test_top_k_bounds_are_enforced(self, client: TestClient) -> None:
        assert client.post("/api/search", json={"query": "x", "top_k": 0}).status_code == 422
        assert client.post("/api/search", json={"query": "x", "top_k": 9999}).status_code == 422

    def test_similar_chunks(self, client: TestClient) -> None:
        seed = client.post("/api/search", json={"query": "fusion", "top_k": 1}).json()
        chunk_id = seed["hits"][0]["citation"]["chunk_id"]
        body = client.get(f"/api/chunks/{chunk_id}/similar", params={"limit": 2}).json()
        assert all(h["citation"]["chunk_id"] != chunk_id for h in body)


class TestIngest:
    def test_inline_text(self, empty_client: TestClient) -> None:
        response = empty_client.post(
            "/api/ingest",
            json={"text": "# Pasted\n\nA note about reranking cross-encoders.", "title": "Pasted"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["documents_created"] == 1
        assert body["chunks_created"] >= 1

    def test_file_path_source(self, empty_client: TestClient, tmp_path: Path) -> None:
        path = tmp_path / "api.md"
        path.write_text("# API\n\nContent ingested through the HTTP layer.")
        body = empty_client.post("/api/ingest", json={"source": str(path)}).json()
        assert body["documents_created"] == 1

    def test_neither_source_nor_text_is_a_422(self, empty_client: TestClient) -> None:
        response = empty_client.post("/api/ingest", json={})
        assert response.status_code == 422
        assert response.json()["code"] == "validation_error"

    def test_unsupported_source_is_reported_in_errors(self, empty_client: TestClient) -> None:
        body = empty_client.post("/api/ingest", json={"source": "/tmp/thing.exe"}).json()
        assert body["documents_created"] == 0
        assert body["errors"]

    def test_upload_stores_the_file_for_citations(
        self, empty_client: TestClient, tmp_path: Path
    ) -> None:
        content = b"# Uploaded\n\nBody text about hybrid retrieval and fusion."
        response = empty_client.post(
            "/api/ingest/upload",
            files={"file": ("uploaded.md", content, "text/markdown")},
            data={"collection": "default"},
        )
        assert response.status_code == 200
        assert response.json()["documents_created"] == 1

    def test_embed_backfill_endpoint(self, empty_client: TestClient, tmp_path: Path) -> None:
        path = tmp_path / "later.md"
        path.write_text("# Later\n\nEmbedded in a second step.")
        empty_client.post("/api/ingest", json={"source": str(path), "embed": False})
        body = empty_client.post("/api/embed").json()
        assert body["embedded"] >= 1


class TestCorpus:
    def test_list_documents(self, client: TestClient) -> None:
        body = client.get("/api/documents").json()
        assert body["total"] == 3
        assert len(body["documents"]) == 3

    def test_pagination(self, client: TestClient) -> None:
        page = client.get("/api/documents", params={"limit": 2, "offset": 0}).json()
        assert len(page["documents"]) == 2
        assert page["total"] == 3

    def test_source_type_filter(self, client: TestClient) -> None:
        body = client.get("/api/documents", params={"source_type": "markdown"}).json()
        assert all(d["source_type"] == "markdown" for d in body["documents"])

    def test_get_one_document(self, client: TestClient) -> None:
        document_id = client.get("/api/documents").json()["documents"][0]["id"]
        assert client.get(f"/api/documents/{document_id}").json()["id"] == document_id

    def test_missing_document_is_a_404_with_a_code(self, client: TestClient) -> None:
        response = client.get("/api/documents/doc_nope")
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"

    def test_document_chunks_expose_boundaries(self, client: TestClient) -> None:
        document_id = client.get("/api/documents").json()["documents"][0]["id"]
        body = client.get(f"/api/documents/{document_id}/chunks").json()
        assert body["document"]["id"] == document_id
        assert body["chunks"]
        assert body["chunks"][0]["citation"]["label"] is not None

    def test_chunk_context(self, client: TestClient) -> None:
        document = next(
            d for d in client.get("/api/documents").json()["documents"] if d["n_chunks"] >= 2
        )
        chunks = client.get(f"/api/documents/{document['id']}/chunks").json()["chunks"]
        focus = chunks[0]["id"]
        body = client.get(f"/api/chunks/{focus}/context", params={"window": 1}).json()
        assert body["focus_chunk_id"] == focus
        assert len(body["chunks"]) >= 2

    def test_delete_document(self, client: TestClient) -> None:
        document_id = client.get("/api/documents").json()["documents"][0]["id"]
        assert client.delete(f"/api/documents/{document_id}").status_code == 200
        assert client.get("/api/documents").json()["total"] == 2

    def test_stats(self, client: TestClient) -> None:
        body = client.get("/api/collections/default/stats").json()
        assert body["stats"]["n_documents"] == 3
        assert body["stats"]["n_embedded"] == body["stats"]["n_chunks"]

    def test_collections_listing(self, client: TestClient) -> None:
        assert "default" in client.get("/api/collections").json()

    def test_heatmap_after_searching(self, client: TestClient) -> None:
        client.post("/api/search", json={"query": "fusion", "top_k": 2})
        body = client.get("/api/collections/default/heatmap").json()
        assert body["entries"]
        assert body["entries"][0]["hits"] >= 1

    def test_recent_queries(self, client: TestClient) -> None:
        client.post("/api/search", json={"query": "a distinctive probe query"})
        assert "a distinctive probe query" in client.get("/api/collections/default/queries").json()

    def test_delete_collection(self, client: TestClient) -> None:
        body = client.delete("/api/collections/default").json()
        assert body["deleted_documents"] == 3


class TestOpenAPI:
    def test_schema_is_generated(self, client: TestClient) -> None:
        spec = client.get("/openapi.json").json()
        assert "/api/search" in spec["paths"]
        assert "/api/ingest" in spec["paths"]
        assert spec["info"]["title"] == "Chat With Your Entire Knowledge Base"


class TestAsk:
    def test_answers_with_citations(self, client: TestClient) -> None:
        response = client.post(
            "/api/ask", json={"query": "what does the damping constant k default to?"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert body["generator"] == "extractive"
        assert body["context_chunks"] > 0
        for citation in body["citations"]:
            assert citation["chunk_id"]
            assert citation["deep_link"]
            assert citation["marker"] >= 1

    def test_sentences_carry_markers_and_spans(self, client: TestClient) -> None:
        body = client.post("/api/ask", json={"query": "reciprocal rank fusion"}).json()
        assert body["sentences"]
        text = body["answer"]
        for sentence in body["sentences"]:
            assert text[sentence["char_start"] : sentence["char_end"]] == sentence["text"]

    def test_every_marker_resolves_to_a_citation(self, client: TestClient) -> None:
        body = client.post("/api/ask", json={"query": "how is fusion done?"}).json()
        markers = {m for s in body["sentences"] for m in s["citation_markers"]}
        available = {c["marker"] for c in body["citations"]}
        assert markers <= available

    def test_retrieval_diagnostics_are_included_by_default(self, client: TestClient) -> None:
        body = client.post("/api/ask", json={"query": "fusion"}).json()
        assert body["retrieval"] is not None
        assert body["retrieval"]["strategy"] == "hybrid"

    def test_retrieval_can_be_omitted(self, client: TestClient) -> None:
        body = client.post("/api/ask", json={"query": "fusion", "include_retrieval": False}).json()
        assert body["retrieval"] is None

    def test_retrieval_overrides_are_honoured(self, client: TestClient) -> None:
        body = client.post(
            "/api/ask", json={"query": "fusion", "strategy": "lexical", "top_k": 2}
        ).json()
        assert body["retrieval"]["strategy"] == "lexical"
        assert len(body["retrieval"]["hits"]) <= 2

    def test_off_corpus_question_is_refused(self, client: TestClient) -> None:
        body = client.post("/api/ask", json={"query": "who won the 1998 world cup?"}).json()
        assert body["refused"] is True
        assert body["citations"] == []

    def test_blank_query_is_rejected(self, client: TestClient) -> None:
        assert client.post("/api/ask", json={"query": ""}).status_code == 422

    def test_empty_corpus_refuses(self, empty_client: TestClient) -> None:
        body = empty_client.post("/api/ask", json={"query": "anything"}).json()
        assert body["refused"] is True
        assert body["context_chunks"] == 0


class TestAskStream:
    def _events(self, raw: str) -> list[tuple[str, dict]]:
        import json

        out: list[tuple[str, dict]] = []
        for block in raw.strip().split("\n\n"):
            lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
            if "event" in lines and "data" in lines:
                out.append((lines["event"], json.loads(lines["data"])))
        return out

    def test_streams_deltas_then_done(self, client: TestClient) -> None:
        response = client.post("/api/ask/stream", json={"query": "damping constant"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = self._events(response.text)
        assert [name for name, _ in events].count("done") == 1
        deltas = [payload["text"] for name, payload in events if name == "delta"]
        done = next(payload["answer"] for name, payload in events if name == "done")
        assert "".join(deltas) == done["answer"]
        assert done["citations"] or done["refused"]

    def test_stream_reports_the_generator(self, client: TestClient) -> None:
        response = client.post("/api/ask/stream", json={"query": "fusion"})
        done = next(p["answer"] for n, p in self._events(response.text) if n == "done")
        assert done["generator"] == "extractive"


class TestHealthReportsGeneration:
    def test_generator_is_reported(self, client: TestClient) -> None:
        body = client.get("/api/health").json()
        assert body["generator"] == "extractive"
        assert body["generation_model"] == "extractive-v1"


class TestVerification:
    def test_answers_are_verified_by_default(self, client: TestClient) -> None:
        body = client.post("/api/ask", json={"query": "how is fusion done?"}).json()
        assert body["verified"] is True
        assert "verification_ms" in body["timings_ms"]

    def test_extractive_answers_verify_clean(self, client: TestClient) -> None:
        """Extractive output is verbatim from sources, so it must verify faithful."""
        body = client.post(
            "/api/ask", json={"query": "what does the damping constant k default to?"}
        ).json()
        if not body["refused"]:
            assert body["faithfulness"] == 1.0
            assert body["flagged_count"] == 0

    def test_sentences_carry_verdicts(self, client: TestClient) -> None:
        body = client.post("/api/ask", json={"query": "reciprocal rank fusion"}).json()
        verdicts = {s["verdict"] for s in body["sentences"]}
        assert verdicts <= {
            "supported",
            "partial",
            "unsupported",
            "uncited",
            "not_a_claim",
            None,
        }

    def test_verification_can_be_skipped(self, client: TestClient) -> None:
        body = client.post("/api/ask", json={"query": "fusion", "verify": False}).json()
        assert body["verified"] is False
        assert body["faithfulness"] is None

    def test_health_reports_the_verifier(self, client: TestClient) -> None:
        assert client.get("/api/health").json()["verifier"] == "lexical-verify"


class TestVisualization:
    def test_map_returns_points_and_clusters(self, client: TestClient) -> None:
        body = client.get("/api/collections/default/map").json()
        assert body["n_plotted"] > 0
        assert body["points"]
        assert body["clusters"]
        assert body["method"] in ("umap", "tsne", "pca")

    def test_coordinates_are_normalised(self, client: TestClient) -> None:
        for point in client.get("/api/collections/default/map").json()["points"]:
            assert 0.0 <= point["x"] <= 1.0
            assert 0.0 <= point["y"] <= 1.0

    def test_explicit_pca_reports_explained_variance(self, client: TestClient) -> None:
        """The honest caveat to show alongside the plot."""
        body = client.get("/api/collections/default/map", params={"method": "pca"}).json()
        assert body["method"] == "pca"
        assert body["explained_variance"] is not None

    def test_retrieval_coverage_reflects_searches(self, client: TestClient) -> None:
        assert client.get("/api/collections/default/map").json()["retrieval_coverage"] == 0.0
        client.post("/api/search", json={"query": "reciprocal rank fusion", "top_k": 3})
        body = client.get("/api/collections/default/map").json()
        assert body["retrieval_coverage"] > 0.0

    def test_cluster_count_can_be_fixed(self, client: TestClient) -> None:
        body = client.get("/api/collections/default/map", params={"clusters": 2}).json()
        assert len(body["clusters"]) <= 2

    def test_sampling_is_reported(self, client: TestClient) -> None:
        body = client.get("/api/collections/default/map", params={"max_points": 10}).json()
        assert body["n_plotted"] <= 10

    def test_empty_collection_map(self, empty_client: TestClient) -> None:
        body = empty_client.get("/api/collections/default/map").json()
        assert body["points"] == []
        assert body["notes"]

    def test_graph_nodes_and_edges(self, client: TestClient) -> None:
        body = client.get("/api/collections/default/graph", params={"min_similarity": 0.0}).json()
        assert len(body["nodes"]) == 3
        assert all(e["source"] != e["target"] for e in body["edges"])

    def test_graph_threshold_prunes(self, client: TestClient) -> None:
        body = client.get("/api/collections/default/graph", params={"min_similarity": 0.999}).json()
        assert body["edges"] == []

    def test_coverage_lists_never_retrieved_chunks(self, client: TestClient) -> None:
        """The actionable half: chunks nothing has ever reached."""
        body = client.get("/api/collections/default/coverage").json()
        assert body["n_chunks"] > 0
        assert body["coverage"] == 0.0
        assert body["never_retrieved"]
        assert body["most_retrieved"] == []

    def test_coverage_after_searching(self, client: TestClient) -> None:
        client.post("/api/search", json={"query": "reciprocal rank fusion", "top_k": 3})
        body = client.get("/api/collections/default/coverage").json()
        assert body["n_retrieved"] > 0
        assert body["coverage"] > 0.0
        assert body["most_retrieved"]
        assert body["n_queries_logged"] >= 1

    def test_invalid_method_is_rejected(self, client: TestClient) -> None:
        assert (
            client.get("/api/collections/default/map", params={"method": "magic"}).status_code
            == 422
        )
