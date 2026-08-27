"""The façade: one object that owns a store, an embedder, and a retriever.

Everything above this layer — CLI, HTTP API, evaluation harness — goes through
:class:`KnowledgeBase`. That is deliberate: it means the evaluation numbers
describe the same code path that answers a real query, which is the only way a
retrieval benchmark is worth anything.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from kb.config import Settings, get_settings
from kb.embeddings import build_embedder
from kb.embeddings.base import Embedder
from kb.generate import Generator, build_generator
from kb.ingest.pipeline import IngestionPipeline
from kb.ingest.registry import ConnectorRegistry, default_registry
from kb.models import (
    Answer,
    Chunk,
    CollectionStats,
    Document,
    IngestionReport,
    RetrievalRequest,
    RetrievalResult,
    ScoredChunk,
    SourceType,
)
from kb.retrieval.hybrid import HybridRetriever, Reranker, request_from_settings
from kb.store import SQLiteStore

log = logging.getLogger(__name__)


class KnowledgeBase:
    """A knowledge base: ingest sources, then search them."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: SQLiteStore | None = None,
        embedder: Embedder | None = None,
        registry: ConnectorRegistry | None = None,
        reranker: Reranker | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.store = store or SQLiteStore(self.settings.db_path)
        self.embedder = embedder or build_embedder(self.settings)
        self.registry = registry or default_registry(self.settings)
        self.reranker = reranker if reranker is not None else self._build_reranker()
        self.generator = generator or build_generator(self.settings)
        self.pipeline = IngestionPipeline(
            self.store, self.embedder, self.settings, registry=self.registry
        )
        self.retriever = HybridRetriever(self.store, self.embedder, reranker=self.reranker)

    # ------------------------------------------------------------------ #

    def _build_reranker(self) -> Reranker | None:
        """Attach the configured reranker when the rerank package is available.

        Kept as a soft import so the retrieval core does not depend on the
        reranking layer — the pipeline degrades to fusion-only rather than
        failing to construct.
        """
        try:
            from kb.rerank import build_reranker
        except ImportError:
            return None
        try:
            return build_reranker(self.settings)
        except Exception as exc:
            log.warning("reranker unavailable (%s); continuing without it", exc)
            return None

    # ------------------------------------------------------------------ #
    # ingestion
    # ------------------------------------------------------------------ #

    def ingest(
        self, source: str, *, collection: str = "default", embed: bool = True, **options: Any
    ) -> IngestionReport:
        return self.pipeline.ingest(source, collection=collection, embed=embed, **options)

    def ingest_many(
        self, sources: list[str], *, collection: str = "default", embed: bool = True, **options: Any
    ) -> IngestionReport:
        return self.pipeline.ingest_many(sources, collection=collection, embed=embed, **options)

    def ingest_text(
        self,
        text: str,
        *,
        title: str,
        collection: str = "default",
        embed: bool = True,
        uri: str | None = None,
        markdown: bool | None = None,
    ) -> IngestionReport:
        return self.pipeline.ingest_text(
            text, title=title, collection=collection, embed=embed, uri=uri, markdown=markdown
        )

    def embed_pending(self, *, collection: str = "default") -> int:
        return self.pipeline.embed_pending(collection=collection)

    def reembed(self, *, collection: str = "default") -> int:
        return self.pipeline.reembed_collection(collection=collection)

    # ------------------------------------------------------------------ #
    # retrieval
    # ------------------------------------------------------------------ #

    def search(self, query: str, **overrides: Any) -> RetrievalResult:
        """Search using configured defaults, overridden per call."""
        collection = overrides.pop("collection", "default")
        request = request_from_settings(self.settings, query, **overrides)
        request = request.model_copy(update={"collection": collection})
        return self.retriever.retrieve(request)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        return self.retriever.retrieve(request)

    def similar_chunks(
        self, chunk_id: str, *, collection: str = "default", limit: int = 10
    ) -> list[ScoredChunk]:
        return self.retriever.dense.similar_to_chunk(chunk_id, collection=collection, limit=limit)

    # ------------------------------------------------------------------ #
    # answering
    # ------------------------------------------------------------------ #

    def ask(self, query: str, **overrides: Any) -> Answer:
        """Retrieve, then answer with citations resolved to source positions.

        Retrieval overrides are passed straight through, so an answer can be
        produced under any retrieval configuration — which is what lets the
        evaluation harness compare end-to-end answer quality across retrieval
        settings rather than only comparing rankings.
        """
        result = self.search(query, **overrides)
        answer = self.generator.generate(query, result.results, retrieval=result)
        answer.timings_ms.update({f"retrieval_{k}": v for k, v in result.timings_ms.items()})
        return answer

    def ask_stream(self, query: str, **overrides: Any) -> Iterator[tuple[str, Answer | None]]:
        """Stream an answer: ``(delta, None)`` while generating, then ``("", answer)``.

        Citations are only resolvable once the text is complete — a marker may be
        half-emitted — so they arrive with the terminal event rather than being
        patched in mid-stream.
        """
        result = self.search(query, **overrides)
        for delta, answer in self.generator.stream(query, result.results, retrieval=result):
            if answer is not None:
                answer.timings_ms.update(
                    {f"retrieval_{k}": v for k, v in result.timings_ms.items()}
                )
            yield delta, answer

    # ------------------------------------------------------------------ #
    # corpus inspection
    # ------------------------------------------------------------------ #

    def stats(self, collection: str = "default") -> CollectionStats:
        return self.store.collection_stats(collection)

    def collections(self) -> list[str]:
        return self.store.list_collections()

    def documents(
        self,
        collection: str = "default",
        *,
        limit: int = 100,
        offset: int = 0,
        source_type: SourceType | None = None,
        search: str | None = None,
    ) -> list[Document]:
        return self.store.list_documents(
            collection, limit=limit, offset=offset, source_type=source_type, search=search
        )

    def document(self, document_id: str) -> Document:
        return self.store.get_document(document_id)

    def document_chunks(self, document_id: str) -> list[Chunk]:
        return self.store.document_chunks(document_id)

    def chunk(self, chunk_id: str) -> Chunk:
        return self.store.get_chunk(chunk_id)

    def chunk_with_context(self, chunk_id: str, *, window: int = 1) -> list[Chunk]:
        return self.store.chunk_neighbours(chunk_id, window=window)

    def delete_document(self, document_id: str) -> None:
        self.store.delete_document(document_id)

    def delete_collection(self, collection: str) -> int:
        return self.store.delete_collection(collection)

    def heatmap(self, collection: str = "default", *, limit: int = 500) -> list[dict[str, Any]]:
        return self.store.retrieval_heatmap(collection, limit=limit)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> KnowledgeBase:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"KnowledgeBase(db={self.settings.db_path}, embedder={self.embedder.model!r}, "
            f"strategy={self.settings.retrieval_strategy.value})"
        )


def open_knowledge_base(
    db_path: str | Path | None = None, **setting_overrides: Any
) -> KnowledgeBase:
    """Convenience constructor used by the CLI and by scripts.

    ``open_knowledge_base("./data/kb.db", top_k=5)``
    """
    base = get_settings()
    overrides = dict(setting_overrides)
    if db_path is not None:
        path = Path(db_path)
        overrides.setdefault("data_dir", path.parent if path.suffix else path)
        if path.suffix:
            overrides.setdefault("db_filename", path.name)
    settings = base.model_copy(update=overrides) if overrides else base
    return KnowledgeBase(settings)
