"""The ingestion pipeline: parse → chunk → deduplicate → store → embed.

Ordering decisions worth naming:

* **Deduplicate before chunking, by document content hash.** Re-ingesting an
  unchanged file is the common case (a rerun, a re-uploaded export), and it is
  free to detect.
* **Deduplicate again after chunking, by chunk hash within the document.**
  Overlapping page boundaries and repeated licence headers otherwise create
  identical chunks that compete with each other for the same top-k slot.
* **Embed after the write, in batches, transactionally.** Ingestion that fails
  at the embedding step leaves a searchable document (BM25 works immediately)
  rather than nothing, and ``kb embed`` can finish the job later.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence

from kb.chunking.base import ChunkDraft, Chunker
from kb.chunking.recursive import RecursiveChunker
from kb.config import Settings
from kb.embeddings.base import Embedder
from kb.errors import IngestionError, KBError
from kb.ingest.base import ParsedDocument, Segment
from kb.ingest.registry import ConnectorRegistry, default_registry
from kb.models import (
    Chunk,
    Document,
    IngestionReport,
    content_hash,
    estimate_tokens,
)
from kb.store import SQLiteStore

log = logging.getLogger(__name__)


class IngestionPipeline:
    """Turns sources into searchable, embedded chunks."""

    def __init__(
        self,
        store: SQLiteStore,
        embedder: Embedder,
        settings: Settings,
        *,
        registry: ConnectorRegistry | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self.registry = registry or default_registry(settings)
        self._default_chunker = RecursiveChunker(
            settings.chunk_size, settings.chunk_overlap, settings.min_chunk_size
        )

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def ingest(
        self,
        source: str,
        *,
        collection: str = "default",
        embed: bool = True,
        **options: object,
    ) -> IngestionReport:
        """Ingest one source specifier, expanding directories and globs."""
        started = time.perf_counter()
        report = IngestionReport()

        try:
            expanded = list(self.registry.expand(source))
        except KBError as exc:
            report.errors.append({"source": source, "error": exc.message})
            report.elapsed_ms = (time.perf_counter() - started) * 1000
            return report

        for item in expanded:
            report = report.merge(
                self._ingest_one(item, collection=collection, embed=False, **options)
            )

        if embed and report.chunks_created:
            self.embed_pending(collection=collection)

        report.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return report

    def ingest_many(
        self,
        sources: Sequence[str],
        *,
        collection: str = "default",
        embed: bool = True,
        **options: object,
    ) -> IngestionReport:
        started = time.perf_counter()
        report = IngestionReport()
        for source in sources:
            report = report.merge(
                self.ingest(source, collection=collection, embed=False, **options)
            )
        if embed and report.chunks_created:
            self.embed_pending(collection=collection)
        report.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return report

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
        """Ingest a string directly — the paste-a-document path."""
        options: dict[str, object] = {"text": text, "title": title}
        if uri:
            options["uri"] = uri
        if markdown is not None:
            options["markdown"] = markdown
        return self.ingest(f"inline:{title}", collection=collection, embed=embed, **options)

    # ------------------------------------------------------------------ #

    def _ingest_one(
        self, source: str, *, collection: str, embed: bool, **options: object
    ) -> IngestionReport:
        report = IngestionReport()
        try:
            connector = self.registry.resolve(source)
            parsed_docs = list(connector.parse(source, **options))
        except KBError as exc:
            log.warning("ingest failed for %s: %s", source, exc.message)
            report.errors.append({"source": source, "error": exc.message})
            return report
        except Exception as exc:
            log.exception("unexpected ingest failure for %s", source)
            report.errors.append({"source": source, "error": f"{type(exc).__name__}: {exc}"})
            return report

        for parsed in parsed_docs:
            try:
                document, chunks, was_duplicate = self._store_parsed(parsed, collection)
            except KBError as exc:
                report.errors.append({"source": parsed.uri or source, "error": exc.message})
                continue
            if was_duplicate:
                report.duplicates_skipped += 1
                continue
            if document is None:
                report.documents_skipped += 1
                continue
            report.documents.append(document)
            report.chunks_created += len(chunks)

        if embed and report.chunks_created:
            self.embed_pending(collection=collection)
        return report

    def _store_parsed(
        self, parsed: ParsedDocument, collection: str
    ) -> tuple[Document | None, list[Chunk], bool]:
        text = parsed.text_for_hash()
        if not text.strip():
            return None, [], False

        doc_hash = content_hash(f"{parsed.uri}\x00{text}")
        existing = self.store.find_document_by_hash(collection, doc_hash)
        if existing is not None:
            log.debug("skipping unchanged document %s", parsed.uri)
            return existing, [], True

        document = Document(
            collection=collection,
            source_type=parsed.source_type,
            title=parsed.title,
            uri=parsed.uri,
            content_hash=doc_hash,
            byte_size=parsed.byte_size or len(text.encode("utf-8")),
            token_estimate=estimate_tokens(text),
            language=parsed.language,
            author=parsed.author,
            published_at=parsed.published_at,
            metadata=parsed.metadata,
        )

        chunks = self._build_chunks(parsed, document)
        if not chunks:
            raise IngestionError(
                f"{parsed.title!r} produced no chunks", details={"uri": parsed.uri}
            )

        stored = self.store.add_document(document, chunks)
        return stored, chunks, False

    def _build_chunks(self, parsed: ParsedDocument, document: Document) -> list[Chunk]:
        """Chunk each segment and attach locators, dropping intra-document duplicates."""
        chunker: Chunker = parsed.chunker or self._default_chunker
        chunks: list[Chunk] = []
        seen_hashes: set[str] = set()
        ordinal = 0

        for segment in parsed.segments:
            segment_chunker = segment.chunker or chunker
            for draft in self._chunk_segment(segment, segment_chunker):
                body = draft.text.strip()
                if not body:
                    continue
                digest = content_hash(body)
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                chunks.append(
                    Chunk(
                        document_id=document.id,
                        collection=document.collection,
                        ordinal=ordinal,
                        text=body,
                        kind=draft.kind if draft.kind else segment.kind,
                        locator=segment.build_locator(draft),
                        content_hash=digest,
                        document_title=document.title,
                        source_type=document.source_type,
                        heading_context=draft.heading_context,
                        metadata={**segment.metadata, **draft.metadata},
                    )
                )
                ordinal += 1
        return chunks

    def _chunk_segment(self, segment: Segment, chunker: Chunker) -> list[ChunkDraft]:
        drafts = chunker.chunk(segment.text)
        if not drafts and segment.text.strip():
            # A segment shorter than the minimum chunk size still deserves to be
            # searchable — a one-line page or a short transcript cue.
            drafts = [
                ChunkDraft(
                    text=segment.text.strip(),
                    char_start=0,
                    char_end=len(segment.text),
                    line_start=1,
                    line_end=max(1, segment.text.count("\n") + 1),
                    kind=segment.kind,
                )
            ]
        return drafts

    # ------------------------------------------------------------------ #
    # embedding
    # ------------------------------------------------------------------ #

    def embed_pending(self, *, collection: str = "default", batch_size: int | None = None) -> int:
        """Embed every chunk in ``collection`` that lacks a vector for the model.

        Idempotent and resumable: interrupting it and running it again picks up
        exactly where it stopped, which matters when embedding a large corpus
        against a rate-limited API.
        """
        size = batch_size or self.settings.embedding_batch_size
        total = 0
        while True:
            pending = self.store.unembedded_chunk_ids(
                collection, model=self.embedder.model, limit=size
            )
            if not pending:
                break
            chunks = self.store.get_chunks(pending)
            if not chunks:
                break
            vectors = self.embedder.embed_documents([_embedding_text(c) for c in chunks])
            written = self.store.upsert_embeddings(
                zip([c.id for c in chunks], vectors, strict=True),
                collection=collection,
                model=self.embedder.model,
            )
            total += written
            log.debug("embedded %s chunks (%s total)", written, total)
            if len(pending) < size:
                break
        return total

    def reembed_collection(self, *, collection: str = "default") -> int:
        """Drop and rebuild every vector — used after changing embedding model."""
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM embeddings WHERE collection = ?", (collection,))
        return self.embed_pending(collection=collection)


def _embedding_text(chunk: Chunk) -> str:
    """What actually gets embedded.

    The heading path is prepended when it is not already in the text. A chunk
    that reads "It defaults to 60." is unretrievable on its own; the same chunk
    under "Retrieval › Fusion › RRF" is not.
    """
    if chunk.heading_context and chunk.heading_context not in chunk.text[:200]:
        return f"{chunk.heading_context}\n\n{chunk.text}"
    return chunk.text


def build_pipeline(
    store: SQLiteStore,
    embedder: Embedder,
    settings: Settings,
    *,
    registry: ConnectorRegistry | None = None,
) -> IngestionPipeline:
    return IngestionPipeline(store, embedder, settings, registry=registry)


def iter_sources(registry: ConnectorRegistry, sources: Iterable[str]) -> Iterable[str]:
    """Expand many sources, skipping the ones no connector claims."""
    for source in sources:
        try:
            yield from registry.expand(source)
        except KBError:
            continue
