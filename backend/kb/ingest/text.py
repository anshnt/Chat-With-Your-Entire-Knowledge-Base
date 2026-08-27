"""Plain-text and Markdown connectors.

Both produce :class:`~kb.models.TextLocator` addresses (line range + heading
path). Markdown gets the heading-aware chunker so a citation can say
``Architecture › Retrieval`` rather than "line 412"; plain text falls back to the
recursive splitter.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from kb.chunking.base import ChunkDraft
from kb.chunking.markdown import MarkdownChunker
from kb.chunking.recursive import RecursiveChunker
from kb.config import Settings
from kb.errors import IngestionError
from kb.ingest.base import (
    ParsedDocument,
    Segment,
    has_extension,
    read_text_file,
    strip_front_matter,
    title_from_markdown,
    title_from_path,
)
from kb.models import Locator, SourceType, TextLocator

MARKDOWN_EXTENSIONS = (".md", ".markdown", ".mdx", ".mdown", ".mkd")
TEXT_EXTENSIONS = (".txt", ".text", ".rst", ".log", ".csv", ".tsv")


def _text_locator_factory(file_path: str) -> Any:
    def build(draft: ChunkDraft) -> Locator:
        return TextLocator(
            line_start=max(1, draft.line_start),
            line_end=max(draft.line_start, draft.line_end),
            char_start=draft.char_start,
            char_end=draft.char_end,
            heading_path=list(draft.heading_path),
            file_path=file_path,
        )

    return build


class MarkdownConnector:
    """Ingests Markdown files, preserving their outline."""

    name = "markdown"
    source_type = SourceType.MARKDOWN

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def can_handle(self, source: str) -> bool:
        return has_extension(source, MARKDOWN_EXTENSIONS)

    def parse(self, source: str, **options: Any) -> Iterable[ParsedDocument]:
        path = Path(source.removeprefix("file://")).expanduser()
        if not path.is_file():
            raise IngestionError(f"not a file: {path}")
        raw = read_text_file(path, max_bytes=self.settings.max_document_bytes)
        body, front_matter = strip_front_matter(raw)
        title = (
            options.get("title")
            or front_matter.get("title")
            or title_from_markdown(body, title_from_path(path))
        )
        chunker = MarkdownChunker(
            self.settings.chunk_size, self.settings.chunk_overlap, self.settings.min_chunk_size
        )
        yield ParsedDocument(
            title=str(title),
            uri=str(path.resolve()),
            source_type=self.source_type,
            segments=[Segment(text=body, build_locator=_text_locator_factory(str(path)))],
            raw_text=body,
            byte_size=path.stat().st_size,
            author=_maybe_str(front_matter.get("author")),
            metadata={"front_matter": front_matter} if front_matter else {},
            chunker=chunker,
        )


class TextConnector:
    """Ingests plain-text files."""

    name = "text"
    source_type = SourceType.TEXT

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def can_handle(self, source: str) -> bool:
        return has_extension(source, TEXT_EXTENSIONS)

    def parse(self, source: str, **options: Any) -> Iterable[ParsedDocument]:
        path = Path(source.removeprefix("file://")).expanduser()
        if not path.is_file():
            raise IngestionError(f"not a file: {path}")
        body = read_text_file(path, max_bytes=self.settings.max_document_bytes)
        chunker = RecursiveChunker(
            self.settings.chunk_size, self.settings.chunk_overlap, self.settings.min_chunk_size
        )
        yield ParsedDocument(
            title=str(options.get("title") or title_from_path(path)),
            uri=str(path.resolve()),
            source_type=self.source_type,
            segments=[Segment(text=body, build_locator=_text_locator_factory(str(path)))],
            raw_text=body,
            byte_size=path.stat().st_size,
            chunker=chunker,
        )


class InlineTextConnector:
    """Ingests text passed directly rather than read from disk.

    Used by the API's paste-a-document path and by tests, which is why it exists
    as a first-class connector instead of a special case in the pipeline.
    """

    name = "inline"
    source_type = SourceType.TEXT

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def can_handle(self, source: str) -> bool:
        return source.startswith("inline:")

    def parse(self, source: str, **options: Any) -> Iterable[ParsedDocument]:
        text = str(options.get("text", ""))
        if not text.strip():
            raise IngestionError("inline source requires non-empty 'text'")
        title = str(options.get("title") or source.removeprefix("inline:") or "Pasted document")
        is_markdown = bool(options.get("markdown", _looks_like_markdown(text)))
        chunker = (
            MarkdownChunker(
                self.settings.chunk_size, self.settings.chunk_overlap, self.settings.min_chunk_size
            )
            if is_markdown
            else RecursiveChunker(
                self.settings.chunk_size, self.settings.chunk_overlap, self.settings.min_chunk_size
            )
        )
        yield ParsedDocument(
            title=title,
            uri=str(options.get("uri") or f"inline:{title}"),
            source_type=SourceType.MARKDOWN if is_markdown else SourceType.TEXT,
            segments=[Segment(text=text, build_locator=_text_locator_factory(""))],
            raw_text=text,
            byte_size=len(text.encode("utf-8")),
            chunker=chunker,
        )


def _looks_like_markdown(text: str) -> bool:
    head = "\n".join(text.splitlines()[:60])
    return any(marker in head for marker in ("\n# ", "\n## ", "\n- ", "```")) or head.startswith(
        "# "
    )


def _maybe_str(value: Any) -> str | None:
    return str(value) if isinstance(value, str | int | float) else None
