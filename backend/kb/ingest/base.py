"""Connector contracts.

A connector's only job is to turn *a source* into :class:`ParsedDocument` values.
It does not chunk, embed, or write to the store — that is the pipeline's job —
and it does not decide how a citation is addressed beyond supplying the function
that builds one.

The key type is :class:`Segment`: a run of text plus a ``build_locator`` callback.
This is how one chunker serves every source type. A PDF yields one segment per
page whose callback produces a :class:`~kb.models.PdfLocator` with that page
number; a YouTube transcript yields one segment per time window whose callback
produces a :class:`~kb.models.YouTubeLocator` with that start time. Chunking runs
inside a segment, so the resulting chunk always inherits a correct address, and
the chunker never learns what a page or a timestamp is.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from kb.chunking.base import ChunkDraft, Chunker
from kb.models import ChunkKind, Locator, SourceType

#: Builds the address of a chunk from its position within a segment.
LocatorFactory = Callable[[ChunkDraft], Locator]


@dataclass(slots=True)
class Segment:
    """A run of text from a document, with the means to address positions in it."""

    text: str
    build_locator: LocatorFactory
    kind: ChunkKind = ChunkKind.PROSE
    chunker: Chunker | None = None
    """Overrides the pipeline's default chunker for this segment (e.g. code)."""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedDocument:
    """A source, parsed but not yet chunked."""

    title: str
    uri: str
    source_type: SourceType
    segments: list[Segment]
    raw_text: str = ""
    byte_size: int = 0
    language: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chunker: Chunker | None = None
    """Default chunker for every segment that does not specify its own."""

    def text_for_hash(self) -> str:
        """Canonical text used to detect that this source is already ingested."""
        return self.raw_text or "\n\n".join(s.text for s in self.segments)


@runtime_checkable
class Connector(Protocol):
    """Turns a source specifier into parsed documents."""

    name: str
    source_type: SourceType

    def can_handle(self, source: str) -> bool:  # pragma: no cover - protocol
        """True if this connector recognises ``source``."""
        ...

    def parse(
        self, source: str, **options: Any
    ) -> Iterable[ParsedDocument]:  # pragma: no cover - protocol
        """Parse ``source`` into one or more documents."""
        ...


# --------------------------------------------------------------------------- #
# shared helpers for file-backed connectors
# --------------------------------------------------------------------------- #


def has_extension(source: str, extensions: Sequence[str]) -> bool:
    """True if ``source`` looks like a local path with one of ``extensions``."""
    if "://" in source and not source.startswith("file://"):
        return False
    suffix = Path(source.removeprefix("file://")).suffix.lower()
    return suffix in {e.lower() for e in extensions}


def read_text_file(path: Path, *, max_bytes: int) -> str:
    """Read a text file, tolerating imperfect encodings.

    Real knowledge bases contain files saved on Windows in 1997. Decoding
    strictly would abort ingestion of an entire directory over one of them, so
    the fallback chain degrades to lossy UTF-8 rather than failing.
    """
    from kb.errors import IngestionError

    size = path.stat().st_size
    if size > max_bytes:
        raise IngestionError(
            f"{path} is {size} bytes, above the {max_bytes} byte limit",
            details={"path": str(path), "size": str(size)},
        )
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def title_from_path(path: Path) -> str:
    """Human-readable title from a filename: ``my_design-doc.md`` → ``My Design Doc``."""
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    return stem.title() if stem.islower() else (stem or path.name)


def title_from_markdown(text: str, fallback: str) -> str:
    """Prefer a document's own H1 over its filename."""
    for line in text.splitlines()[:40]:
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or fallback
        if stripped.startswith("title:"):
            candidate = stripped.split(":", 1)[1].strip().strip("\"'")
            if candidate:
                return candidate
    return fallback


def strip_front_matter(text: str) -> tuple[str, dict[str, Any]]:
    """Split YAML front matter from Markdown body.

    Front matter is metadata, not prose: leaving it in the body pollutes both
    the BM25 index and the embeddings with keys like ``draft: false``.
    """
    if not text.startswith("---"):
        return text, {}
    lines = text.split("\n")
    if len(lines) < 3:
        return text, {}
    for i in range(1, min(len(lines), 200)):
        if lines[i].strip() in ("---", "..."):
            block = "\n".join(lines[1:i])
            body = "\n".join(lines[i + 1 :])
            try:
                import yaml

                parsed = yaml.safe_load(block) or {}
                meta = parsed if isinstance(parsed, dict) else {"front_matter": parsed}
            except Exception:
                meta = {}
            return body.lstrip("\n"), meta
    return text, {}
