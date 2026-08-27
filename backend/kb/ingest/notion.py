"""Notion export connector.

A Notion Markdown/CSV export is a directory tree (or a zip of one) with a very
specific shape, and handling that shape is the whole job:

* **Filenames carry a 32-hex page id.** ``Runbooks a1b2c3….md`` — so the id must
  be stripped from the title (or every page is titled with a hash) and *kept* as
  the page id, because it is what turns a citation into a ``notion.so`` link.
* **Nesting encodes the page hierarchy.** A subdirectory named after a page holds
  that page's children. Reconstructing the ancestor path is what lets a citation
  read ``Engineering › Runbooks › On-call`` instead of a filename.
* **Databases export as CSV.** A one-row-per-line dump embeds terribly. Each row
  becomes a ``key: value`` block instead, which is both readable and retrievable,
  and the table's own name comes from the file.
* **Percent-encoded filenames.** Exports URL-encode spaces and punctuation, so
  titles have to be decoded or they read as ``On%20call%20rota``.

Handles a directory, a ``.zip``, or a single exported file.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from kb.chunking.base import ChunkDraft
from kb.chunking.markdown import MarkdownChunker
from kb.config import Settings
from kb.errors import IngestionError
from kb.ingest.base import ParsedDocument, Segment, strip_front_matter
from kb.models import ChunkKind, Locator, NotionLocator, SourceType

log = logging.getLogger(__name__)

#: Notion appends a 32-hex id to every exported file and directory name.
_PAGE_ID_RE = re.compile(r"[ _-]([0-9a-f]{32})(?=\.|$)", re.I)
_BARE_ID_RE = re.compile(r"^([0-9a-f]{32})$", re.I)

MARKDOWN_SUFFIXES = (".md", ".markdown")
CSV_SUFFIXES = (".csv",)

#: Notion writes an "All" view alongside each database CSV; indexing both
#: duplicates every row.
_DUPLICATE_VIEW_RE = re.compile(r"_all\.csv$", re.I)


class NotionConnector:
    """Ingests a Notion Markdown/CSV export."""

    name = "notion"
    source_type = SourceType.NOTION

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def can_handle(self, source: str) -> bool:
        if source.startswith("notion:"):
            return True
        path = Path(source.removeprefix("file://")).expanduser()
        if path.suffix.lower() == ".zip":
            return looks_like_notion_zip(path)
        if path.is_dir():
            return looks_like_notion_export(path)
        if path.suffix.lower() in MARKDOWN_SUFFIXES + CSV_SUFFIXES and path.is_file():
            # A single exported file still carries the page id in its name.
            return bool(_PAGE_ID_RE.search(path.stem + path.suffix))
        return False

    # ------------------------------------------------------------------ #

    def parse(self, source: str, **options: Any) -> Iterable[ParsedDocument]:
        """Parse an export directory, a zip of one, or a single exported page.

        ``options`` is unused: a Notion export carries its own titles and page
        ids in filenames, so there is nothing useful for a caller to override.
        """
        del options
        path = Path(source.removeprefix("notion:").removeprefix("file://")).expanduser()
        if not path.exists():
            raise IngestionError(f"not found: {path}")

        if path.suffix.lower() == ".zip":
            return list(self._parse_zip(path))
        if path.is_dir():
            return list(self._parse_directory(path))
        return list(self._parse_file(path, ancestors=[]))

    # ------------------------------------------------------------------ #

    def _parse_zip(self, path: Path) -> Iterator[ParsedDocument]:
        """Read the export without unpacking it to disk."""
        try:
            archive = zipfile.ZipFile(path)
        except zipfile.BadZipFile as exc:
            raise IngestionError(f"{path.name} is not a readable zip: {exc}") from exc

        with archive:
            names = [
                n
                for n in archive.namelist()
                if not n.endswith("/")
                and Path(n).suffix.lower() in MARKDOWN_SUFFIXES + CSV_SUFFIXES
                and not _DUPLICATE_VIEW_RE.search(n)
            ]
            if not names:
                raise IngestionError(
                    f"{path.name} contains no Markdown or CSV files — is it a Notion export?"
                )
            produced = 0
            for name in sorted(names):
                try:
                    raw = archive.read(name)
                except Exception as exc:
                    log.warning("could not read %s from %s: %s", name, path.name, exc)
                    continue
                text = _decode(raw)
                if not text.strip():
                    continue
                member = Path(name)
                ancestors = [
                    clean_notion_name(part) for part in member.parts[:-1] if part not in (".", "")
                ]
                document = self._build(
                    text,
                    filename=member.name,
                    ancestors=ancestors,
                    uri=f"{path.resolve()}!{name}",
                )
                if document is not None:
                    produced += 1
                    yield document
            if produced == 0:
                raise IngestionError(f"{path.name} produced no usable pages")

    def _parse_directory(self, root: Path) -> Iterator[ParsedDocument]:
        produced = 0
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in MARKDOWN_SUFFIXES + CSV_SUFFIXES:
                continue
            if _DUPLICATE_VIEW_RE.search(file_path.name):
                continue
            relative = file_path.relative_to(root)
            ancestors = [clean_notion_name(part) for part in relative.parts[:-1]]
            for document in self._parse_file(file_path, ancestors=ancestors):
                produced += 1
                yield document
        if produced == 0:
            raise IngestionError(f"no Notion pages found under {root}")

    def _parse_file(self, path: Path, *, ancestors: list[str]) -> Iterator[ParsedDocument]:
        try:
            text = _decode(path.read_bytes())
        except OSError as exc:
            raise IngestionError(f"could not read {path}: {exc}") from exc
        if not text.strip():
            return
        document = self._build(
            text, filename=path.name, ancestors=ancestors, uri=str(path.resolve())
        )
        if document is not None:
            yield document

    # ------------------------------------------------------------------ #

    def _build(
        self, text: str, *, filename: str, ancestors: list[str], uri: str
    ) -> ParsedDocument | None:
        title, page_id = split_notion_filename(filename)
        page_path = [*ancestors, title]

        if filename.lower().endswith(CSV_SUFFIXES):
            body = csv_to_markdown(text, title)
            kind = ChunkKind.TABLE
            if not body.strip():
                return None
        else:
            body, _ = strip_front_matter(text)
            body = strip_notion_artifacts(body)
            kind = ChunkKind.PROSE
            if len(body.strip()) < 20:
                # An empty Notion page is common and not worth a document.
                return None

        chunker = MarkdownChunker(
            self.settings.chunk_size, self.settings.chunk_overlap, self.settings.min_chunk_size
        )
        return ParsedDocument(
            title=" › ".join(page_path) if len(page_path) > 1 else title,
            uri=uri,
            source_type=self.source_type,
            segments=[
                Segment(
                    text=body,
                    build_locator=_notion_locator_factory(page_path, page_id),
                    kind=kind,
                )
            ],
            raw_text=body,
            byte_size=len(text.encode("utf-8")),
            metadata={
                "notion_page_id": page_id,
                "page_path": page_path,
                "filename": filename,
            },
            chunker=chunker,
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def looks_like_notion_export(root: Path) -> bool:
    """True when a directory contains files with Notion's id-suffixed names."""
    try:
        for candidate in root.rglob("*"):
            if candidate.is_file() and _PAGE_ID_RE.search(candidate.name):
                return True
    except OSError:  # pragma: no cover - unreadable directory
        return False
    return False


def looks_like_notion_zip(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return any(_PAGE_ID_RE.search(Path(n).name) for n in archive.namelist()[:400])
    except (zipfile.BadZipFile, OSError):
        return False


def split_notion_filename(filename: str) -> tuple[str, str | None]:
    """``Runbooks a1b2….md`` → ``("Runbooks", "a1b2…")``."""
    stem = Path(filename).stem
    decoded = unquote(stem)
    match = _PAGE_ID_RE.search(decoded)
    if match:
        page_id = match.group(1).lower()
        title = decoded[: match.start()].strip(" _-")
        return (title or page_id), page_id
    bare = _BARE_ID_RE.match(decoded)
    if bare:
        return decoded, bare.group(1).lower()
    return decoded.strip(), None


def clean_notion_name(name: str) -> str:
    """A directory or page name with its id and percent-encoding removed."""
    title, _ = split_notion_filename(name)
    return title


def csv_to_markdown(text: str, table_name: str) -> str:
    """Turn a Notion database export into retrievable text.

    A CSV row dumped verbatim embeds terribly: the column values run together
    with no indication of what they mean. One ``key: value`` block per row keeps
    the field names attached to their values, which both reads well as a citation
    and gives BM25 and the embedder something to match on.
    """
    try:
        reader = csv.reader(io.StringIO(text))
        rows = [row for row in reader if any(cell.strip() for cell in row)]
    except csv.Error as exc:
        log.warning("could not parse CSV %s: %s", table_name, exc)
        return ""
    if len(rows) < 2:
        return ""

    header = [h.strip() or f"column {i + 1}" for i, h in enumerate(rows[0])]
    blocks: list[str] = [f"# {table_name}\n"]
    for index, row in enumerate(rows[1:], start=1):
        padded = list(row) + [""] * (len(header) - len(row))
        # The first column is Notion's title field, so it names the entry.
        entry_title = padded[0].strip() or f"Row {index}"
        lines = [f"## {entry_title}", ""]
        lines += [
            f"- {name}: {value.strip()}"
            for name, value in zip(header, padded, strict=False)
            if value.strip()
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks).strip()


def strip_notion_artifacts(text: str) -> str:
    """Remove export artefacts that are not content.

    Notion writes the page title as an H1 and then repeats page properties as a
    metadata block; both duplicate information the locator already carries, and
    leaving them in means every chunk of a page shares the same opening.
    """
    lines = text.split("\n")
    out: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        # Notion links every child page as "[Title](Title%20<id>.md)".
        if re.fullmatch(r"\[.*\]\([^)]*[0-9a-f]{32}[^)]*\)", stripped, re.I):
            continue
        # Property lines appear immediately after the title.
        if index < 12 and re.match(r"^[A-Z][\w ]{0,28}:\s*$", stripped):
            continue
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _notion_locator_factory(page_path: list[str], page_id: str | None) -> Any:
    def build(draft: ChunkDraft) -> Locator:
        return NotionLocator(
            page_path=list(page_path),
            notion_page_id=page_id,
            line_start=max(1, draft.line_start),
            line_end=max(draft.line_start, draft.line_end),
            heading_path=list(draft.heading_path),
        )

    return build
