"""PDF connector.

Each page becomes one :class:`~kb.ingest.base.Segment`, so every chunk carries
the page it came from and a citation can link to ``#page=12`` — the whole point
of the exercise. Chunking happens *within* a page, which means a chunk never
straddles a page boundary and therefore never has an ambiguous address.

Two things extracted PDF text always needs and rarely gets:

* **De-hyphenation.** ``retriev-\\nal`` must become ``retrieval`` or neither BM25
  nor the embedder will ever match the word.
* **Header/footer removal.** A line repeated on most pages ("CONFIDENTIAL",
  "Acme Corp — 2024") is boilerplate. It is detected by frequency across pages
  rather than by position, which is robust to varying layouts, and stripped so
  it stops polluting every chunk in the document.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from kb.chunking.base import ChunkDraft
from kb.chunking.recursive import RecursiveChunker
from kb.config import Settings
from kb.errors import IngestionError, MissingDependencyError
from kb.ingest.base import ParsedDocument, Segment, has_extension, title_from_path
from kb.models import Locator, PdfLocator, SourceType

PDF_EXTENSIONS = (".pdf",)

_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s*)?[ivxlcdm\d]{1,6}\s*(?:/\s*\d+)?\s*$", re.IGNORECASE)

#: A line is boilerplate if it appears on at least this share of pages *and* on
#: at least ``_MIN_BOILERPLATE_OCCURRENCES`` of them. The absolute floor matters:
#: on a short document a ratio alone would strip a heading that legitimately
#: appears twice.
_BOILERPLATE_RATIO = 0.6
_MIN_BOILERPLATE_OCCURRENCES = 3
_MIN_PAGES_FOR_BOILERPLATE = 3


class PDFConnector:
    """Ingests PDFs page by page."""

    name = "pdf"
    source_type = SourceType.PDF

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def can_handle(self, source: str) -> bool:
        return has_extension(source, PDF_EXTENSIONS)

    def parse(self, source: str, **options: Any) -> Iterable[ParsedDocument]:
        path = Path(source.removeprefix("file://")).expanduser()
        if not path.is_file():
            raise IngestionError(f"not a file: {path}")
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - pypdf is a base dependency
            raise MissingDependencyError("pypdf", "all") from exc

        try:
            reader = PdfReader(str(path))
        except Exception as exc:
            raise IngestionError(f"could not read PDF {path.name}: {exc}") from exc

        if reader.is_encrypted:
            # An empty user password is common for "protected" PDFs.
            try:
                reader.decrypt("")
            except Exception as exc:
                raise IngestionError(f"{path.name} is encrypted and cannot be read") from exc

        pages = _extract_pages(reader)
        if not any(p.strip() for p in pages):
            raise IngestionError(
                f"{path.name} contains no extractable text — it is likely a scan. "
                "Run OCR over it first, then ingest the output."
            )

        pages = _strip_boilerplate(pages)
        page_count = len(pages)
        file_url = options.get("file_url") or f"/files/{path.name}"
        chunker = RecursiveChunker(
            self.settings.chunk_size, self.settings.chunk_overlap, self.settings.min_chunk_size
        )

        segments: list[Segment] = []
        for index, page_text in enumerate(pages, start=1):
            cleaned = _clean_page(page_text)
            if not cleaned.strip():
                continue
            segments.append(
                Segment(
                    text=cleaned,
                    build_locator=_pdf_locator_factory(index, page_count, str(file_url)),
                    metadata={"page": index},
                )
            )

        if not segments:
            raise IngestionError(f"{path.name} yielded no usable text after cleaning")

        info = _document_info(reader)
        yield ParsedDocument(
            title=str(options.get("title") or info.get("title") or title_from_path(path)),
            uri=str(path.resolve()),
            source_type=self.source_type,
            segments=segments,
            raw_text="\n\n".join(s.text for s in segments),
            byte_size=path.stat().st_size,
            author=info.get("author"),
            metadata={"page_count": page_count, "file_url": str(file_url)},
            chunker=chunker,
        )


def _pdf_locator_factory(page: int, page_count: int, file_url: str) -> Any:
    def build(draft: ChunkDraft) -> Locator:
        return PdfLocator(
            page=page,
            page_count=page_count,
            char_start=draft.char_start,
            char_end=draft.char_end,
            file_url=file_url,
        )

    return build


def _extract_pages(reader: Any) -> list[str]:
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return pages


def _document_info(reader: Any) -> dict[str, str | None]:
    try:
        meta = reader.metadata or {}
    except Exception:
        return {}
    title = (getattr(meta, "title", None) or "").strip() or None
    author = (getattr(meta, "author", None) or "").strip() or None
    return {"title": title, "author": author}


def _clean_page(text: str) -> str:
    """Repair the artefacts of PDF text extraction."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = text.replace("ﬀ", "ff").replace("ﬁ", "fi").replace("ﬂ", "fl")
    lines = []
    for line in text.split("\n"):
        stripped = _MULTISPACE_RE.sub(" ", line).strip()
        if stripped and not _PAGE_NUMBER_RE.match(stripped):
            lines.append(stripped)
    # Re-join wrapped lines into paragraphs: a line not ending in sentence
    # punctuation is a soft wrap, not a paragraph break.
    out: list[str] = []
    for line in lines:
        if out and not out[-1].endswith((".", "!", "?", ":", ";")) and not line[:1].isupper():
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    return "\n".join(out).strip()


def _strip_boilerplate(pages: list[str]) -> list[str]:
    """Remove lines that repeat across most pages (running heads and footers)."""
    if len(pages) < _MIN_PAGES_FOR_BOILERPLATE:
        return pages
    counts: Counter[str] = Counter()
    for page in pages:
        # Only the first and last few lines can be a running head or footer.
        lines = [line.strip() for line in page.split("\n") if line.strip()]
        counts.update(set(lines[:3] + lines[-3:]))
    threshold = max(_MIN_BOILERPLATE_OCCURRENCES, int(len(pages) * _BOILERPLATE_RATIO))
    boilerplate = {line for line, count in counts.items() if count >= threshold and len(line) < 120}
    if not boilerplate:
        return pages
    return [
        "\n".join(line for line in page.split("\n") if line.strip() not in boilerplate)
        for page in pages
    ]
