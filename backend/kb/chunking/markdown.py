"""Heading-aware Markdown chunker.

Markdown carries its own outline, and throwing that away is the most common
avoidable mistake in RAG ingestion. This chunker:

* segments on ATX headings and keeps the full ancestor path per chunk, so a
  citation can read ``Architecture › Retrieval › Fusion`` instead of "line 412";
* never splits inside a fenced code block, and marks such chunks
  :class:`~kb.models.ChunkKind.CODE` so a code-aware reranker can treat them
  differently;
* keeps tables intact and marks them ``TABLE``, since half a table is useless;
* prefixes each chunk with its heading path. That prefix measurably helps both
  BM25 and dense retrieval — the chunk becomes self-describing instead of
  relying on the reader knowing which section it fell out of.
"""

from __future__ import annotations

import re

from kb.chunking.base import ChunkDraft, line_starts, normalize_whitespace
from kb.chunking.recursive import RecursiveChunker
from kb.models import ChunkKind

_ATX_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(?:```|~~~)")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_SETEXT_H1_RE = re.compile(r"^=+\s*$")
_SETEXT_H2_RE = re.compile(r"^-{2,}\s*$")


class Section:
    """A heading and the body text beneath it, before size-based splitting."""

    __slots__ = ("char_end", "char_start", "heading_path", "line_start", "lines")

    def __init__(self, heading_path: list[str], line_start: int, char_start: int) -> None:
        self.heading_path = heading_path
        self.lines: list[str] = []
        self.line_start = line_start
        self.char_start = char_start
        self.char_end = char_start

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


class MarkdownChunker:
    """Split Markdown into chunks that respect its structure."""

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 180,
        min_chunk_size: int = 120,
        *,
        prefix_headings: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.prefix_headings = prefix_headings
        self._fallback = RecursiveChunker(chunk_size, chunk_overlap, min_chunk_size)

    # ------------------------------------------------------------------ #

    def chunk(self, text: str) -> list[ChunkDraft]:
        text = normalize_whitespace(text)
        if not text:
            return []
        sections = self._split_sections(text)
        drafts: list[ChunkDraft] = []
        for section in sections:
            drafts.extend(self._chunk_section(section))
        return drafts

    # ------------------------------------------------------------------ #

    def _split_sections(self, text: str) -> list[Section]:
        lines = text.split("\n")
        starts = line_starts(text)
        sections: list[Section] = []
        stack: list[tuple[int, str]] = []  # (level, title)
        current = Section([], 1, 0)
        in_fence = False

        for idx, line in enumerate(lines):
            if _FENCE_RE.match(line):
                in_fence = not in_fence

            heading = None
            if not in_fence:
                m = _ATX_RE.match(line)
                if m:
                    heading = (len(m.group(1)), m.group(2).strip())
                elif (
                    idx > 0
                    and lines[idx - 1].strip()
                    and (_SETEXT_H1_RE.match(line) or _SETEXT_H2_RE.match(line))
                ):
                    # Setext heading: the *previous* line was the title, so pull
                    # it back out of the section we already appended it to.
                    level = 1 if _SETEXT_H1_RE.match(line) else 2
                    if current.lines and current.lines[-1].strip() == lines[idx - 1].strip():
                        current.lines.pop()
                    heading = (level, lines[idx - 1].strip())

            if heading is not None:
                level, title = heading
                if current.text:
                    current.char_end = starts[idx] if idx < len(starts) else len(text)
                    sections.append(current)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                char_start = starts[idx] if idx < len(starts) else len(text)
                current = Section([t for _, t in stack], idx + 1, char_start)
                continue

            current.lines.append(line)

        if current.text:
            current.char_end = len(text)
            sections.append(current)
        return sections

    def _chunk_section(self, section: Section) -> list[ChunkDraft]:
        body = section.text
        if not body:
            return []
        prefix = self._heading_prefix(section.heading_path)
        blocks = _split_blocks(body)

        drafts: list[ChunkDraft] = []
        buf: list[tuple[str, ChunkKind]] = []
        buf_len = 0
        line_cursor = section.line_start

        def flush(start_line: int) -> None:
            nonlocal buf, buf_len
            if not buf:
                return
            merged = "\n\n".join(b for b, _ in buf).strip()
            if merged:
                kind = _dominant_kind([k for _, k in buf])
                drafts.append(
                    ChunkDraft(
                        text=f"{prefix}{merged}" if prefix else merged,
                        char_start=section.char_start,
                        char_end=section.char_end,
                        line_start=start_line,
                        line_end=start_line + merged.count("\n"),
                        heading_path=list(section.heading_path),
                        kind=kind,
                    )
                )
            buf, buf_len = [], 0

        block_start_line = line_cursor
        for block, kind in blocks:
            blen = len(block)
            if buf and buf_len + blen > self.chunk_size:
                flush(block_start_line)
                block_start_line = line_cursor
            if blen > self.chunk_size and kind is not ChunkKind.CODE:
                flush(block_start_line)
                # Oversized prose block: fall back to the recursive splitter,
                # keeping the section's heading path on every piece.
                for sub in self._fallback.chunk(block):
                    drafts.append(
                        ChunkDraft(
                            text=f"{prefix}{sub.text}" if prefix else sub.text,
                            char_start=section.char_start + sub.char_start,
                            char_end=section.char_start + sub.char_end,
                            line_start=line_cursor + sub.line_start - 1,
                            line_end=line_cursor + sub.line_end - 1,
                            heading_path=list(section.heading_path),
                            kind=kind,
                        )
                    )
                line_cursor += block.count("\n") + 2
                block_start_line = line_cursor
                continue
            buf.append((block, kind))
            buf_len += blen
            line_cursor += block.count("\n") + 2
        flush(block_start_line)

        return [d for d in drafts if len(d.text.strip()) >= min(self.min_chunk_size, 1)]

    def _heading_prefix(self, heading_path: list[str]) -> str:
        if not self.prefix_headings or not heading_path:
            return ""
        return " › ".join(heading_path) + "\n\n"


def _split_blocks(text: str) -> list[tuple[str, ChunkKind]]:
    """Break a section body into typed blocks: code fences, tables, prose."""
    lines = text.split("\n")
    blocks: list[tuple[str, ChunkKind]] = []
    buf: list[str] = []
    kind = ChunkKind.PROSE
    in_fence = False

    def flush() -> None:
        nonlocal buf, kind
        body = "\n".join(buf).strip()
        if body:
            blocks.append((body, kind))
        buf = []
        kind = ChunkKind.PROSE

    for line in lines:
        if _FENCE_RE.match(line):
            if in_fence:
                buf.append(line)
                in_fence = False
                flush()
            else:
                flush()
                in_fence = True
                kind = ChunkKind.CODE
                buf.append(line)
            continue
        if in_fence:
            buf.append(line)
            continue
        if _TABLE_ROW_RE.match(line):
            if kind is not ChunkKind.TABLE:
                flush()
                kind = ChunkKind.TABLE
            buf.append(line)
            continue
        if kind is ChunkKind.TABLE:
            flush()
        if not line.strip():
            flush()
            continue
        buf.append(line)

    flush()
    return blocks


def _dominant_kind(kinds: list[ChunkKind]) -> ChunkKind:
    """Pick the most specific kind present, since specificity is informative."""
    for special in (ChunkKind.CODE, ChunkKind.TABLE):
        if special in kinds:
            return special
    return kinds[0] if kinds else ChunkKind.PROSE
