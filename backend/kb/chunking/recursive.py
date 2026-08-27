"""Recursive, structure-aware chunker.

The splitter tries the largest natural boundary first — sections, then
paragraphs, then sentences, then words — and only falls back to a hard character
cut when a single word exceeds the budget. That ordering matters for retrieval
quality: a chunk that ends mid-sentence both embeds worse and reads badly as a
citation.

Overlap is applied at sentence granularity rather than character granularity, so
the repeated context is always readable.
"""

from __future__ import annotations

import re

from kb.chunking.base import (
    ChunkDraft,
    line_of_offset,
    line_starts,
    split_sentences,
)
from kb.models import ChunkKind

# Boundaries in descending order of "how natural a break is here".
_SEPARATORS: tuple[str, ...] = (
    "\n\n\n",  # section break
    "\n\n",  # paragraph
    "\n",  # line
    ". ",  # sentence
    "; ",
    ", ",
    " ",
)

_FENCE_RE = re.compile(r"^\s*(```|~~~)")


class RecursiveChunker:
    """Split text into overlapping chunks on the best available boundary."""

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 180,
        min_chunk_size: int = 120,
        *,
        kind: ChunkKind = ChunkKind.PROSE,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.kind = kind

    # ------------------------------------------------------------------ #

    def chunk(self, text: str) -> list[ChunkDraft]:
        if not text.strip():
            return []
        starts = line_starts(text)
        pieces = self._split(text, 0)
        merged = self._merge_small(pieces)
        windows = self._apply_overlap(merged, text)
        drafts: list[ChunkDraft] = []
        for body, start, end in windows:
            body = body.strip()
            if not body:
                continue
            drafts.append(
                ChunkDraft(
                    text=body,
                    char_start=start,
                    char_end=end,
                    line_start=line_of_offset(starts, start),
                    line_end=line_of_offset(starts, max(start, end - 1)),
                    kind=self.kind,
                )
            )
        return drafts

    # ------------------------------------------------------------------ #

    def _split(self, text: str, offset: int) -> list[tuple[str, int, int]]:
        """Recursively split ``text`` into pieces at or below ``chunk_size``."""
        if len(text) <= self.chunk_size:
            return [(text, offset, offset + len(text))]

        for sep in _SEPARATORS:
            if sep not in text:
                continue
            parts = _split_keep_offsets(text, sep, offset)
            if len(parts) == 1:
                continue
            out: list[tuple[str, int, int]] = []
            buf: list[tuple[str, int, int]] = []
            buf_len = 0
            for part, p_start, p_end in parts:
                plen = len(part)
                if buf and buf_len + plen > self.chunk_size:
                    out.append(_join(buf))
                    buf, buf_len = [], 0
                if plen > self.chunk_size:
                    if buf:
                        out.append(_join(buf))
                        buf, buf_len = [], 0
                    out.extend(self._split(part, p_start))
                    continue
                buf.append((part, p_start, p_end))
                buf_len += plen
            if buf:
                out.append(_join(buf))
            return [p for p in out if p[0].strip()]

        # No separator left: hard cut. Only reachable for pathological input
        # such as a single 5,000-character token.
        return [
            (
                text[i : i + self.chunk_size],
                offset + i,
                offset + min(i + self.chunk_size, len(text)),
            )
            for i in range(0, len(text), self.chunk_size)
        ]

    def _merge_small(self, pieces: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
        """Fold runt chunks into their neighbour.

        A 30-character chunk is noise in a vector index: it matches everything
        weakly and nothing strongly.
        """
        if not pieces:
            return []
        out: list[tuple[str, int, int]] = []
        for text, start, end in pieces:
            if (
                out
                and len(text.strip()) < self.min_chunk_size
                and len(out[-1][0]) + len(text) <= self.chunk_size * 1.5
            ):
                prev_text, prev_start, _ = out[-1]
                out[-1] = (f"{prev_text}\n{text}", prev_start, end)
            else:
                out.append((text, start, end))
        # A single leading runt has no previous neighbour; merge it forwards.
        if len(out) > 1 and len(out[0][0].strip()) < self.min_chunk_size:
            (t0, s0, _), (t1, _, e1) = out[0], out[1]
            out[0:2] = [(f"{t0}\n{t1}", s0, e1)]
        return out

    def _apply_overlap(
        self, pieces: list[tuple[str, int, int]], full_text: str
    ) -> list[tuple[str, int, int]]:
        """Prepend the tail of the previous chunk to each chunk.

        Overlap is taken as whole trailing sentences so the repeated span reads
        as prose. The char range is widened to match, which keeps the locator
        honest about what the chunk actually covers.
        """
        if self.chunk_overlap <= 0 or len(pieces) < 2:
            return pieces
        out: list[tuple[str, int, int]] = [pieces[0]]
        for text, start, end in pieces[1:]:
            prev_text, prev_start, _ = out[-1]
            tail = _trailing_sentences(prev_text, self.chunk_overlap)
            if tail and not _inside_code_fence(full_text, start):
                new_start = max(prev_start, start - len(tail))
                out.append((f"{tail}\n{text}", new_start, end))
            else:
                out.append((text, start, end))
        return out


def _split_keep_offsets(text: str, sep: str, offset: int) -> list[tuple[str, int, int]]:
    """Split on ``sep`` while tracking absolute offsets, keeping the separator."""
    parts: list[tuple[str, int, int]] = []
    cursor = 0
    while True:
        idx = text.find(sep, cursor)
        if idx == -1:
            if cursor < len(text):
                parts.append((text[cursor:], offset + cursor, offset + len(text)))
            break
        cut = idx + len(sep)
        parts.append((text[cursor:cut], offset + cursor, offset + cut))
        cursor = cut
    return parts or [(text, offset, offset + len(text))]


def _join(buf: list[tuple[str, int, int]]) -> tuple[str, int, int]:
    return ("".join(p[0] for p in buf), buf[0][1], buf[-1][2])


def _trailing_sentences(text: str, budget: int) -> str:
    """Last whole sentences of ``text`` fitting in ``budget`` characters."""
    sentences = split_sentences(text)
    if not sentences:
        return text[-budget:] if len(text) > budget else text
    picked: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        if total + len(sentence) > budget and picked:
            break
        picked.insert(0, sentence)
        total += len(sentence) + 1
    return " ".join(picked)


def _inside_code_fence(text: str, offset: int) -> bool:
    """True if ``offset`` sits inside a fenced code block.

    Overlapping into a code fence produces chunks with unbalanced fences, which
    render as garbage in the UI, so overlap is skipped there.
    """
    fences = 0
    for line in text[:offset].splitlines():
        if _FENCE_RE.match(line):
            fences += 1
    return fences % 2 == 1
