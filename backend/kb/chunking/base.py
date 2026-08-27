"""Chunking contracts.

A chunker's job is *not* to build :class:`~kb.models.Chunk` objects — it does not
know whether the text came from a PDF page or a YouTube transcript. It produces
:class:`ChunkDraft` spans that carry the text plus enough positional information
(line range, char range, heading path) for a connector to construct the right
:class:`~kb.models.Locator`.

Keeping that split means the same recursive splitter serves every source type,
and adding a source never means touching chunking logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from kb.models import ChunkKind


@dataclass(slots=True)
class ChunkDraft:
    """A candidate chunk: text plus where it came from in the input."""

    text: str
    char_start: int
    char_end: int
    line_start: int = 1
    line_end: int = 1
    heading_path: list[str] = field(default_factory=list)
    kind: ChunkKind = ChunkKind.PROSE
    metadata: dict = field(default_factory=dict)

    @property
    def heading_context(self) -> str:
        return " › ".join(self.heading_path)

    def with_text(self, text: str) -> ChunkDraft:
        return ChunkDraft(
            text=text,
            char_start=self.char_start,
            char_end=self.char_end,
            line_start=self.line_start,
            line_end=self.line_end,
            heading_path=list(self.heading_path),
            kind=self.kind,
            metadata=dict(self.metadata),
        )


class Chunker(Protocol):
    """Anything that turns a document's text into positioned drafts."""

    def chunk(self, text: str) -> list[ChunkDraft]:  # pragma: no cover - protocol
        ...


# --------------------------------------------------------------------------- #
# shared text utilities
# --------------------------------------------------------------------------- #

# Sentence boundary: a terminator followed by whitespace and a capital/quote/digit,
# with the common abbreviations that would otherwise cause false splits excluded.
_ABBREVIATIONS = (
    "e.g",
    "i.e",
    "etc",
    "vs",
    "fig",
    "eq",
    "no",
    "cf",
    "al",
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "inc",
    "ltd",
    "st",
    "approx",
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])[\s]+(?=[\"'(\[]?[A-Z0-9])")
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences without dragging in an NLP dependency.

    Deliberately conservative: abbreviations like ``e.g.`` are re-joined rather
    than producing one-word fragments, because fragments wreck both the
    extractive generator and per-sentence citation verification.
    """
    if not text.strip():
        return []
    raw = _SENTENCE_RE.split(text.strip())
    out: list[str] = []
    for piece in raw:
        piece = piece.strip()
        if not piece:
            continue
        if out and _ends_with_abbreviation(out[-1]):
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out


def _ends_with_abbreviation(sentence: str) -> bool:
    tail = sentence.rstrip().rstrip(".").split()
    if not tail:
        return False
    return tail[-1].lower().rstrip(".") in _ABBREVIATIONS


def tokenize_words(text: str) -> list[str]:
    """Lowercase word tokens — the shared vocabulary for lexical scoring."""
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def line_of_offset(prefix_line_starts: list[int], offset: int) -> int:
    """1-based line number containing ``offset``, via binary search."""
    lo, hi = 0, len(prefix_line_starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if prefix_line_starts[mid] <= offset:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def line_starts(text: str) -> list[int]:
    """Character offset at which each line begins."""
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def normalize_whitespace(text: str) -> str:
    """Collapse runs of blank lines and trailing spaces, preserving paragraphs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
