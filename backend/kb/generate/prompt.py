"""Prompt assembly and citation-marker parsing.

Two jobs, both of which are where grounded generation usually goes wrong.

**Context packing.** Chunks go in in rank order until the token budget is spent,
and a chunk is either included whole or not at all — a half-chunk is a citation
that points at text the model never saw. Each source is labelled with a *marker*
(`[1]`, `[2]`) rather than a chunk id, because a model shown `chk_9f2a1c…` will
cheerfully invent one that looks just like it, whereas a small integer range is
easy to constrain and trivial to validate.

**Marker parsing.** The model's output is untrusted. Markers are extracted,
validated against the sources actually supplied, and anything out of range is
stripped from the text rather than rendered as a dead citation. A hallucinated
`[9]` when six sources were given is exactly the failure a citation UI must never
paper over.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from kb.chunking.base import split_sentences
from kb.models import AnswerSentence, Chunk, ScoredChunk, estimate_tokens

#: ``[1]``, ``[1, 2]``, ``[1][3]`` — every form a model reasonably produces.
CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

SYSTEM_PROMPT = """You answer questions using only the numbered sources provided.

Rules:
- Cite the source for every factual claim, as [1] or [2, 3], placed at the end of the sentence it supports.
- Use only what the sources say. Never add facts from your own knowledge, even if you are confident they are correct.
- If the sources do not contain the answer, say so plainly and stop. Do not guess, and do not pad the reply with related information the user did not ask for.
- If the sources disagree, say that they disagree and cite each side.
- Be direct. Lead with the answer, then the detail that supports it.
- Never cite a source number that was not provided."""

PROMPT_TEMPLATE = """Sources:

{sources}

---

Question: {query}

Answer using only the sources above, citing each factual claim as [n]."""

REFUSAL_MARKERS = (
    "do not contain",
    "does not contain",
    "no information",
    "not mentioned in the sources",
    "cannot answer",
    "can't answer",
    "the sources do not",
    "not covered by the sources",
    "i don't have enough information",
    "i do not have enough information",
)


def pack_context(
    candidates: Sequence[ScoredChunk],
    *,
    token_budget: int = 6000,
    max_chunks: int | None = None,
) -> list[Chunk]:
    """Select chunks for the prompt, best-first, within ``token_budget``.

    Whole chunks only. A truncated chunk produces a citation pointing at text the
    model was never shown, which is worse than one fewer source.
    """
    selected: list[Chunk] = []
    used = 0
    for candidate in candidates:
        if max_chunks is not None and len(selected) >= max_chunks:
            break
        cost = candidate.chunk.token_estimate or estimate_tokens(candidate.chunk.text)
        if selected and used + cost > token_budget:
            continue
        selected.append(candidate.chunk)
        used += cost
        if used >= token_budget:
            break
    return selected


def format_sources(chunks: Sequence[Chunk]) -> str:
    """Render chunks as the numbered source block the model reads.

    The position label (``p. 12``, ``Retrieval › Fusion``) is included because it
    is genuinely useful context for the model — a chunk from a section titled
    "Deprecated" should be treated differently from one titled "Current" — and
    because it makes the prompt legible when debugging a bad answer.
    """
    blocks: list[str] = []
    for marker, chunk in enumerate(chunks, start=1):
        header = f"[{marker}] {chunk.document_title}"
        position = chunk.locator.label()
        if position:
            header += f" — {position}"
        blocks.append(f"{header}\n{chunk.text.strip()}")
    return "\n\n".join(blocks)


def build_prompt(query: str, chunks: Sequence[Chunk]) -> str:
    return PROMPT_TEMPLATE.format(sources=format_sources(chunks), query=query.strip())


def parse_markers(text: str) -> list[int]:
    """Every citation marker appearing in ``text``, deduplicated, in order."""
    seen: dict[int, None] = {}
    for match in CITATION_RE.finditer(text):
        for part in match.group(1).split(","):
            part = part.strip()
            if part.isdigit():
                seen.setdefault(int(part), None)
    return list(seen)


def strip_invalid_markers(text: str, valid: set[int]) -> tuple[str, list[int]]:
    """Remove citation markers that were never supplied as sources.

    Returns the cleaned text and the invalid markers found. A hallucinated marker
    is dropped rather than rendered, because a citation chip that leads nowhere is
    worse than no chip: it looks like evidence.
    """
    invalid: list[int] = []

    def replace(match: re.Match[str]) -> str:
        numbers = [p.strip() for p in match.group(1).split(",")]
        kept = []
        for raw in numbers:
            if not raw.isdigit():
                continue
            value = int(raw)
            if value in valid:
                kept.append(value)
            else:
                invalid.append(value)
        return f"[{', '.join(str(k) for k in kept)}]" if kept else ""

    cleaned = CITATION_RE.sub(replace, text)
    # Collapse whitespace left behind by a removed marker, without touching
    # paragraph breaks.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return cleaned.strip(), sorted(set(invalid))


#: A run of citation markers at the very start of a string, e.g. "[1] [2] rest".
_LEADING_MARKERS_RE = re.compile(r"^\s*((?:\[\d+(?:\s*,\s*\d+)*\]\s*)+)")


def split_into_sentences(text: str) -> list[AnswerSentence]:
    """Split an answer into sentences with their markers and char offsets.

    Offsets are resolved by scanning forward for each sentence, so a frontend can
    highlight the exact span a verification verdict applies to.
    """
    pieces: list[str] = []
    for paragraph in text.split("\n"):
        stripped = paragraph.strip()
        if stripped:
            pieces.extend(split_sentences(stripped) or [stripped])

    pieces = _reattach_trailing_markers(pieces)

    sentences: list[AnswerSentence] = []
    cursor = 0
    for sentence in pieces:
        start = text.find(sentence, cursor)
        if start == -1:
            start = cursor
        end = start + len(sentence)
        cursor = end
        sentences.append(
            AnswerSentence(
                text=sentence,
                citation_markers=parse_markers(sentence),
                char_start=start,
                char_end=end,
            )
        )
    return sentences


def _reattach_trailing_markers(pieces: list[str]) -> list[str]:
    """Move a leading ``[n]`` run onto the previous sentence.

    Citations are written *after* the claim they support — "…defaults to 60. [1]" —
    so a naive sentence split puts the marker at the head of the *next* sentence
    and attributes the citation to the wrong claim. Since verification runs per
    sentence, that misattribution would make every verdict meaningless, so it is
    corrected here rather than worked around downstream.
    """
    out: list[str] = []
    for piece in pieces:
        match = _LEADING_MARKERS_RE.match(piece)
        if match and out:
            markers = match.group(1).strip()
            remainder = piece[match.end() :].strip()
            out[-1] = f"{out[-1]} {markers}".strip()
            if remainder:
                out.append(remainder)
            continue
        out.append(piece)
    return out


def looks_like_refusal(text: str) -> bool:
    """True when the answer says the sources do not cover the question.

    Detected rather than inferred from citation count: a well-behaved refusal
    legitimately cites nothing, and must not be scored as an unfaithful answer.
    """
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)
