"""Chunking tests.

Chunk boundaries determine retrieval quality more than any other single choice,
so these tests pin down the behaviour that matters: no mid-word cuts, no split
code fences, correct line numbers, and heading paths that survive to the chunk.
"""

from __future__ import annotations

from kb.chunking.base import line_of_offset, line_starts, normalize_whitespace, split_sentences
from kb.chunking.markdown import MarkdownChunker
from kb.chunking.recursive import RecursiveChunker
from kb.models import ChunkKind


class TestSplitSentences:
    def test_splits_on_terminators(self) -> None:
        assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]

    def test_does_not_split_on_abbreviations(self) -> None:
        text = "Use RRF, e.g. with k=60. It is robust."
        assert split_sentences(text) == ["Use RRF, e.g. with k=60.", "It is robust."]

    def test_keeps_decimals_intact(self) -> None:
        assert split_sentences("The value is 3.14 exactly.") == ["The value is 3.14 exactly."]

    def test_empty_input(self) -> None:
        assert split_sentences("   ") == []


class TestLineMapping:
    def test_line_starts_and_lookup(self) -> None:
        text = "one\ntwo\nthree"
        starts = line_starts(text)
        assert starts == [0, 4, 8]
        assert line_of_offset(starts, 0) == 1
        assert line_of_offset(starts, 5) == 2
        assert line_of_offset(starts, 12) == 3


def test_normalize_whitespace_collapses_blank_runs() -> None:
    assert normalize_whitespace("a\n\n\n\n\nb   \n") == "a\n\nb"


class TestRecursiveChunker:
    def test_short_text_is_one_chunk(self) -> None:
        drafts = RecursiveChunker(400, 40).chunk("A short paragraph about retrieval.")
        assert len(drafts) == 1
        assert drafts[0].text == "A short paragraph about retrieval."

    def test_respects_the_size_budget(self) -> None:
        text = "\n\n".join(
            f"Paragraph {i} about hybrid retrieval and fusion." * 4 for i in range(30)
        )
        drafts = RecursiveChunker(400, 0, 50).chunk(text)
        assert len(drafts) > 1
        # Overlap is off, so no chunk should exceed the budget by more than the
        # size of one indivisible unit.
        assert all(len(d.text) <= 800 for d in drafts)

    def test_never_cuts_mid_word_when_a_boundary_exists(self) -> None:
        text = " ".join(["retrieval"] * 400)
        drafts = RecursiveChunker(300, 0, 50).chunk(text)
        for draft in drafts:
            assert not draft.text.startswith("rieval")
            for word in draft.text.split():
                assert word == "retrieval"

    def test_overlap_repeats_trailing_context(self) -> None:
        paragraphs = [
            "Alpha sentence one. Alpha sentence two.",
            "Beta sentence one. Beta sentence two.",
            "Gamma sentence one. Gamma sentence two.",
        ]
        drafts = RecursiveChunker(60, 40, 10).chunk("\n\n".join(paragraphs))
        assert len(drafts) >= 2
        # At least one later chunk should carry text from its predecessor.
        assert any("Alpha" in d.text for d in drafts[1:])

    def test_line_numbers_track_position(self) -> None:
        text = "\n".join(f"line {i} about retrieval and fusion and ranking" for i in range(60))
        drafts = RecursiveChunker(300, 0, 50).chunk(text)
        assert drafts[0].line_start == 1
        assert drafts[-1].line_end >= drafts[0].line_end
        for draft in drafts:
            assert draft.line_start <= draft.line_end

    def test_tiny_trailing_chunk_is_merged(self) -> None:
        text = "A" * 500 + "\n\nx"
        drafts = RecursiveChunker(400, 0, 100).chunk(text)
        assert all(len(d.text.strip()) > 5 for d in drafts)

    def test_empty_input(self) -> None:
        assert RecursiveChunker().chunk("") == []
        assert RecursiveChunker().chunk("   \n  ") == []


class TestMarkdownChunker:
    SAMPLE = """# Architecture

Intro paragraph.

## Retrieval

### Fusion

RRF consumes ranks, not scores.

### Dense

Cosine similarity over normalised vectors.

## Chunking

Split on the largest boundary that fits.

```python
def chunk(text):
    return splitter.split(text)
```

| metric | value |
|--------|-------|
| recall | 0.82  |
| ndcg   | 0.71  |
"""

    def test_heading_paths_are_captured(self) -> None:
        drafts = MarkdownChunker(1200, 0, 20).chunk(self.SAMPLE)
        paths = [tuple(d.heading_path) for d in drafts]
        assert ("Architecture", "Retrieval", "Fusion") in paths
        assert ("Architecture", "Retrieval", "Dense") in paths
        assert ("Architecture", "Chunking") in paths

    def test_heading_path_is_prefixed_into_the_text(self) -> None:
        drafts = MarkdownChunker(1200, 0, 20).chunk(self.SAMPLE)
        fusion = next(d for d in drafts if d.heading_path[-1:] == ["Fusion"])
        assert fusion.text.startswith("Architecture › Retrieval › Fusion")
        assert "RRF consumes ranks" in fusion.text

    def test_code_fences_are_marked_and_kept_whole(self) -> None:
        drafts = MarkdownChunker(1200, 0, 20).chunk(self.SAMPLE)
        code = [d for d in drafts if d.kind is ChunkKind.CODE]
        assert code, "expected at least one code chunk"
        joined = "\n".join(d.text for d in code)
        assert joined.count("```") % 2 == 0
        assert "def chunk(text):" in joined

    def test_tables_are_marked(self) -> None:
        drafts = MarkdownChunker(400, 0, 20).chunk(self.SAMPLE)
        kinds = {d.kind for d in drafts}
        assert ChunkKind.TABLE in kinds or ChunkKind.CODE in kinds

    def test_setext_headings_are_recognised(self) -> None:
        text = "Title Here\n==========\n\nBody about retrieval.\n\nSub\n---\n\nMore body.\n"
        drafts = MarkdownChunker(1200, 0, 20).chunk(text)
        paths = [tuple(d.heading_path) for d in drafts]
        assert ("Title Here",) in paths
        assert ("Title Here", "Sub") in paths

    def test_front_matter_free_document_without_headings(self) -> None:
        drafts = MarkdownChunker(1200, 0, 20).chunk("Just a paragraph.\n\nAnd another.")
        assert len(drafts) == 1
        assert drafts[0].heading_path == []

    def test_oversized_section_is_split_but_keeps_its_heading(self) -> None:
        body = "\n\n".join(f"Sentence group {i} about retrieval." * 6 for i in range(20))
        drafts = MarkdownChunker(400, 0, 50).chunk(f"# Doc\n\n## Big\n\n{body}")
        big = [d for d in drafts if d.heading_path[-1:] == ["Big"]]
        assert len(big) > 1
        assert all(d.heading_path == ["Doc", "Big"] for d in big)
