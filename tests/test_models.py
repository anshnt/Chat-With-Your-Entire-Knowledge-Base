"""Locators are the load-bearing abstraction, so they get the most tests."""

from __future__ import annotations

import pytest

from kb.models import (
    Chunk,
    ChunkKind,
    GitHubLocator,
    NotionLocator,
    PdfLocator,
    SourceType,
    TextLocator,
    WebLocator,
    YouTubeLocator,
    estimate_tokens,
    parse_locator,
    with_text_fragment,
)


class TestPdfLocator:
    def test_deep_link_uses_page_fragment(self) -> None:
        locator = PdfLocator(page=12, page_count=40, file_url="/files/report.pdf")
        assert locator.deep_link() == "/files/report.pdf#page=12"

    def test_no_link_without_a_served_file(self) -> None:
        assert PdfLocator(page=3).deep_link() is None

    def test_label_includes_page_count_when_known(self) -> None:
        assert PdfLocator(page=3, page_count=10).label() == "p. 3 / 10"
        assert PdfLocator(page=3).label() == "p. 3"

    def test_pages_are_one_based(self) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 1"):
            PdfLocator(page=0)


class TestTextLocator:
    def test_deep_link_is_a_line_range(self) -> None:
        locator = TextLocator(line_start=88, line_end=104, file_path="notes.md")
        assert locator.deep_link() == "notes.md#L88-L104"

    def test_label_prefers_headings_over_line_numbers(self) -> None:
        locator = TextLocator(
            line_start=1, line_end=2, heading_path=["Architecture", "Retrieval", "Fusion"]
        )
        assert locator.label() == "Retrieval › Fusion"

    def test_label_falls_back_to_lines(self) -> None:
        assert TextLocator(line_start=5, line_end=5).label() == "line 5"
        assert TextLocator(line_start=5, line_end=9).label() == "lines 5-9"


class TestWebLocator:
    def test_deep_link_builds_a_text_fragment(self) -> None:
        locator = WebLocator(
            url="https://example.com/docs",
            quote_prefix="RRF consumes ranks",
            quote_suffix="not scores",
        )
        link = locator.deep_link()
        assert link == "https://example.com/docs#:~:text=RRF%20consumes%20ranks,not%20scores"

    def test_single_ended_fragment_when_prefix_equals_suffix(self) -> None:
        locator = WebLocator(url="https://example.com", quote_prefix="hello", quote_suffix="hello")
        assert locator.deep_link() == "https://example.com#:~:text=hello"

    def test_existing_fragment_is_replaced_not_appended(self) -> None:
        locator = WebLocator(url="https://example.com/p#section", quote_prefix="hi")
        assert locator.deep_link() == "https://example.com/p#:~:text=hi"

    def test_url_only_when_no_quote(self) -> None:
        assert WebLocator(url="https://example.com").deep_link() == "https://example.com"

    def test_label_falls_back_to_host(self) -> None:
        assert WebLocator(url="https://docs.example.com/a/b").label() == "docs.example.com"


class TestGitHubLocator:
    def test_deep_link_is_a_blob_line_range(self) -> None:
        locator = GitHubLocator(
            repo="anshnt/kb",
            ref="main",
            path="backend/kb/retrieval/fusion.py",
            line_start=10,
            line_end=20,
        )
        assert locator.deep_link() == (
            "https://github.com/anshnt/kb/blob/main/backend/kb/retrieval/fusion.py#L10-L20"
        )

    def test_single_line_anchor(self) -> None:
        locator = GitHubLocator(repo="a/b", path="f.py", line_start=7, line_end=7)
        assert locator.deep_link().endswith("#L7")

    def test_label_includes_symbol_when_known(self) -> None:
        locator = GitHubLocator(repo="a/b", path="f.py", line_start=7, line_end=9, symbol="fuse")
        assert locator.label() == "f.py:7 (fuse)"


class TestYouTubeLocator:
    def test_deep_link_seeks_to_the_start_time(self) -> None:
        locator = YouTubeLocator(video_id="dQw4w9WgXcQ", start_seconds=93.4)
        assert locator.deep_link() == "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=93s"

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(0, "0:00"), (9, "0:09"), (93, "1:33"), (600, "10:00"), (3661, "1:01:01")],
    )
    def test_label_is_a_timestamp(self, seconds: float, expected: str) -> None:
        assert YouTubeLocator(video_id="x", start_seconds=seconds).label() == expected


class TestNotionLocator:
    def test_deep_link_strips_dashes_from_page_id(self) -> None:
        locator = NotionLocator(
            line_start=1, line_end=2, notion_page_id="1a2b3c4d-5e6f-7080-9012-345678901234"
        )
        assert locator.deep_link() == "https://www.notion.so/1a2b3c4d5e6f70809012345678901234"

    def test_label_uses_page_path_when_no_headings(self) -> None:
        locator = NotionLocator(line_start=1, line_end=2, page_path=["Team", "Runbooks"])
        assert locator.label() == "Runbooks"


class TestLocatorRoundTrip:
    @pytest.mark.parametrize(
        "locator",
        [
            PdfLocator(page=2, file_url="/f.pdf"),
            TextLocator(line_start=1, line_end=3, heading_path=["A", "B"]),
            WebLocator(url="https://e.com", quote_prefix="q"),
            GitHubLocator(repo="a/b", path="f.py", line_start=1, line_end=2),
            YouTubeLocator(video_id="v", start_seconds=1.5),
            NotionLocator(line_start=1, line_end=1, page_path=["P"]),
        ],
    )
    def test_serialise_and_parse_preserves_behaviour(self, locator) -> None:
        restored = parse_locator(locator.model_dump(mode="json"))
        assert type(restored) is type(locator)
        assert restored.deep_link() == locator.deep_link()
        assert restored.label() == locator.label()

    def test_unknown_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown locator kind"):
            parse_locator({"kind": "hologram"})


class TestChunk:
    def _chunk(self, title: str, **kwargs) -> Chunk:
        return Chunk(
            document_id="doc_1",
            ordinal=0,
            text="Reciprocal Rank Fusion combines ranked lists.",
            document_title=title,
            source_type=SourceType.MARKDOWN,
            locator=TextLocator(line_start=1, line_end=2, **kwargs),
        )

    def test_hash_and_token_estimate_are_derived(self) -> None:
        chunk = self._chunk("Retrieval")
        assert len(chunk.content_hash) == 64
        assert chunk.token_estimate > 0

    def test_citation_label_combines_title_and_position(self) -> None:
        chunk = self._chunk("Architecture", heading_path=["Retrieval", "Fusion"])
        assert chunk.citation_label() == "Architecture — Retrieval › Fusion"

    def test_citation_label_strips_a_redundant_leading_title(self) -> None:
        chunk = self._chunk("Retrieval", heading_path=["Retrieval", "Fusion"])
        assert chunk.citation_label() == "Retrieval — Fusion"

    def test_citation_label_drops_a_title_that_repeats_itself(self) -> None:
        chunk = self._chunk("Retrieval", heading_path=["Retrieval"])
        assert chunk.citation_label() == "Retrieval"

    def test_chunk_kind_defaults_to_prose(self) -> None:
        assert self._chunk("x").kind is ChunkKind.PROSE


def test_estimate_tokens_scales_with_length() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("hi") >= 1


def test_with_text_fragment_percent_encodes() -> None:
    assert with_text_fragment("https://e.com/p", "a b") == "https://e.com/p#:~:text=a%20b"
