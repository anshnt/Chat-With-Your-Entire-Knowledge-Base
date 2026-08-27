"""PDF connector tests over real PDF bytes."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.pdf_fixture import build_pdf

from kb.config import Settings
from kb.errors import IngestionError
from kb.ingest.pdf import PDFConnector, _clean_page, _strip_boilerplate
from kb.knowledge_base import KnowledgeBase
from kb.models import PdfLocator, SourceType


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    """A three-page PDF with a running header and distinct per-page content."""
    pages = [
        [
            "ACME CONFIDENTIAL",
            "Hybrid Retrieval Report",
            "Hybrid search combines BM25 lexical matching with dense",
            "vector retrieval over the same corpus.",
            "1",
        ],
        [
            "ACME CONFIDENTIAL",
            "Reciprocal Rank Fusion merges ranked lists using ranks",
            "rather than raw scores. The damping constant defaults to 60.",
            "2",
        ],
        [
            "ACME CONFIDENTIAL",
            "Cross-encoder reranking scores each query-document pair",
            "jointly, which is accurate but too slow for a whole corpus.",
            "3",
        ],
    ]
    path = tmp_path / "report.pdf"
    path.write_bytes(build_pdf(pages))
    return path


class TestCleanPage:
    def test_repairs_hyphenated_line_breaks(self) -> None:
        assert "retrieval" in _clean_page("retriev-\nal is hard")

    def test_normalises_ligatures(self) -> None:
        assert _clean_page("eﬃcient ﬁlter") == "efficient filter" or "fi" in _clean_page("ﬁlter")

    def test_drops_bare_page_numbers(self) -> None:
        assert _clean_page("Real content here.\n12") == "Real content here."

    def test_rejoins_soft_wrapped_lines(self) -> None:
        cleaned = _clean_page("The quick brown\nfox jumps over.")
        assert cleaned == "The quick brown fox jumps over."


class TestStripBoilerplate:
    def test_removes_lines_repeated_across_pages(self) -> None:
        pages = [f"ACME CONFIDENTIAL\nUnique body {i}\nFooter 2024" for i in range(6)]
        cleaned = _strip_boilerplate(pages)
        assert all("ACME CONFIDENTIAL" not in page for page in cleaned)
        assert all("Footer 2024" not in page for page in cleaned)
        assert all(f"Unique body {i}" in cleaned[i] for i in range(6))

    def test_short_documents_are_left_alone(self) -> None:
        """With three pages there is not enough evidence to call anything boilerplate."""
        pages = ["Header\nBody one", "Header\nBody two"]
        assert _strip_boilerplate(pages) == pages


class TestPDFConnector:
    def test_can_handle_only_pdfs(self, tmp_settings: Settings) -> None:
        connector = PDFConnector(tmp_settings)
        assert connector.can_handle("a.pdf")
        assert connector.can_handle("/x/y/A.PDF")
        assert not connector.can_handle("a.md")
        assert not connector.can_handle("https://example.com/a.pdf")

    def test_one_segment_per_page(self, tmp_settings: Settings, pdf_path: Path) -> None:
        parsed = next(iter(PDFConnector(tmp_settings).parse(str(pdf_path))))
        assert len(parsed.segments) == 3
        assert parsed.metadata["page_count"] == 3
        assert parsed.source_type is SourceType.PDF

    def test_running_header_is_stripped(self, tmp_settings: Settings, pdf_path: Path) -> None:
        parsed = next(iter(PDFConnector(tmp_settings).parse(str(pdf_path))))
        assert all("CONFIDENTIAL" not in s.text for s in parsed.segments)

    def test_garbage_bytes_raise_ingestion_error(
        self, tmp_settings: Settings, tmp_path: Path
    ) -> None:
        path = tmp_path / "broken.pdf"
        path.write_bytes(b"%PDF-1.4\ntruncated garbage")
        with pytest.raises(IngestionError):
            list(PDFConnector(tmp_settings).parse(str(path)))

    def test_missing_file_raises(self, tmp_settings: Settings) -> None:
        with pytest.raises(IngestionError, match="not a file"):
            list(PDFConnector(tmp_settings).parse("/nope/missing.pdf"))


class TestPDFIngestion:
    def test_chunks_carry_page_locators(self, empty_kb: KnowledgeBase, pdf_path: Path) -> None:
        report = empty_kb.ingest(str(pdf_path))
        assert report.documents_created == 1
        assert not report.errors

        chunks = empty_kb.document_chunks(report.documents[0].id)
        assert chunks
        for chunk in chunks:
            assert isinstance(chunk.locator, PdfLocator)
            assert 1 <= chunk.locator.page <= 3
            assert chunk.locator.page_count == 3

    def test_a_chunk_never_straddles_two_pages(
        self, empty_kb: KnowledgeBase, pdf_path: Path
    ) -> None:
        """Chunking inside a page is what keeps every citation unambiguous."""
        report = empty_kb.ingest(str(pdf_path))
        chunks = empty_kb.document_chunks(report.documents[0].id)
        # Content unique to page 2 must only ever appear in a page-2 chunk.
        for chunk in chunks:
            if "damping constant" in chunk.text:
                assert chunk.locator.page == 2
            if "Cross-encoder" in chunk.text:
                assert chunk.locator.page == 3

    def test_citation_links_to_the_right_page(
        self, empty_kb: KnowledgeBase, pdf_path: Path
    ) -> None:
        empty_kb.ingest(str(pdf_path))
        result = empty_kb.search("what does the damping constant default to", top_k=3)
        hit = next(r for r in result.results if "damping constant" in r.chunk.text)
        assert hit.chunk.deep_link() == "/files/report.pdf#page=2"
        assert hit.chunk.citation_label().endswith("p. 2 / 3")

    def test_search_finds_pdf_content(self, empty_kb: KnowledgeBase, pdf_path: Path) -> None:
        empty_kb.ingest(str(pdf_path))
        result = empty_kb.search("cross-encoder reranking", top_k=3)
        assert any("Cross-encoder" in r.chunk.text for r in result.results)

    def test_scanned_pdf_gives_an_actionable_error(
        self, empty_kb: KnowledgeBase, tmp_path: Path
    ) -> None:
        """A PDF with no text layer should say to run OCR, not fail cryptically."""
        path = tmp_path / "scan.pdf"
        path.write_bytes(build_pdf([[]]))
        report = empty_kb.ingest(str(path))
        assert report.documents_created == 0
        assert report.errors
        assert "OCR" in report.errors[0]["error"]
