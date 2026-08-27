"""Ingestion tests: connectors, deduplication, and locator correctness."""

from __future__ import annotations

from pathlib import Path

import pytest

from kb.config import Settings
from kb.errors import UnsupportedSourceError
from kb.ingest.base import strip_front_matter, title_from_markdown, title_from_path
from kb.ingest.registry import default_registry
from kb.knowledge_base import KnowledgeBase
from kb.models import ChunkKind, SourceType, TextLocator


class TestHelpers:
    def test_strip_front_matter_parses_yaml(self) -> None:
        body, meta = strip_front_matter("---\ntitle: Design\nauthor: ansh\n---\n\n# Design\n")
        assert body.startswith("# Design")
        assert meta == {"title": "Design", "author": "ansh"}

    def test_no_front_matter_is_untouched(self) -> None:
        body, meta = strip_front_matter("# Design\n")
        assert body == "# Design\n"
        assert meta == {}

    def test_malformed_front_matter_is_not_fatal(self) -> None:
        body, meta = strip_front_matter("---\n: : :\n\tbad\n---\nbody\n")
        assert "body" in body
        assert isinstance(meta, dict)

    def test_title_from_markdown_prefers_h1(self) -> None:
        assert title_from_markdown("# Real Title\n\nbody", "fallback") == "Real Title"

    def test_title_from_markdown_falls_back(self) -> None:
        assert title_from_markdown("no heading here", "Fallback") == "Fallback"

    def test_title_from_path_humanises(self) -> None:
        assert title_from_path(Path("my_design-doc.md")) == "My Design Doc"


class TestRegistry:
    def test_routes_by_extension(self, tmp_settings: Settings) -> None:
        registry = default_registry(tmp_settings)
        assert registry.resolve("notes.md").name == "markdown"
        assert registry.resolve("report.pdf").name == "pdf"
        assert registry.resolve("log.txt").name == "text"
        assert registry.resolve("inline:x").name == "inline"

    def test_unknown_extension_is_rejected(self, tmp_settings: Settings) -> None:
        with pytest.raises(UnsupportedSourceError):
            default_registry(tmp_settings).resolve("binary.exe")

    def test_directory_walk_skips_noise(self, tmp_settings: Settings, tmp_path: Path) -> None:
        root = tmp_path / "project"
        (root / "docs").mkdir(parents=True)
        (root / "node_modules" / "pkg").mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "docs" / "guide.md").write_text("# Guide\n\nbody")
        (root / "node_modules" / "pkg" / "readme.md").write_text("# Vendored\n\nbody")
        (root / ".git" / "notes.md").write_text("# Git\n\nbody")
        (root / "top.txt").write_text("top level")

        found = sorted(Path(p).name for p in default_registry(tmp_settings).expand(str(root)))
        assert found == ["guide.md", "top.txt"]

    def test_glob_expansion(self, tmp_settings: Settings, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("# A\n\nbody")
        (tmp_path / "b.md").write_text("# B\n\nbody")
        (tmp_path / "c.pdf").write_bytes(b"not a real pdf")
        found = sorted(
            Path(p).name for p in default_registry(tmp_settings).expand(str(tmp_path / "*.md"))
        )
        assert found == ["a.md", "b.md"]

    def test_unmatched_glob_raises(self, tmp_settings: Settings, tmp_path: Path) -> None:
        with pytest.raises(UnsupportedSourceError, match="matched no ingestable files"):
            list(default_registry(tmp_settings).expand(str(tmp_path / "*.md")))

    def test_urls_pass_through_untouched(self, tmp_settings: Settings) -> None:
        registry = default_registry(tmp_settings)
        assert list(registry.expand("https://example.com/docs")) == ["https://example.com/docs"]


class TestMarkdownIngestion:
    def test_creates_document_and_chunks(self, empty_kb: KnowledgeBase, tmp_path: Path) -> None:
        path = tmp_path / "design.md"
        path.write_text(
            "---\ntitle: Design Notes\nauthor: ansh\n---\n\n"
            "# Design Notes\n\n## Retrieval\n\nHybrid search fuses BM25 and dense results.\n"
        )
        report = empty_kb.ingest(str(path))
        assert report.documents_created == 1
        assert report.chunks_created >= 1

        document = report.documents[0]
        assert document.title == "Design Notes"
        assert document.author == "ansh"
        assert document.source_type is SourceType.MARKDOWN

    def test_front_matter_is_not_indexed_as_prose(
        self, empty_kb: KnowledgeBase, tmp_path: Path
    ) -> None:
        path = tmp_path / "d.md"
        path.write_text("---\ndraft: false\nsecret_key: abc123\n---\n\n# Doc\n\nReal body text.\n")
        empty_kb.ingest(str(path))
        chunks = empty_kb.document_chunks(empty_kb.documents()[0].id)
        assert all("secret_key" not in c.text for c in chunks)

    def test_locators_carry_heading_paths(self, empty_kb: KnowledgeBase, tmp_path: Path) -> None:
        path = tmp_path / "arch.md"
        path.write_text("# Arch\n\n## Retrieval\n\n### Fusion\n\nRRF uses ranks, not scores.\n")
        empty_kb.ingest(str(path))
        chunks = empty_kb.document_chunks(empty_kb.documents()[0].id)
        fusion = next(c for c in chunks if "RRF uses ranks" in c.text)
        assert isinstance(fusion.locator, TextLocator)
        assert fusion.locator.heading_path == ["Arch", "Retrieval", "Fusion"]
        assert fusion.deep_link().endswith(
            f"#L{fusion.locator.line_start}-L{fusion.locator.line_end}"
        )

    def test_code_blocks_are_marked(self, empty_kb: KnowledgeBase, tmp_path: Path) -> None:
        path = tmp_path / "code.md"
        path.write_text(
            "# Code\n\n## Example\n\n```python\ndef fuse(a, b):\n    return a + b\n```\n"
        )
        empty_kb.ingest(str(path))
        chunks = empty_kb.document_chunks(empty_kb.documents()[0].id)
        assert any(c.kind is ChunkKind.CODE for c in chunks)


class TestTextIngestion:
    def test_plain_text_file(self, empty_kb: KnowledgeBase, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("A note about hybrid retrieval and BM25.\n")
        report = empty_kb.ingest(str(path))
        assert report.documents_created == 1
        assert report.documents[0].source_type is SourceType.TEXT

    def test_inline_text(self, empty_kb: KnowledgeBase) -> None:
        report = empty_kb.ingest_text(
            "# Pasted\n\nSome content about reranking.", title="Pasted Doc"
        )
        assert report.documents_created == 1
        assert report.documents[0].title == "Pasted Doc"

    def test_inline_markdown_is_detected(self, empty_kb: KnowledgeBase) -> None:
        report = empty_kb.ingest_text("# Heading\n\n## Sub\n\nbody", title="MD")
        assert report.documents[0].source_type is SourceType.MARKDOWN

    def test_odd_encoding_does_not_abort(self, empty_kb: KnowledgeBase, tmp_path: Path) -> None:
        path = tmp_path / "legacy.txt"
        path.write_bytes("Naïve café — em dash".encode("cp1252"))
        report = empty_kb.ingest(str(path))
        assert report.documents_created == 1
        assert not report.errors


class TestDeduplication:
    def test_reingesting_unchanged_content_is_a_noop(
        self, empty_kb: KnowledgeBase, corpus_dir: Path
    ) -> None:
        first = empty_kb.ingest(str(corpus_dir))
        assert first.documents_created == 3

        second = empty_kb.ingest(str(corpus_dir))
        assert second.documents_created == 0
        assert second.duplicates_skipped == 3
        assert empty_kb.stats().n_documents == 3

    def test_changed_content_creates_a_new_document(
        self, empty_kb: KnowledgeBase, tmp_path: Path
    ) -> None:
        path = tmp_path / "living.md"
        path.write_text("# Living\n\nversion one of the body text.")
        empty_kb.ingest(str(path))

        path.write_text("# Living\n\nversion two of the body text, revised.")
        report = empty_kb.ingest(str(path))
        assert report.documents_created == 1

    def test_identical_chunks_within_a_document_are_dropped(
        self, empty_kb: KnowledgeBase, tmp_path: Path
    ) -> None:
        repeated = "\n\n".join(["The exact same paragraph repeated verbatim."] * 5)
        path = tmp_path / "repeat.md"
        path.write_text(f"# Repeat\n\n{repeated}\n")
        empty_kb.ingest(str(path))
        chunks = empty_kb.document_chunks(empty_kb.documents()[0].id)
        bodies = [c.text for c in chunks]
        assert len(bodies) == len(set(bodies))


class TestEmbeddingBackfill:
    def test_ingest_embeds_by_default(self, empty_kb: KnowledgeBase, corpus_dir: Path) -> None:
        empty_kb.ingest(str(corpus_dir))
        stats = empty_kb.stats()
        assert stats.n_embedded == stats.n_chunks

    def test_no_embed_leaves_bm25_working(self, empty_kb: KnowledgeBase, corpus_dir: Path) -> None:
        """Ingestion must degrade gracefully: lexical search before embedding."""
        empty_kb.ingest(str(corpus_dir), embed=False)
        stats = empty_kb.stats()
        assert stats.n_chunks > 0
        assert stats.n_embedded == 0

        result = empty_kb.search("reciprocal rank fusion", strategy="lexical")
        assert result.results

    def test_backfill_is_idempotent(self, empty_kb: KnowledgeBase, corpus_dir: Path) -> None:
        empty_kb.ingest(str(corpus_dir), embed=False)
        first = empty_kb.embed_pending()
        assert first > 0
        assert empty_kb.embed_pending() == 0

    def test_rebuild_reembeds_everything(self, empty_kb: KnowledgeBase, corpus_dir: Path) -> None:
        empty_kb.ingest(str(corpus_dir))
        total = empty_kb.stats().n_chunks
        assert empty_kb.reembed() == total


class TestErrorHandling:
    def test_missing_file_is_reported_not_raised(self, empty_kb: KnowledgeBase) -> None:
        report = empty_kb.ingest("/nonexistent/path/file.md")
        assert report.documents_created == 0
        assert report.errors

    def test_one_bad_file_does_not_abort_a_directory(
        self, empty_kb: KnowledgeBase, tmp_path: Path
    ) -> None:
        root = tmp_path / "mixed"
        root.mkdir()
        (root / "good.md").write_text("# Good\n\nUseful content about retrieval.")
        (root / "broken.pdf").write_bytes(b"%PDF-1.4 truncated garbage")

        report = empty_kb.ingest(str(root))
        assert report.documents_created == 1
        assert len(report.errors) == 1
