"""Tests for the web, GitHub, YouTube and Notion connectors.

No network: the web connector is driven through an httpx mock transport, GitHub
through a local checkout, and YouTube through its pure grouping functions. That
is deliberate — a connector test that needs the internet does not run in CI, so
it stops being run at all.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import httpx
import pytest

from kb.config import Settings
from kb.errors import IngestionError
from kb.ingest.github import (
    CodeChunker,
    GitHubConnector,
    is_indexable,
    parse_repo_source,
)
from kb.ingest.notion import (
    NotionConnector,
    csv_to_markdown,
    split_notion_filename,
    strip_notion_artifacts,
)
from kb.ingest.registry import default_registry
from kb.ingest.web import (
    WebConnector,
    canonical_url,
    extract_links,
    extract_title,
    fragment_anchors,
    select_content,
    strip_chrome,
    to_markdown,
)
from kb.ingest.youtube import (
    Cue,
    clean_cue,
    extract_video_id,
    group_into_windows,
)
from kb.knowledge_base import KnowledgeBase
from kb.models import GitHubLocator, NotionLocator, SourceType, WebLocator


def soup_of(html: str):
    from bs4 import BeautifulSoup

    return BeautifulSoup(html, "lxml")


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #


class TestRouting:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("https://example.com/docs", "web"),
            ("http://example.com", "web"),
            ("https://github.com/a/b", "github"),
            ("https://github.com/a/b/tree/main/src", "github"),
            ("https://github.com/a/b/blob/main/src/f.py", "github"),
            ("gh:anshnt/kb", "github"),
            ("anshnt/kb", "github"),
            ("https://youtu.be/dQw4w9WgXcQ", "youtube"),
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "youtube"),
            ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "youtube"),
            ("yt:dQw4w9WgXcQ", "youtube"),
            ("notes.md", "markdown"),
            ("report.pdf", "pdf"),
            ("log.txt", "text"),
        ],
    )
    def test_sources_reach_the_right_connector(self, source: str, expected: str) -> None:
        """The web connector claims any http(s) URL, so precedence matters."""
        registry = default_registry(Settings())
        assert registry.resolve(source).name == expected


# --------------------------------------------------------------------------- #
# web
# --------------------------------------------------------------------------- #

ARTICLE_HTML = """<!doctype html>
<html lang="en">
<head>
  <title>Hybrid Retrieval — Acme Docs</title>
  <meta name="author" content="ansh">
</head>
<body>
  <nav class="site-nav"><a href="/">Home</a><a href="/docs">Docs</a></nav>
  <div class="cookie-consent">We use cookies. <button>Accept</button></div>
  <article>
    <h1>Hybrid Retrieval</h1>
    <p>Hybrid search combines BM25 lexical matching with dense vector retrieval.</p>
    <h2>Fusion</h2>
    <p>Reciprocal Rank Fusion merges ranked lists using <code>ranks</code> rather than scores.</p>
    <p>The damping constant defaults to 60.</p>
    <ul><li>Lexical finds identifiers</li><li>Dense finds paraphrase</li></ul>
    <pre>def fuse(a, b):
    return a + b</pre>
    <table><tr><th>metric</th><th>value</th></tr><tr><td>recall</td><td>0.82</td></tr></table>
  </article>
  <aside class="related"><a href="/other">Other articles</a></aside>
  <footer>© Acme</footer>
  <script>console.log('tracking');</script>
</body>
</html>"""


class TestWebExtraction:
    def test_canonical_url_drops_fragment_and_trailing_slash(self) -> None:
        assert canonical_url("https://e.com/a/#frag") == "https://e.com/a"
        assert canonical_url("https://e.com/") == "https://e.com/"

    def test_title_prefers_h1_over_title_tag(self) -> None:
        assert extract_title(soup_of(ARTICLE_HTML)) == "Hybrid Retrieval"

    def test_title_tag_is_split_from_the_site_name(self) -> None:
        html = "<html><head><title>Article — Site Name</title></head><body></body></html>"
        assert extract_title(soup_of(html)) == "Article"

    def test_content_selection_finds_the_article(self) -> None:
        container = select_content(soup_of(ARTICLE_HTML))
        assert container.name == "article"

    def test_chrome_is_stripped(self) -> None:
        """Navigation and banners match every query weakly and none strongly."""
        soup = soup_of(ARTICLE_HTML)
        strip_chrome(soup)
        text = soup.get_text(" ", strip=True)
        assert "cookies" not in text.lower()
        assert "tracking" not in text
        assert "© Acme" not in text
        assert "Other articles" not in text
        assert "Hybrid search combines" in text

    def test_density_fallback_prefers_prose_over_navigation(self) -> None:
        """No semantic container: text density has to pick the article."""
        html = (
            "<body><div id='wrap'>"
            "<div class='links'>"
            + "".join(f"<a href='/{i}'>Link {i}</a>" for i in range(60))
            + "</div>"
            "<div class='body-copy'><p>" + ("Real prose about retrieval. " * 40) + "</p></div>"
            "</div></body>"
        )
        container = select_content(soup_of(html))
        assert "Real prose about retrieval" in container.get_text()

    def test_markdown_conversion_keeps_structure(self) -> None:
        """Heading structure is exactly as useful here as in a Markdown file."""
        markdown = to_markdown(select_content(soup_of(ARTICLE_HTML)))
        assert "# Hybrid Retrieval" in markdown
        assert "## Fusion" in markdown
        assert "`ranks`" in markdown
        assert "- Lexical finds identifiers" in markdown
        assert "```" in markdown
        assert "| metric | value |" in markdown

    def test_links_are_absolute_and_filtered(self) -> None:
        html = (
            "<body><a href='/docs'>a</a><a href='#x'>b</a><a href='mailto:x@y'>c</a>"
            "<a href='https://other.com/p'>d</a><a href='/style.css'>e</a></body>"
        )
        links = extract_links(soup_of(html), "https://e.com/base")
        assert "https://e.com/docs" in links
        assert "https://other.com/p" in links
        assert not any("mailto" in link or ".css" in link for link in links)
        assert not any(link.endswith("#x") for link in links)


class TestFragmentAnchors:
    def test_produces_a_prefix_and_suffix(self) -> None:
        prefix, suffix = fragment_anchors(
            "The damping constant defaults to sixty in the standard formulation today", words=3
        )
        assert prefix == "The damping constant"
        assert suffix == "standard formulation today"

    def test_short_text_uses_the_same_anchor_for_both(self) -> None:
        prefix, suffix = fragment_anchors("two words", words=6)
        assert prefix == suffix == "two words"

    def test_markdown_syntax_is_stripped(self) -> None:
        """The fragment must match the rendered page, which has no markup."""
        prefix, _ = fragment_anchors("## Heading\n\n**bold** text here", words=4)
        assert "#" not in prefix
        assert "*" not in prefix

    def test_empty_text(self) -> None:
        assert fragment_anchors("") == ("", "")


class TestWebConnectorFetching:
    def _connector_with(self, settings: Settings, pages: dict[str, str]) -> WebConnector:
        """A connector whose httpx client is backed by a mock transport."""

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url.endswith("/robots.txt"):
                return httpx.Response(404)
            body = pages.get(url) or pages.get(url.rstrip("/"))
            if body is None:
                return httpx.Response(404)
            return httpx.Response(200, text=body, headers={"content-type": "text/html"})

        connector = WebConnector(settings)
        transport = httpx.MockTransport(handler)
        original = httpx.Client

        class PatchedClient(original):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs) -> None:
                kwargs["transport"] = transport
                super().__init__(*args, **kwargs)

        connector._client_cls = PatchedClient  # type: ignore[attr-defined]
        return connector

    def test_fetches_and_extracts_a_page(self, monkeypatch, tmp_path: Path) -> None:
        settings = Settings(data_dir=tmp_path)
        pages = {"https://example.com/docs": ARTICLE_HTML}

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url.endswith("/robots.txt"):
                return httpx.Response(404)
            body = pages.get(url.rstrip("/"))
            if body is None:
                return httpx.Response(404)
            return httpx.Response(200, text=body, headers={"content-type": "text/html"})

        transport = httpx.MockTransport(handler)
        real_client = httpx.Client
        monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=transport, **kw))

        documents = list(WebConnector(settings).parse("https://example.com/docs"))
        assert len(documents) == 1
        document = documents[0]
        assert document.title == "Hybrid Retrieval"
        assert document.source_type is SourceType.WEB
        assert document.author == "ansh"
        assert document.language == "en"
        assert "Reciprocal Rank Fusion" in document.raw_text
        assert "cookies" not in document.raw_text.lower()

    def test_robots_txt_is_respected(self, monkeypatch, tmp_path: Path) -> None:
        """A crawler that ignores robots.txt is one nobody should run."""

        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url).endswith("/robots.txt"):
                return httpx.Response(200, text="User-agent: *\nDisallow: /private")
            return httpx.Response(200, text=ARTICLE_HTML, headers={"content-type": "text/html"})

        transport = httpx.MockTransport(handler)
        real_client = httpx.Client
        monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=transport, **kw))
        with pytest.raises(IngestionError, match="no readable content"):
            list(WebConnector(Settings(data_dir=tmp_path)).parse("https://example.com/private/x"))

    def test_non_html_is_skipped(self, monkeypatch, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url).endswith("/robots.txt"):
                return httpx.Response(404)
            return httpx.Response(200, text="{}", headers={"content-type": "application/json"})

        transport = httpx.MockTransport(handler)
        real_client = httpx.Client
        monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=transport, **kw))
        with pytest.raises(IngestionError):
            list(WebConnector(Settings(data_dir=tmp_path)).parse("https://example.com/data.json"))

    def test_crawl_follows_same_origin_links_only(self, monkeypatch, tmp_path: Path) -> None:
        index = (
            "<html><body><article><h1>Index</h1>"
            "<p>" + ("Index prose about retrieval. " * 12) + "</p>"
            "<a href='/page-two'>two</a><a href='https://other.com/p'>ext</a>"
            "</article></body></html>"
        )
        page_two = (
            "<html><body><article><h1>Page Two</h1>"
            "<p>" + ("Second page prose about fusion. " * 12) + "</p></article></body></html>"
        )
        pages = {
            "https://example.com/index": index,
            "https://example.com/page-two": page_two,
        }
        fetched: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url).rstrip("/")
            if url.endswith("/robots.txt"):
                return httpx.Response(404)
            fetched.append(url)
            body = pages.get(url)
            if body is None:
                return httpx.Response(404)
            return httpx.Response(200, text=body, headers={"content-type": "text/html"})

        transport = httpx.MockTransport(handler)
        real_client = httpx.Client
        monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=transport, **kw))

        documents = list(
            WebConnector(Settings(data_dir=tmp_path)).parse(
                "https://example.com/index", crawl=True, max_pages=10, max_depth=1
            )
        )
        assert {d.title for d in documents} == {"Index", "Page Two"}
        assert not any("other.com" in url for url in fetched)

    def test_locators_carry_text_fragments(self, monkeypatch, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url).endswith("/robots.txt"):
                return httpx.Response(404)
            return httpx.Response(200, text=ARTICLE_HTML, headers={"content-type": "text/html"})

        transport = httpx.MockTransport(handler)
        real_client = httpx.Client
        monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=transport, **kw))

        settings = Settings(data_dir=tmp_path / "data", embedding_dim=128, chunk_size=400)
        knowledge_base = KnowledgeBase(settings)
        report = knowledge_base.ingest("https://example.com/docs")
        assert report.documents_created == 1, report.errors

        chunks = knowledge_base.document_chunks(report.documents[0].id)
        assert chunks
        for chunk in chunks:
            assert isinstance(chunk.locator, WebLocator)
            link = chunk.deep_link()
            assert link is not None
            assert link.startswith("https://example.com/docs")
        assert any("#:~:text=" in (c.deep_link() or "") for c in chunks)
        knowledge_base.close()


# --------------------------------------------------------------------------- #
# github
# --------------------------------------------------------------------------- #


class TestParseRepoSource:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("anshnt/kb", ("anshnt", "kb", None, None)),
            ("gh:anshnt/kb", ("anshnt", "kb", None, None)),
            ("https://github.com/anshnt/kb", ("anshnt", "kb", None, None)),
            ("https://github.com/anshnt/kb.git", ("anshnt", "kb", None, None)),
            ("https://github.com/anshnt/kb/tree/main", ("anshnt", "kb", "main", None)),
            (
                "https://github.com/anshnt/kb/tree/main/backend",
                ("anshnt", "kb", "main", "backend"),
            ),
        ],
    )
    def test_parses_every_form(self, source: str, expected: tuple) -> None:
        assert parse_repo_source(source) == expected

    def test_unparseable_source_raises(self) -> None:
        with pytest.raises(IngestionError, match="could not parse"):
            parse_repo_source("not a repo at all!!")


class TestIsIndexable:
    @pytest.mark.parametrize(
        "path",
        [
            "package-lock.json",
            "yarn.lock",
            "Cargo.lock",
            "node_modules/pkg/index.js",
            "vendor/lib/x.go",
            "dist/bundle.min.js",
            "src/bundle.js.map",
            "__pycache__/x.pyc",
            "image.png",
            ".hidden.py",
        ],
    )
    def test_generated_and_vendored_paths_are_excluded(self, path: str) -> None:
        """A 2 MB lock file costs real money and answers no questions."""
        assert not is_indexable(path, 1000, 512_000)

    @pytest.mark.parametrize(
        "path", ["src/main.py", "README.md", "pkg/server.go", "config.yaml", "pyproject.toml"]
    )
    def test_source_and_docs_are_included(self, path: str) -> None:
        assert is_indexable(path, 1000, 512_000)

    def test_oversized_files_are_excluded(self) -> None:
        assert not is_indexable("src/generated.py", 900_000, 512_000)


class TestCodeChunker:
    PYTHON = '''"""Fusion module."""

import math


def reciprocal_rank_fusion(rankings, k=60):
    """Fuse ranked lists using ranks."""
    return {}


class WeightedFusion:
    """Weighted sum after min-max normalisation."""

    def __init__(self, weight=0.4):
        self.weight = weight


def multiline_docstring_case():
    """
    A definition inside a docstring is a string, not a declaration.
    def not_a_real_definition(): pass
    """
    return None
'''

    def test_splits_at_top_level_definitions(self) -> None:
        drafts = CodeChunker(1600).chunk(self.PYTHON)
        symbols = [d.metadata.get("symbol") for d in drafts]
        assert "reciprocal_rank_fusion" in symbols
        assert "WeightedFusion" in symbols
        assert "multiline_docstring_case" in symbols

    def test_a_one_line_docstring_does_not_suppress_later_definitions(self) -> None:
        """The bug this guards: `\"\"\"x\"\"\"` contains two delimiters, so treating
        it as a fence opener hid every definition in the rest of the file."""
        symbols = [d.metadata.get("symbol") for d in CodeChunker(1600).chunk(self.PYTHON)]
        assert "reciprocal_rank_fusion" in symbols

    def test_definitions_inside_a_docstring_are_ignored(self) -> None:
        symbols = [d.metadata.get("symbol") for d in CodeChunker(1600).chunk(self.PYTHON)]
        assert "not_a_real_definition" not in symbols

    def test_line_numbers_are_exact(self) -> None:
        """Line numbers are the address; an off-by-one is a broken permalink."""
        lines = self.PYTHON.split("\n")
        for draft in CodeChunker(1600).chunk(self.PYTHON):
            assert draft.line_start >= 1
            assert draft.line_end <= len(lines)
            first_line = draft.text.split("\n")[0]
            assert lines[draft.line_start - 1].strip() == first_line.strip()

    def test_nested_definitions_are_not_boundaries(self) -> None:
        drafts = CodeChunker(1600).chunk(self.PYTHON)
        cls = next(d for d in drafts if d.metadata.get("symbol") == "WeightedFusion")
        assert "def __init__" in cls.text

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("func Search(q string) []Result {\n\treturn nil\n}\n", "Search"),
            ("pub fn fuse(a: f64) -> f64 { a }\n", "fuse"),
            ("export function fuseAll(a) { return a; }\n", "fuseAll"),
            ("export class Reranker {\n  score() {}\n}\n", "Reranker"),
            ("export const fuse = async (a, b) => a + b;\n", "fuse"),
            ("ingest_docs() {\n  kb ingest ./docs\n}\n", "ingest_docs"),
        ],
    )
    def test_other_languages(self, source: str, expected: str) -> None:
        prelude = "// header\n// header\n// header\n// header\n"
        drafts = CodeChunker(1600).chunk(prelude + source)
        assert expected in [d.metadata.get("symbol") for d in drafts]

    def test_structureless_content_falls_back_to_windows(self) -> None:
        data = "\n".join(f"key_{i}: value_{i}" for i in range(200))
        drafts = CodeChunker(400).chunk(data)
        assert len(drafts) > 1
        assert all(d.line_start <= d.line_end for d in drafts)

    def test_empty_input(self) -> None:
        assert CodeChunker().chunk("") == []


class TestGitHubIngestion:
    @pytest.fixture
    def checkout(self, tmp_path: Path) -> Path:
        root = tmp_path / "checkout"
        (root / "src").mkdir(parents=True)
        (root / "src" / "fusion.py").write_text(TestCodeChunker.PYTHON)
        (root / "README.md").write_text("# demo\n\nA demonstration repository.\n")
        (root / "package-lock.json").write_text('{"lockfileVersion": 3}')
        (root / "node_modules").mkdir()
        (root / "node_modules" / "vendored.py").write_text("# skip me\n")
        return root

    def test_local_checkout_produces_github_permalinks(
        self, tmp_path: Path, checkout: Path
    ) -> None:
        settings = Settings(data_dir=tmp_path / "data", embedding_dim=128)
        knowledge_base = KnowledgeBase(settings)
        report = knowledge_base.ingest("anshnt/demo", local_path=str(checkout), ref="main")
        assert report.documents_created == 2, report.errors

        chunks = [
            c for document in report.documents for c in knowledge_base.document_chunks(document.id)
        ]
        assert chunks
        for chunk in chunks:
            assert isinstance(chunk.locator, GitHubLocator)
            assert chunk.locator.repo == "anshnt/demo"
            assert chunk.locator.ref == "main"
            link = chunk.deep_link()
            assert link.startswith("https://github.com/anshnt/demo/blob/main/")
            assert "#L" in link
        knowledge_base.close()

    def test_symbols_reach_the_citation_label(self, tmp_path: Path, checkout: Path) -> None:
        """`fusion.py:6 (reciprocal_rank_fusion)` beats `fusion.py:6`."""
        settings = Settings(data_dir=tmp_path / "data", embedding_dim=128)
        knowledge_base = KnowledgeBase(settings)
        report = knowledge_base.ingest("anshnt/demo", local_path=str(checkout), ref="main")
        labels = [
            c.citation_label()
            for document in report.documents
            for c in knowledge_base.document_chunks(document.id)
        ]
        assert any("(reciprocal_rank_fusion)" in label for label in labels)
        knowledge_base.close()

    def test_excluded_paths_are_not_ingested(self, tmp_path: Path, checkout: Path) -> None:
        settings = Settings(data_dir=tmp_path / "data", embedding_dim=128)
        knowledge_base = KnowledgeBase(settings)
        report = knowledge_base.ingest("anshnt/demo", local_path=str(checkout), ref="main")
        titles = [d.title for d in report.documents]
        assert not any("node_modules" in t or "package-lock" in t for t in titles)
        knowledge_base.close()

    def test_code_is_searchable(self, tmp_path: Path, checkout: Path) -> None:
        settings = Settings(data_dir=tmp_path / "data", embedding_dim=256)
        knowledge_base = KnowledgeBase(settings)
        knowledge_base.ingest("anshnt/demo", local_path=str(checkout), ref="main")
        result = knowledge_base.search("reciprocal rank fusion ranked lists", top_k=3)
        assert result.results
        assert any("reciprocal_rank_fusion" in r.chunk.text for r in result.results)
        knowledge_base.close()

    def test_missing_local_path_is_reported(self, tmp_path: Path) -> None:
        connector = GitHubConnector(Settings(data_dir=tmp_path))
        with pytest.raises(IngestionError, match="not a directory"):
            list(connector.parse("a/b", local_path=str(tmp_path / "nope")))


# --------------------------------------------------------------------------- #
# youtube
# --------------------------------------------------------------------------- #


class TestExtractVideoId:
    @pytest.mark.parametrize(
        "source",
        [
            "dQw4w9WgXcQ",
            "yt:dQw4w9WgXcQ",
            "youtube:dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ?t=42",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "https://www.youtube.com/live/dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        ],
    )
    def test_every_url_shape(self, source: str) -> None:
        assert extract_video_id(source) == "dQw4w9WgXcQ"

    @pytest.mark.parametrize(
        "source",
        [
            "https://example.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/short",
            "far-too-long-to-be-an-id",
        ],
    )
    def test_non_videos_return_none(self, source: str) -> None:
        assert extract_video_id(source) is None


class TestCleanCue:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("[Music] hello there", "hello there"),
            ("[Applause]", ""),
            ("hello  \n there", "hello there"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_noise_markers_are_removed(self, raw, expected: str) -> None:
        assert clean_cue(raw) == expected


class TestGroupIntoWindows:
    def cues(self, n: int, step: float = 3.0) -> list[Cue]:
        return [Cue(text=f"cue {i}", start=i * step, duration=step) for i in range(n)]

    def test_groups_by_time_not_by_characters(self) -> None:
        """Auto-captions have no punctuation, so time is the only usable unit."""
        windows = group_into_windows(self.cues(40), 30.0)
        assert len(windows) > 1
        for _, start, end in windows:
            assert end - start <= 40.0

    def test_windows_advance(self) -> None:
        windows = group_into_windows(self.cues(60), 20.0)
        starts = [start for _, start, _ in windows]
        assert starts == sorted(starts)
        assert len(set(starts)) == len(starts)

    def test_windows_overlap_in_time(self) -> None:
        windows = group_into_windows(self.cues(40), 30.0, overlap_ratio=0.3)
        assert len(windows) >= 2
        assert windows[1][1] < windows[0][2]

    def test_every_cue_appears_somewhere(self) -> None:
        cues = self.cues(25)
        joined = " ".join(text for text, _, _ in group_into_windows(cues, 15.0))
        for cue in cues:
            assert cue.text in joined

    def test_a_cue_longer_than_the_window(self) -> None:
        cues = [Cue(text="very long cue", start=0.0, duration=300.0)]
        windows = group_into_windows(cues, 30.0)
        assert len(windows) == 1

    def test_empty_input(self) -> None:
        assert group_into_windows([], 30.0) == []
        assert group_into_windows(self.cues(3), 0.0) == []

    def test_terminates_on_dense_cues(self) -> None:
        """A tight overlap must not produce an infinite loop."""
        cues = [Cue(text=f"c{i}", start=i * 0.1, duration=0.1) for i in range(200)]
        windows = group_into_windows(cues, 1.0, overlap_ratio=0.9)
        assert 1 < len(windows) < 500


# --------------------------------------------------------------------------- #
# notion
# --------------------------------------------------------------------------- #


class TestSplitNotionFilename:
    @pytest.mark.parametrize(
        ("filename", "title", "page_id"),
        [
            (
                "Runbooks b2c3d4e5f60718293a4b5c6d7e8f901a.md",
                "Runbooks",
                "b2c3d4e5f60718293a4b5c6d7e8f901a",
            ),
            (
                "On%20call%20rota a1b2c3d4e5f60718293a4b5c6d7e8f90.md",
                "On call rota",
                "a1b2c3d4e5f60718293a4b5c6d7e8f90",
            ),
            ("Plain Page.md", "Plain Page", None),
        ],
    )
    def test_id_is_split_from_the_title(
        self, filename: str, title: str, page_id: str | None
    ) -> None:
        assert split_notion_filename(filename) == (title, page_id)


class TestCsvToMarkdown:
    def test_rows_become_keyed_blocks(self) -> None:
        """A raw CSV row embeds terribly: the values lose their field names."""
        csv_text = "Name,Owner,Tier\nretrieval-api,platform,1\ningest-worker,platform,2\n"
        markdown = csv_to_markdown(csv_text, "Services")
        assert "# Services" in markdown
        assert "## retrieval-api" in markdown
        assert "- Owner: platform" in markdown
        assert "- Tier: 1" in markdown

    def test_empty_values_are_omitted(self) -> None:
        markdown = csv_to_markdown("Name,Owner\nthing,\n", "T")
        assert "- Owner:" not in markdown

    def test_header_only_produces_nothing(self) -> None:
        assert csv_to_markdown("Name,Owner\n", "T") == ""


def test_strip_notion_artifacts_removes_child_page_links() -> None:
    text = (
        "# Engineering\n\nOwner:\n\n"
        "[Runbooks](Runbooks%20b2c3d4e5f60718293a4b5c6d7e8f901a.md)\n\n"
        "Real body text about the retrieval service.\n"
    )
    cleaned = strip_notion_artifacts(text)
    assert "Runbooks%20" not in cleaned
    assert "Real body text" in cleaned


class TestNotionIngestion:
    @pytest.fixture
    def export(self, tmp_path: Path) -> Path:
        root = tmp_path / "Export-abc123"
        nested = root / "Engineering a1b2c3d4e5f60718293a4b5c6d7e8f90"
        nested.mkdir(parents=True)
        (root / "Engineering a1b2c3d4e5f60718293a4b5c6d7e8f90.md").write_text(
            "# Engineering\n\nOwner:\n\n"
            "[Runbooks](Runbooks%20b2c3d4e5f60718293a4b5c6d7e8f901a.md)\n\n"
            "The engineering team owns the retrieval service and its evaluation harness.\n"
        )
        (nested / "Runbooks b2c3d4e5f60718293a4b5c6d7e8f901a.md").write_text(
            "# Runbooks\n\n## On-call rotation\n\n"
            "The on-call engineer is paged when retrieval latency exceeds two hundred "
            "milliseconds at the ninety-fifth percentile.\n"
        )
        (nested / "Services c3d4e5f60718293a4b5c6d7e8f901a2b.csv").write_text(
            "Name,Owner,Tier,Notes\nretrieval-api,platform,1,Serves hybrid search and reranking\n"
        )
        return root

    def test_export_directory_is_recognised(self, tmp_path: Path, export: Path) -> None:
        connector = NotionConnector(Settings(data_dir=tmp_path))
        assert connector.can_handle(str(export))

    def test_pages_and_databases_are_ingested(self, tmp_path: Path, export: Path) -> None:
        settings = Settings(data_dir=tmp_path / "data", embedding_dim=128)
        knowledge_base = KnowledgeBase(settings)
        report = knowledge_base.ingest(str(export))
        assert report.documents_created == 3, report.errors
        assert not report.errors
        knowledge_base.close()

    def test_page_ids_survive_into_locators(self, tmp_path: Path, export: Path) -> None:
        settings = Settings(data_dir=tmp_path / "data", embedding_dim=128)
        knowledge_base = KnowledgeBase(settings)
        report = knowledge_base.ingest(str(export))
        for document in report.documents:
            for chunk in knowledge_base.document_chunks(document.id):
                assert isinstance(chunk.locator, NotionLocator)
                assert chunk.locator.notion_page_id
                assert chunk.deep_link().startswith("https://www.notion.so/")
        knowledge_base.close()

    def test_hierarchy_reaches_the_title(self, tmp_path: Path, export: Path) -> None:
        """`Engineering › Runbooks` beats a filename with a hash in it."""
        settings = Settings(data_dir=tmp_path / "data", embedding_dim=128)
        knowledge_base = KnowledgeBase(settings)
        report = knowledge_base.ingest(str(export))
        titles = [d.title for d in report.documents]
        assert any("Engineering" in t and "Runbooks" in t for t in titles)
        assert not any("a1b2c3d4" in t for t in titles)
        knowledge_base.close()

    def test_zip_export_is_ingested(self, tmp_path: Path, export: Path) -> None:
        archive = tmp_path / "notion-export.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            for path in export.rglob("*"):
                if path.is_file():
                    handle.write(path, path.relative_to(export.parent).as_posix())

        settings = Settings(data_dir=tmp_path / "data2", embedding_dim=128)
        knowledge_base = KnowledgeBase(settings)
        assert NotionConnector(settings).can_handle(str(archive))
        report = knowledge_base.ingest(str(archive))
        assert report.documents_created == 3, report.errors
        knowledge_base.close()

    def test_search_finds_notion_content(self, tmp_path: Path, export: Path) -> None:
        settings = Settings(data_dir=tmp_path / "data", embedding_dim=256)
        knowledge_base = KnowledgeBase(settings)
        knowledge_base.ingest(str(export))
        result = knowledge_base.search("who is paged when latency is high", top_k=3)
        assert result.results
        assert any("on-call" in r.chunk.text.lower() for r in result.results)
        knowledge_base.close()

    def test_a_non_notion_zip_is_not_claimed(self, tmp_path: Path) -> None:
        archive = tmp_path / "plain.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("notes.md", "# Notes\n\nbody")
        assert not NotionConnector(Settings(data_dir=tmp_path)).can_handle(str(archive))
