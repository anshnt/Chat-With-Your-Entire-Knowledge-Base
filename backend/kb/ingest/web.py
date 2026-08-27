"""Website connector.

Two hard parts, and the interesting work is in both.

**Boilerplate removal.** A raw web page is mostly navigation, cookie banners,
footers and related-article rails. Indexing that fills the corpus with text that
matches every query weakly and none strongly, and it wrecks the heading paths
that make citations readable. There is no dependency here (no readability, no
trafilatura), so extraction is done structurally: strip non-content elements,
then score candidate containers by text density — the ratio of text to markup —
which is the single most reliable signal for "this is the article".

**Citations that land on the right words.** Web pages have no page numbers, so
:class:`~kb.models.WebLocator` uses the Text Fragments spec
(``#:~:text=start,end``). The connector records the first and last few words of
each chunk as the fragment anchors, and the browser scrolls to and highlights the
quote on a page nobody controls.

Crawling is polite by construction: same-origin only, depth- and page-capped,
``robots.txt`` respected, and one request at a time.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

from kb.chunking.base import ChunkDraft
from kb.chunking.markdown import MarkdownChunker
from kb.config import Settings
from kb.errors import IngestionError, MissingDependencyError
from kb.ingest.base import ParsedDocument, Segment
from kb.models import Locator, SourceType, WebLocator

log = logging.getLogger(__name__)

#: Elements that are never article content.
_STRIP_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "button",
    "iframe",
    "svg",
    "template",
    "dialog",
)

#: Class/id substrings that mark navigation and chrome across most sites.
_STRIP_PATTERNS = re.compile(
    r"(nav|menu|sidebar|footer|header|banner|cookie|consent|advert|promo|social|share"
    r"|comment|related|recommend|newsletter|subscribe|breadcrumb|pagination|toc"
    r"|skip-link|screen-reader)",
    re.I,
)

#: Containers that plausibly hold the article, in descending preference.
_CONTENT_SELECTORS = (
    "article",
    "main",
    '[role="main"]',
    "#content",
    ".content",
    "#main",
    ".post",
    ".entry-content",
    ".markdown-body",
    ".article-body",
    ".documentation",
)

#: Words used at each end of a chunk to build its text fragment. Too few and the
#: fragment matches the wrong place; too many and any whitespace difference
#: between our extraction and the live DOM breaks the match.
FRAGMENT_WORDS = 6

_WHITESPACE_RE = re.compile(r"[ \t]*\n[ \t]*")


class WebConnector:
    """Fetches and extracts readable content from web pages."""

    name = "web"
    source_type = SourceType.WEB

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._robots: dict[str, RobotFileParser | None] = {}

    def can_handle(self, source: str) -> bool:
        return source.startswith(("http://", "https://"))

    # ------------------------------------------------------------------ #

    def parse(self, source: str, **options: Any) -> Iterable[ParsedDocument]:
        """Fetch ``source`` and, if asked, crawl same-origin links from it."""
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - httpx is a base dependency
            raise MissingDependencyError("httpx", "all") from exc

        crawl = bool(options.get("crawl", False))
        max_pages = int(options.get("max_pages", self.settings.web_crawl_max_pages))
        max_depth = int(options.get("max_depth", self.settings.web_crawl_max_depth))
        follow_external = bool(options.get("follow_external", False))

        headers = {
            "User-Agent": self.settings.web_user_agent,
            "Accept": "text/html,application/xhtml+xml",
        }
        with httpx.Client(
            timeout=self.settings.web_request_timeout,
            follow_redirects=True,
            headers=headers,
        ) as client:
            yield from self._crawl(
                client,
                source,
                crawl=crawl,
                max_pages=max_pages if crawl else 1,
                max_depth=max_depth if crawl else 0,
                follow_external=follow_external,
                title_override=options.get("title"),
            )

    # ------------------------------------------------------------------ #

    def _crawl(
        self,
        client: Any,
        start_url: str,
        *,
        crawl: bool,
        max_pages: int,
        max_depth: int,
        follow_external: bool,
        title_override: str | None,
    ) -> Iterator[ParsedDocument]:
        seen: set[str] = set()
        queue: list[tuple[str, int]] = [(canonical_url(start_url), 0)]
        origin = urlparse(start_url).netloc
        produced = 0
        failures: list[str] = []

        while queue and produced < max_pages:
            url, depth = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)

            if not self._allowed(client, url):
                log.info("skipping %s: disallowed by robots.txt", url)
                continue

            try:
                response = client.get(url)
                response.raise_for_status()
            except Exception as exc:
                failures.append(f"{url}: {exc}")
                # One bad page must not abort a crawl; the caller sees it in the
                # ingestion report only if *nothing* succeeded.
                log.warning("could not fetch %s: %s", url, exc)
                continue

            content_type = response.headers.get("content-type", "")
            if "html" not in content_type and "xml" not in content_type:
                log.info("skipping %s: content-type %s", url, content_type)
                continue

            try:
                document, links = self._parse_html(response.text, url, title_override)
            except IngestionError as exc:
                failures.append(f"{url}: {exc.message}")
                continue

            produced += 1
            yield document

            if crawl and depth < max_depth:
                for link in links:
                    if link in seen:
                        continue
                    if not follow_external and urlparse(link).netloc != origin:
                        continue
                    queue.append((link, depth + 1))

        if produced == 0:
            raise IngestionError(
                f"no readable content retrieved from {start_url}",
                details={"failures": "; ".join(failures[:5])},
            )

    def _allowed(self, client: Any, url: str) -> bool:
        """Check ``robots.txt``, caching one parser per origin.

        A crawler that ignores robots.txt is a crawler nobody should run.
        """
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            parser: RobotFileParser | None = RobotFileParser()
            try:
                response = client.get(f"{origin}/robots.txt")
                if response.status_code == 200:
                    parser.parse(response.text.splitlines())  # type: ignore[union-attr]
                else:
                    # No robots.txt means no restrictions.
                    parser = None
            except Exception:
                parser = None
            self._robots[origin] = parser

        parser = self._robots[origin]
        if parser is None:
            return True
        return parser.can_fetch(self.settings.web_user_agent, url)

    # ------------------------------------------------------------------ #

    def _parse_html(
        self, html: str, url: str, title_override: str | None
    ) -> tuple[ParsedDocument, list[str]]:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:  # pragma: no cover - bs4 is a base dependency
            raise MissingDependencyError("beautifulsoup4", "all") from exc

        soup = BeautifulSoup(html, "lxml")
        links = extract_links(soup, url)
        title = title_override or extract_title(soup) or urlparse(url).path.strip("/") or url

        container = select_content(soup)
        strip_chrome(container)
        markdown = to_markdown(container)
        if len(markdown.strip()) < 80:
            raise IngestionError(f"{url} yielded almost no readable text after boilerplate removal")

        chunker = MarkdownChunker(
            self.settings.chunk_size,
            self.settings.chunk_overlap,
            self.settings.min_chunk_size,
        )
        document = ParsedDocument(
            title=str(title),
            uri=canonical_url(url),
            source_type=self.source_type,
            segments=[Segment(text=markdown, build_locator=_web_locator_factory(url))],
            raw_text=markdown,
            byte_size=len(html.encode("utf-8")),
            language=extract_language(soup),
            author=extract_author(soup),
            metadata={"url": canonical_url(url), "n_links": len(links)},
            chunker=chunker,
        )
        return document, links


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #


def canonical_url(url: str) -> str:
    """Drop the fragment and any trailing slash, so one page is one document."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return urlunparse(parsed._replace(path=path, fragment=""))


def extract_title(soup: Any) -> str | None:
    """Prefer the page's own h1 over ``<title>``, which often carries the site name."""
    heading = soup.find("h1")
    if heading and heading.get_text(strip=True):
        return heading.get_text(strip=True)
    for prop in ("og:title", "twitter:title"):
        tag = soup.find("meta", attrs={"property": prop}) or soup.find("meta", attrs={"name": prop})
        if tag and tag.get("content"):
            return str(tag["content"]).strip()
    if soup.title and soup.title.string:
        # "Article — Site Name" → "Article".
        return re.split(r"\s+[|—–-]\s+", soup.title.string.strip())[0]
    return None


def extract_author(soup: Any) -> str | None:
    for attrs in ({"name": "author"}, {"property": "article:author"}):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            return str(tag["content"]).strip()[:200]
    return None


def extract_language(soup: Any) -> str | None:
    html_tag = soup.find("html")
    if html_tag and html_tag.get("lang"):
        return str(html_tag["lang"]).split("-")[0]
    return None


def extract_links(soup: Any, base_url: str) -> list[str]:
    """Absolute, deduplicated, HTTP(S) links, excluding asset downloads."""
    seen: dict[str, None] = {}
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute = canonical_url(urljoin(base_url, href))
        if not absolute.startswith(("http://", "https://")):
            continue
        if re.search(
            r"\.(zip|tar|gz|png|jpe?g|gif|svg|css|js|ico|woff2?|mp4|mp3)$", absolute, re.I
        ):
            continue
        seen.setdefault(absolute, None)
    return list(seen)


def select_content(soup: Any) -> Any:
    """Pick the element most likely to hold the article.

    Tries semantic containers first, then falls back to scoring every ``div`` by
    **text density** — the ratio of text length to markup length. Navigation is
    link-dense and text-sparse; an article is the opposite. That single ratio is
    more reliable than any list of site-specific selectors.
    """
    for selector in _CONTENT_SELECTORS:
        try:
            found = soup.select_one(selector)
        except Exception:
            continue
        if found is not None and len(found.get_text(strip=True)) > 200:
            return found

    best, best_score = None, 0.0
    for candidate in soup.find_all(["div", "section", "td"]):
        text = candidate.get_text(" ", strip=True)
        if len(text) < 200:
            continue
        markup_length = len(str(candidate)) or 1
        density = len(text) / markup_length
        link_text = sum(len(a.get_text(strip=True)) for a in candidate.find_all("a"))
        link_ratio = link_text / max(len(text), 1)
        # Density rewards prose; the link penalty pushes down nav lists that are
        # long enough to pass the length filter.
        score = density * len(text) ** 0.5 * (1.0 - min(link_ratio, 0.9))
        if score > best_score:
            best, best_score = candidate, score

    return best if best is not None else (soup.body or soup)


def strip_chrome(node: Any) -> None:
    """Remove non-content elements in place."""
    if node is None:
        return
    for tag in node.find_all(_STRIP_TAGS):
        tag.decompose()
    for tag in node.find_all(attrs={"class": _STRIP_PATTERNS}):
        tag.decompose()
    for tag in node.find_all(attrs={"id": _STRIP_PATTERNS}):
        tag.decompose()
    for tag in node.find_all(attrs={"aria-hidden": "true"}):
        tag.decompose()


def to_markdown(node: Any) -> str:
    """Convert the content element to Markdown.

    Markdown rather than plain text so the heading-aware chunker can do its job:
    a web page's ``<h2>`` structure is exactly as useful for citation labels as a
    Markdown document's, and throwing it away is the common mistake.
    """
    if node is None:
        return ""

    lines: list[str] = []

    def render(element: Any, depth: int = 0) -> None:
        name = getattr(element, "name", None)
        if name is None:
            text = str(element).strip()
            if text:
                lines.append(text)
            return

        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(name[1])
            text = element.get_text(" ", strip=True)
            if text:
                lines.append(f"\n{'#' * level} {text}\n")
            return
        if name == "p":
            text = _inline_text(element)
            if text:
                lines.append(f"\n{text}\n")
            return
        if name == "pre":
            code = element.get_text()
            if code.strip():
                lines.append(f"\n```\n{code.rstrip()}\n```\n")
            return
        if name in ("ul", "ol"):
            ordered = name == "ol"
            for index, item in enumerate(element.find_all("li", recursive=False), start=1):
                text = _inline_text(item)
                if text:
                    marker = f"{index}." if ordered else "-"
                    lines.append(f"{'  ' * depth}{marker} {text}")
            lines.append("")
            return
        if name == "table":
            rendered = _render_table(element)
            if rendered:
                lines.append(f"\n{rendered}\n")
            return
        if name == "blockquote":
            text = element.get_text(" ", strip=True)
            if text:
                lines.append(f"\n> {text}\n")
            return
        if name in ("br", "hr"):
            lines.append("")
            return

        for child in getattr(element, "children", []):
            render(child, depth + (1 if name in ("ul", "ol", "li") else 0))

    render(node)
    text = "\n".join(lines)
    text = _WHITESPACE_RE.sub("\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _inline_text(element: Any) -> str:
    """Text of an element with inline code preserved as backticks."""
    parts: list[str] = []
    for child in element.descendants:
        if getattr(child, "name", None) == "code":
            code = child.get_text(strip=True)
            if code:
                parts.append(f"`{code}`")
        elif getattr(child, "name", None) is None:
            text = str(child).strip()
            if text and not any(
                getattr(parent, "name", None) == "code" for parent in child.parents
            ):
                parts.append(text)
    return re.sub(r"\s{2,}", " ", " ".join(parts)).strip()


def _render_table(table: Any) -> str:
    rows: list[list[str]] = []
    for row in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, *body = rows
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# locators
# --------------------------------------------------------------------------- #


def fragment_anchors(text: str, words: int = FRAGMENT_WORDS) -> tuple[str, str]:
    """First and last few words of ``text``, for a ``#:~:text=`` fragment.

    Heading prefixes and Markdown syntax are stripped first: the fragment has to
    match the *rendered* page, which contains neither.
    """
    body = text
    if "\n\n" in body and body.split("\n\n", 1)[0].count("›"):
        body = body.split("\n\n", 1)[1]
    body = re.sub(r"[#>*`\[\]()]|^-\s+", " ", body)
    tokens = [t for t in body.split() if t]
    if not tokens:
        return "", ""
    prefix = " ".join(tokens[:words])
    suffix = " ".join(tokens[-words:]) if len(tokens) > words else prefix
    return prefix, suffix


def _web_locator_factory(url: str) -> Any:
    def build(draft: ChunkDraft) -> Locator:
        prefix, suffix = fragment_anchors(draft.text)
        return WebLocator(
            url=canonical_url(url),
            heading_path=list(draft.heading_path),
            quote_prefix=prefix or None,
            quote_suffix=suffix or None,
        )

    return build
