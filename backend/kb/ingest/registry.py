"""Connector registry and source resolution.

Sources arrive as strings — a path, a glob, a directory, a URL, ``inline:``.
The registry decides which connector owns each one, and expands directories and
globs into individual files so callers can say ``kb ingest ./docs`` and mean it.

Registration order is precedence order: the first connector whose
``can_handle`` returns true wins, so a more specific connector (Notion export
zip) must be registered before a more general one (any zip).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from kb.config import Settings
from kb.errors import UnsupportedSourceError
from kb.ingest.base import Connector

#: Directories never worth indexing. Walking them wastes minutes and fills the
#: corpus with vendored code and build output that nobody asks questions about.
SKIP_DIRECTORIES: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        "target",
        ".next",
        ".nuxt",
        "vendor",
        ".terraform",
        "site-packages",
        ".idea",
        ".vscode",
    }
)


class ConnectorRegistry:
    """Maps source strings to the connector that can parse them."""

    def __init__(self, connectors: Sequence[Connector] | None = None) -> None:
        self._connectors: list[Connector] = list(connectors or [])

    def register(self, connector: Connector) -> None:
        self._connectors.append(connector)

    def register_first(self, connector: Connector) -> None:
        """Register with highest precedence."""
        self._connectors.insert(0, connector)

    @property
    def connectors(self) -> list[Connector]:
        return list(self._connectors)

    def names(self) -> list[str]:
        return [c.name for c in self._connectors]

    def resolve(self, source: str) -> Connector:
        for connector in self._connectors:
            if connector.can_handle(source):
                return connector
        raise UnsupportedSourceError(
            f"no connector can handle {source!r}",
            details={"source": source, "available": ", ".join(self.names())},
        )

    def find(self, name: str) -> Connector | None:
        return next((c for c in self._connectors if c.name == name), None)

    def can_handle(self, source: str) -> bool:
        return any(c.can_handle(source) for c in self._connectors)

    # ------------------------------------------------------------------ #

    def expand(self, source: str) -> Iterator[str]:
        """Expand a source into individual ingestable sources.

        Directories are walked, globs are matched, and anything that is not a
        local path (URLs, ``inline:``, ``owner/repo``) is passed through for a
        connector to interpret.
        """
        if "://" in source and not source.startswith("file://"):
            yield source
            return
        if source.startswith("inline:"):
            yield source
            return

        candidate = Path(source.removeprefix("file://")).expanduser()
        if candidate.is_dir():
            yield from self._walk(candidate)
            return
        if candidate.is_file():
            yield str(candidate)
            return
        if any(ch in source for ch in "*?["):
            base = candidate.parent if candidate.parent != candidate else Path()
            pattern = candidate.name
            matched = False
            for match in sorted(base.glob(pattern)):
                if match.is_file() and self.can_handle(str(match)):
                    matched = True
                    yield str(match)
                elif match.is_dir():
                    for nested in self._walk(match):
                        matched = True
                        yield nested
            if not matched:
                raise UnsupportedSourceError(f"glob {source!r} matched no ingestable files")
            return
        # Not a local path: hand it to the connectors as-is (e.g. ``owner/repo``).
        yield source

    def _walk(self, directory: Path) -> Iterator[str]:
        for path in sorted(directory.rglob("*")):
            if path.is_dir():
                continue
            if any(part in SKIP_DIRECTORIES for part in path.parts):
                continue
            if path.name.startswith("."):
                continue
            candidate = str(path)
            if self.can_handle(candidate):
                yield candidate


def default_registry(settings: Settings) -> ConnectorRegistry:
    """The connector set available out of the box.

    Optional connectors register themselves only when their dependencies are
    importable, so a base install still ingests PDFs, Markdown and text.
    """
    from kb.ingest.pdf import PDFConnector
    from kb.ingest.text import InlineTextConnector, MarkdownConnector, TextConnector

    registry = ConnectorRegistry(
        [
            InlineTextConnector(settings),
            MarkdownConnector(settings),
            PDFConnector(settings),
            TextConnector(settings),
        ]
    )
    _register_optional(registry, settings)
    return registry


def _register_optional(registry: ConnectorRegistry, settings: Settings) -> None:
    """Attach connectors that ship in later modules, when present.

    Keeps the registry additive: a new connector module becomes available by
    existing, without editing this function's callers.
    """
    for module_name, class_name in (
        ("kb.ingest.notion", "NotionConnector"),
        ("kb.ingest.web", "WebConnector"),
        ("kb.ingest.github", "GitHubConnector"),
        ("kb.ingest.youtube", "YouTubeConnector"),
    ):
        try:
            module = __import__(module_name, fromlist=[class_name])
            connector_cls = getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
        try:
            connector = connector_cls(settings)
        except Exception:
            continue
        # Notion export zips must be matched before any generic archive handler.
        if class_name == "NotionConnector":
            registry.register_first(connector)
        else:
            registry.register(connector)
