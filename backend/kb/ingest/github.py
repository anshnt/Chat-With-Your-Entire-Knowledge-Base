"""GitHub repository connector.

Two things make code different from prose, and both are handled here.

**Code should be chunked on structural boundaries, not on paragraphs.** A chunk
that begins halfway through a function body is unusable as a citation and embeds
badly. :class:`CodeChunker` splits at top-level definitions — ``def``, ``class``,
``func``, ``fn``, ``type``, ``impl``, exported consts — so a chunk is a whole
declaration, and records the enclosing symbol name on the locator. That is what
lets a citation read ``fusion.py:88 (reciprocal_rank_fusion)`` instead of
``fusion.py:88``.

**Line numbers are the address.** :class:`~kb.models.GitHubLocator` carries repo,
ref, path and a line range, which GitHub turns into a permalink with the lines
highlighted. Chunking therefore has to track line numbers exactly, which it does
by construction: the chunker works on line boundaries.

Fetching uses the Git Trees API (one request for the whole file list) plus the
contents API per file, and falls back to a local clone path when given one. The
default file filter excludes lock files, minified bundles, vendored trees and
binaries — indexing a 2 MB ``package-lock.json`` costs real money and answers no
questions.
"""

from __future__ import annotations

import base64
import logging
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from kb.chunking.base import ChunkDraft, line_starts
from kb.chunking.markdown import MarkdownChunker
from kb.config import Settings
from kb.errors import IngestionError, MissingDependencyError
from kb.ingest.base import ParsedDocument, Segment, read_text_file
from kb.models import ChunkKind, GitHubLocator, Locator, SourceType

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"

_REPO_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)(?:\.git)?"
    r"(?:/(?:tree|blob)/(?P<ref>[^/]+)(?:/(?P<path>.*))?)?/?$"
)
_SHORTHAND_RE = re.compile(r"^(?:gh:|github:)(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)$")

#: Extensions worth indexing, mapped to the chunk kind they produce.
CODE_EXTENSIONS: dict[str, ChunkKind] = dict.fromkeys(
    (
        ".py",
        ".pyi",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".mjs",
        ".cjs",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".kts",
        ".scala",
        ".swift",
        ".c",
        ".h",
        ".cc",
        ".cpp",
        ".hpp",
        ".cs",
        ".m",
        ".mm",
        ".rb",
        ".php",
        ".pl",
        ".lua",
        ".r",
        ".jl",
        ".ex",
        ".exs",
        ".erl",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".ps1",
        ".sql",
        ".graphql",
        ".proto",
        ".tf",
        ".hcl",
    ),
    ChunkKind.CODE,
)
PROSE_EXTENSIONS: dict[str, ChunkKind] = dict.fromkeys(
    (".md", ".markdown", ".mdx", ".rst", ".txt", ".adoc"), ChunkKind.PROSE
)
CONFIG_EXTENSIONS: dict[str, ChunkKind] = dict.fromkeys(
    (".yaml", ".yml", ".toml", ".ini", ".cfg", ".json"), ChunkKind.CODE
)

INDEXABLE = {**CODE_EXTENSIONS, **PROSE_EXTENSIONS, **CONFIG_EXTENSIONS}

#: Paths never worth indexing. Generated, vendored, or enormous — and none of
#: them answer questions anyone asks of a knowledge base.
EXCLUDE_PATTERNS = re.compile(
    r"(?:^|/)(?:"
    r"node_modules|vendor|third_party|\.git|dist|build|target|out|coverage"
    r"|__pycache__|\.venv|venv|site-packages|\.next|\.nuxt|\.terraform"
    r")(?:/|$)"
    r"|(?:^|/)(?:"
    r"package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|Cargo\.lock"
    r"|go\.sum|composer\.lock|Gemfile\.lock|uv\.lock"
    r")$"
    r"|\.min\.(?:js|css)$"
    r"|\.(?:map|snap)$",
    re.I,
)


class GitHubConnector:
    """Ingests source files from a GitHub repository or a local checkout."""

    name = "github"
    source_type = SourceType.GITHUB

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def can_handle(self, source: str) -> bool:
        if _SHORTHAND_RE.match(source):
            return True
        if "github.com" in source and _REPO_URL_RE.match(source):
            return True
        # ``owner/repo`` with no local counterpart of the same name.
        return bool(re.fullmatch(r"[\w.-]+/[\w.-]+", source) and not Path(source).exists())

    # ------------------------------------------------------------------ #

    def parse(self, source: str, **options: Any) -> Iterable[ParsedDocument]:
        owner, repo, ref, subpath = parse_repo_source(source)
        ref = str(options.get("ref") or ref or "HEAD")
        subpath = str(options.get("path") or subpath or "")
        max_files = int(options.get("max_files", 400))
        token = str(options.get("token") or "")

        local = options.get("local_path")
        if local:
            yield from self._parse_local(Path(str(local)), owner, repo, ref, subpath, max_files)
            return

        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - httpx is a base dependency
            raise MissingDependencyError("httpx", "all") from exc

        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": self.settings.web_user_agent,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        with httpx.Client(
            timeout=self.settings.web_request_timeout, headers=headers, follow_redirects=True
        ) as client:
            resolved_ref = self._resolve_ref(client, owner, repo, ref)
            entries = self._list_tree(client, owner, repo, resolved_ref, subpath, max_files)
            if not entries:
                raise IngestionError(
                    f"no indexable files found in {owner}/{repo} at {resolved_ref}"
                    + (f" under {subpath}" if subpath else "")
                )
            for path, size in entries:
                try:
                    text = self._fetch_file(client, owner, repo, resolved_ref, path)
                except IngestionError as exc:
                    log.warning("skipping %s: %s", path, exc.message)
                    continue
                if text is None:
                    continue
                yield self._build_document(
                    text, owner=owner, repo=repo, ref=resolved_ref, path=path, byte_size=size
                )

    # ------------------------------------------------------------------ #

    def _parse_local(
        self,
        root: Path,
        owner: str,
        repo: str,
        ref: str,
        subpath: str,
        max_files: int,
    ) -> Iterator[ParsedDocument]:
        """Index a local checkout, still producing GitHub permalinks."""
        base = (root / subpath) if subpath else root
        if not base.is_dir():
            raise IngestionError(f"not a directory: {base}")
        count = 0
        for file_path in sorted(base.rglob("*")):
            if count >= max_files:
                break
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(root).as_posix()
            if not is_indexable(
                relative, file_path.stat().st_size, self.settings.github_max_file_bytes
            ):
                continue
            try:
                text = read_text_file(file_path, max_bytes=self.settings.github_max_file_bytes)
            except IngestionError:
                continue
            if not text.strip():
                continue
            count += 1
            yield self._build_document(
                text,
                owner=owner,
                repo=repo,
                ref=ref,
                path=relative,
                byte_size=file_path.stat().st_size,
            )

    def _resolve_ref(self, client: Any, owner: str, repo: str, ref: str) -> str:
        """Turn ``HEAD`` into the repository's actual default branch.

        A locator pinned to ``HEAD`` produces a link that silently changes meaning
        as the branch moves; the default branch name at least stays honest.
        """
        if ref != "HEAD":
            return ref
        try:
            response = client.get(f"{GITHUB_API}/repos/{owner}/{repo}")
            response.raise_for_status()
            return str(response.json().get("default_branch") or "main")
        except Exception as exc:
            raise IngestionError(
                f"could not read {owner}/{repo}: {exc}. If it is private, pass a token."
            ) from exc

    def _list_tree(
        self,
        client: Any,
        owner: str,
        repo: str,
        ref: str,
        subpath: str,
        max_files: int,
    ) -> list[tuple[str, int]]:
        """One recursive Trees call, then filter locally."""
        try:
            response = client.get(
                f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{ref}",
                params={"recursive": "1"},
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise IngestionError(f"could not list {owner}/{repo}@{ref}: {exc}") from exc

        if payload.get("truncated"):
            log.warning(
                "%s/%s@%s tree was truncated by the API; some files will be missing",
                owner,
                repo,
                ref,
            )

        entries: list[tuple[str, int]] = []
        for item in payload.get("tree", []):
            if item.get("type") != "blob":
                continue
            path = str(item.get("path", ""))
            if subpath and not path.startswith(subpath.rstrip("/") + "/") and path != subpath:
                continue
            size = int(item.get("size") or 0)
            if is_indexable(path, size, self.settings.github_max_file_bytes):
                entries.append((path, size))
            if len(entries) >= max_files:
                break
        return entries

    def _fetch_file(self, client: Any, owner: str, repo: str, ref: str, path: str) -> str | None:
        try:
            response = client.get(
                f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}", params={"ref": ref}
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise IngestionError(f"could not fetch {path}: {exc}") from exc

        if payload.get("encoding") != "base64" or not payload.get("content"):
            return None
        raw = base64.b64decode(payload["content"])
        # A NUL byte means binary, whatever the extension claimed.
        if b"\x00" in raw[:8192]:
            return None
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return None

    # ------------------------------------------------------------------ #

    def _build_document(
        self,
        text: str,
        *,
        owner: str,
        repo: str,
        ref: str,
        path: str,
        byte_size: int,
    ) -> ParsedDocument:
        suffix = Path(path).suffix.lower()
        kind = INDEXABLE.get(suffix, ChunkKind.CODE)
        full_repo = f"{owner}/{repo}"

        if kind is ChunkKind.PROSE and suffix in PROSE_EXTENSIONS:
            chunker: Any = MarkdownChunker(
                self.settings.chunk_size,
                self.settings.chunk_overlap,
                self.settings.min_chunk_size,
            )
        else:
            chunker = CodeChunker(self.settings.code_chunk_size)

        return ParsedDocument(
            title=f"{full_repo}/{path}",
            uri=f"https://github.com/{full_repo}/blob/{ref}/{path}",
            source_type=self.source_type,
            segments=[
                Segment(
                    text=text,
                    build_locator=_github_locator_factory(full_repo, ref, path),
                    kind=kind,
                )
            ],
            raw_text=text,
            byte_size=byte_size or len(text.encode("utf-8")),
            language=_language_for(suffix),
            metadata={"repo": full_repo, "ref": ref, "path": path},
            chunker=chunker,
        )


# --------------------------------------------------------------------------- #
# code chunking
# --------------------------------------------------------------------------- #

#: Top-level declaration starts across the languages this indexes. Deliberately
#: regex-based rather than AST-based: one implementation covers every language,
#: and a missed boundary degrades to a slightly worse chunk rather than an error.
#: Triple-quote and code-fence delimiters. A definition inside one of these
#: is a string, not a declaration.
FENCE_TOKENS = ('"""', "'''", "```")

_DEFINITION_RE = re.compile(
    r"^(?:\s{0,3})(?:"
    r"(?:async\s+)?def\s+(?P<py>\w+)"
    r"|class\s+(?P<cls>\w+)"
    r"|(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(?P<js>\w+)?"
    r"|(?:export\s+)?(?:abstract\s+)?(?:class|interface|enum|type)\s+(?P<ts>\w+)"
    r"|func\s+(?:\([^)]*\)\s*)?(?P<go>\w+)"
    r"|(?:pub\s+)?(?:async\s+)?fn\s+(?P<rs>\w+)"
    r"|(?:pub\s+)?(?:struct|enum|trait|impl|mod)\s+(?P<rsty>\w+)"
    r"|(?:public|private|protected|internal)\s+[\w<>\[\], ]+\s+(?P<java>\w+)\s*\("
    r"|(?:export\s+)?(?:const|let|var)\s+(?P<jsconst>\w+)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\(|function)"
    r"|(?P<sh>\w+)\s*\(\)\s*\{"
    r")"
)


class CodeChunker:
    """Splits source code at top-level declarations, tracking line numbers.

    Falls back to fixed line windows for a file with no recognised declarations
    (a config file, a data table), which is the correct behaviour: an arbitrary
    boundary in structureless content costs nothing.
    """

    def __init__(self, chunk_size: int = 1600, min_lines: int = 3) -> None:
        self.chunk_size = chunk_size
        self.min_lines = min_lines

    def chunk(self, text: str) -> list[ChunkDraft]:
        if not text.strip():
            return []
        lines = text.split("\n")
        starts = line_starts(text)
        boundaries = self._boundaries(lines)

        if not boundaries:
            return self._windows(lines, starts)

        drafts: list[ChunkDraft] = []
        for index, (start_line, symbol) in enumerate(boundaries):
            end_line = boundaries[index + 1][0] - 1 if index + 1 < len(boundaries) else len(lines)
            body = "\n".join(lines[start_line - 1 : end_line]).strip("\n")
            if not body.strip():
                continue
            # An oversized declaration is split into windows, keeping its symbol.
            if len(body) > self.chunk_size * 2:
                drafts.extend(
                    self._windows(
                        lines[start_line - 1 : end_line],
                        starts,
                        line_offset=start_line - 1,
                        symbol=symbol,
                    )
                )
                continue
            drafts.append(
                ChunkDraft(
                    text=body,
                    char_start=starts[start_line - 1] if start_line - 1 < len(starts) else 0,
                    char_end=starts[end_line - 1] if end_line - 1 < len(starts) else len(text),
                    line_start=start_line,
                    line_end=end_line,
                    kind=ChunkKind.CODE,
                    metadata={"symbol": symbol} if symbol else {},
                )
            )
        return drafts

    def _boundaries(self, lines: list[str]) -> list[tuple[int, str | None]]:
        """``(line_number, symbol)`` for each top-level declaration."""
        found: list[tuple[int, str | None]] = []
        in_fence = False
        for index, line in enumerate(lines, start=1):
            # Only an *odd* number of delimiters on a line changes the state.
            # A one-line docstring contains two, and treating it as an opener
            # silently suppressed every definition in the rest of the file.
            for token in FENCE_TOKENS:
                if line.count(token) % 2 == 1:
                    in_fence = not in_fence
            if in_fence:
                continue
            match = _DEFINITION_RE.match(line)
            if match is None:
                continue
            symbol = next((value for value in match.groupdict().values() if value), None)
            found.append((index, symbol))

        if not found:
            return []
        # A leading region (imports, module docstring) is its own chunk.
        if found[0][0] > self.min_lines:
            found.insert(0, (1, None))
        return found

    def _windows(
        self,
        lines: list[str],
        starts: list[int],
        *,
        line_offset: int = 0,
        symbol: str | None = None,
    ) -> list[ChunkDraft]:
        """Fixed windows over lines, for structureless or oversized content."""
        drafts: list[ChunkDraft] = []
        buffer: list[str] = []
        buffer_start = 1
        length = 0

        def flush(end_line: int) -> None:
            nonlocal buffer, length
            body = "\n".join(buffer).strip("\n")
            if body.strip():
                absolute_start = line_offset + buffer_start
                absolute_end = line_offset + end_line
                drafts.append(
                    ChunkDraft(
                        text=body,
                        char_start=starts[min(absolute_start - 1, len(starts) - 1)],
                        char_end=starts[min(absolute_end - 1, len(starts) - 1)],
                        line_start=absolute_start,
                        line_end=absolute_end,
                        kind=ChunkKind.CODE,
                        metadata={"symbol": symbol} if symbol else {},
                    )
                )
            buffer, length = [], 0

        for index, line in enumerate(lines, start=1):
            if buffer and length + len(line) > self.chunk_size:
                flush(index - 1)
                buffer_start = index
            buffer.append(line)
            length += len(line) + 1
        if buffer:
            flush(len(lines))
        return drafts


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def parse_repo_source(source: str) -> tuple[str, str, str | None, str | None]:
    """Parse a repository specifier into ``(owner, repo, ref, path)``."""
    shorthand = _SHORTHAND_RE.match(source)
    if shorthand:
        return shorthand["owner"], shorthand["repo"], None, None

    url_match = _REPO_URL_RE.match(source)
    if url_match:
        return (
            url_match["owner"],
            url_match["repo"],
            url_match["ref"],
            url_match["path"],
        )

    plain = re.fullmatch(r"(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)", source)
    if plain:
        return plain["owner"], plain["repo"], None, None

    raise IngestionError(
        f"could not parse {source!r} as a repository; expected owner/repo or a github.com URL"
    )


def is_indexable(path: str, size: int, max_bytes: int) -> bool:
    """Whether a repository file is worth putting in the index."""
    if EXCLUDE_PATTERNS.search(path):
        return False
    if size > max_bytes:
        return False
    if Path(path).name.startswith("."):
        return False
    return Path(path).suffix.lower() in INDEXABLE


def _language_for(suffix: str) -> str | None:
    return {
        ".py": "python",
        ".pyi": "python",
        ".js": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".kt": "kotlin",
        ".rb": "ruby",
        ".php": "php",
        ".c": "c",
        ".h": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".swift": "swift",
        ".sh": "shell",
        ".bash": "shell",
        ".sql": "sql",
        ".md": "markdown",
    }.get(suffix)


def _github_locator_factory(repo: str, ref: str, path: str) -> Any:
    def build(draft: ChunkDraft) -> Locator:
        return GitHubLocator(
            repo=repo,
            ref=ref,
            path=path,
            line_start=max(1, draft.line_start),
            line_end=max(draft.line_start, draft.line_end),
            symbol=draft.metadata.get("symbol"),
        )

    return build
