# Connectors

A connector turns a *source* into `ParsedDocument`s. It does not chunk, embed, or
write to the store, and it does not decide how a citation is addressed beyond
supplying the callback that builds one.

## Available connectors

| Connector | Recognises | Locator | Extra needed |
|---|---|---|---|
| `notion` | Export directory, `.zip`, or a single exported page | `NotionLocator` | — |
| `markdown` | `.md`, `.markdown`, `.mdx`, `.mdown`, `.mkd` | `TextLocator` | — |
| `pdf` | `.pdf` | `PdfLocator` | — |
| `text` | `.txt`, `.text`, `.rst`, `.log`, `.csv`, `.tsv` | `TextLocator` | — |
| `youtube` | `youtube.com` / `youtu.be` URL, or `yt:VIDEO_ID` | `YouTubeLocator` | `kb-chat[youtube]` |
| `github` | `owner/repo`, `gh:owner/repo`, a `github.com` URL | `GitHubLocator` | — |
| `web` | Any `http(s)` URL | `WebLocator` | — |
| `inline` | `inline:<title>` | `TextLocator` | — |

`kb connectors` prints this table for the connectors actually available in your
install.

## Source resolution

`kb ingest` accepts anything the registry can expand:

```bash
kb ingest ./docs                    # directory, walked recursively
kb ingest './notes/**/*.md'         # glob
kb ingest ~/papers/attention.pdf    # single file
kb ingest https://example.com/docs  # URL (passed through to a connector)
cat notes.md | kb add-text Notes    # stdin
```

Directory walks skip the usual noise — `.git`, `node_modules`, `__pycache__`,
`.venv`, `dist`, `build`, `target`, `vendor`, `site-packages` and friends — because
walking them wastes minutes and fills the corpus with vendored code nobody asks
questions about.

**Registration order is precedence order**, and it is load-bearing here: the web
connector claims *any* `http(s)` URL, so `youtube` and `github` are registered
ahead of it or a `github.com` link gets scraped as a generic page. Notion export
zips are registered first, ahead of any archive handling.

**A connector may claim a whole directory.** When one does, the registry hands
over the tree rather than walking it — a Notion export needs its nesting to
reconstruct the page hierarchy, and expanding it first would deliver orphaned
files with no ancestors.

## What each connector does beyond "read the file"

### Markdown

Strips YAML front matter into metadata rather than leaving it in the body — `draft: false`
in the prose pollutes both the BM25 index and the embeddings. Prefers the
document's own H1 over the filename for the title. Uses the heading-aware
chunker.

### Text

Tolerant decoding: UTF-8, then UTF-8-BOM, then CP1252, then Latin-1, then lossy
UTF-8. Real knowledge bases contain files saved on Windows in 1997, and decoding
strictly would abort an entire directory over one of them.

### PDF

One segment per page, so every chunk carries the page it came from and chunking
never straddles a page boundary. Then three repairs that extracted PDF text
always needs:

- **De-hyphenation** — `retriev-\nal` becomes `retrieval`, or neither BM25 nor the
  embedder will ever match the word.
- **Header/footer removal** — a line appearing on ≥60% of pages (and at least 3 of
  them) is boilerplate. Detected by frequency across pages rather than by
  position, which is robust to varying layouts.
- **Soft-wrap rejoining** — a line not ending in sentence punctuation is a wrap,
  not a paragraph break.

Ligatures are normalised, bare page numbers dropped, and an encrypted PDF is
retried with an empty user password before giving up. A PDF with no text layer
gets an error that says to run OCR, rather than failing cryptically.

Uploaded PDFs are **kept on disk**, not parsed and discarded: a citation's
`/files/report.pdf#page=12` only resolves if the bytes are still reachable.

### Website

No `readability` or `trafilatura` dependency — extraction is structural:

- **Chrome removal.** Non-content elements (`nav`, `aside`, `footer`, `script`)
  are dropped, along with anything whose class or id matches the usual
  navigation/cookie/promo/related patterns. Indexing that text fills the corpus
  with content that matches every query weakly and none strongly.
- **Content selection by text density.** Semantic containers (`article`, `main`,
  `[role=main]`, `.markdown-body`) are tried first; the fallback scores every
  `div` by the ratio of text length to markup length, with a penalty for link
  density. Navigation is link-dense and text-sparse; an article is the opposite.
  That single ratio beats any list of site-specific selectors.
- **Conversion to Markdown, not plain text**, so the heading-aware chunker can
  work. A page's `<h2>` structure is exactly as useful for citation labels as a
  Markdown file's, and discarding it is the common mistake.
- **Citations via Text Fragments.** Web pages have no page numbers, so the
  locator records the first and last few words of each chunk and builds
  `#:~:text=start,end`. Chromium, Edge and Safari scroll to and highlight the
  quote on a page nobody controls. Markdown syntax and heading prefixes are
  stripped from the anchors first, since the fragment has to match the *rendered*
  page.
- **Polite crawling**: same-origin only by default, page- and depth-capped,
  `robots.txt` respected (cached per origin), one request at a time. One failed
  page does not abort a crawl.

```bash
kb ingest https://example.com/docs --crawl --max-pages 25 --max-depth 2
```

### GitHub

- **Code is chunked at top-level declarations**, not on paragraphs. A chunk
  beginning halfway through a function body is unusable as a citation and embeds
  badly. `CodeChunker` recognises `def`, `class`, `function`, `func`, `fn`,
  `struct`/`enum`/`trait`/`impl`, `interface`/`type`, exported consts and shell
  functions — one regex-based implementation across every language, where a
  missed boundary degrades to a slightly worse chunk rather than an error.
- **The enclosing symbol reaches the citation**, so a result reads
  `fusion.py:88 (reciprocal_rank_fusion)` instead of `fusion.py:88`.
- **Declarations inside strings are ignored.** Only an *odd* number of
  triple-quote delimiters on a line changes the fence state: a one-line docstring
  contains two, and treating it as an opener silently suppressed every definition
  in the rest of the file — which is exactly what happened before a test caught
  it.
- **Structureless files fall back to line windows**, which is correct: an
  arbitrary boundary in a config file costs nothing.
- **Generated and vendored paths are excluded** — lock files, `node_modules`,
  `vendor`, `dist`, minified bundles, source maps. Indexing a 2 MB
  `package-lock.json` costs real money and answers no questions.
- `HEAD` is resolved to the repository's actual default branch, because a
  locator pinned to `HEAD` produces a link whose meaning changes as the branch
  moves.

```bash
kb ingest anshnt/kb --ref main --path backend
kb ingest anshnt/kb --local ./checkout      # local files, GitHub permalinks
kb ingest anshnt/private-repo --token "$GITHUB_TOKEN"
```

### YouTube

Transcripts arrive as hundreds of two-second cues — too short to be meaningful
chunks, and none of them a whole thought. Two problems specific to spoken text:

- **No sentence boundaries.** Auto-generated captions have no punctuation, so a
  text chunker has nothing to split on. Grouping by *time* sidesteps that: a
  90-second window is a coherent unit regardless of punctuation.
- **Windows overlap in seconds, not characters**, so the repeated span is a real
  span of speech and a sentence crossing a boundary ends up whole in one window.

Each window is its own segment, so every chunk carries a start time and a
citation links to `?t=93s`. Caption noise markers (`[Music]`, `[Applause]`) are
stripped. Both API shapes of `youtube-transcript-api` are supported, since the
library moved from a classmethod to an instance method between major versions.

```bash
kb ingest 'https://youtu.be/VIDEO_ID'
kb ingest yt:VIDEO_ID
```

### Notion export

A Notion Markdown/CSV export has a very specific shape, and handling it is the
whole job:

- **Filenames carry a 32-hex page id** (`Runbooks a1b2c3….md`). It is stripped
  from the title — otherwise every page is titled with a hash — and *kept* as the
  page id, because it is what turns a citation into a `notion.so` link.
- **Nesting encodes hierarchy.** A subdirectory named after a page holds its
  children, so the ancestor path is reconstructed and a citation reads
  `Engineering › Runbooks › On-call` instead of a filename.
- **Databases export as CSV.** A row dumped verbatim embeds terribly — the values
  lose their field names. Each row becomes a `key: value` block under its title
  field instead, which reads well as a citation and gives BM25 and the embedder
  something to match. Notion's duplicate `_all.csv` view is skipped.
- **Filenames are percent-encoded**, so they are decoded or titles read as
  `On%20call%20rota`.
- Child-page link stubs and the property block Notion repeats after every title
  are stripped: they duplicate what the locator already carries, and leaving them
  in means every chunk of a page shares the same opening.

Handles a directory, a `.zip` (read without unpacking), or a single exported
file.

```bash
kb ingest ./Export-abc123
kb ingest ~/Downloads/notion-export.zip
```

## Writing a connector

Implement the `Connector` protocol:

```python
class MyConnector:
    name = "myformat"
    source_type = SourceType.TEXT

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def can_handle(self, source: str) -> bool:
        return source.endswith(".myformat")

    def parse(self, source: str, **options) -> Iterable[ParsedDocument]:
        text = Path(source).read_text()
        yield ParsedDocument(
            title=title_from_path(Path(source)),
            uri=str(Path(source).resolve()),
            source_type=self.source_type,
            segments=[Segment(text=text, build_locator=self._locator(source))],
            raw_text=text,
        )

    def _locator(self, path: str):
        def build(draft: ChunkDraft) -> Locator:
            return TextLocator(
                line_start=draft.line_start,
                line_end=draft.line_end,
                heading_path=draft.heading_path,
                file_path=path,
            )
        return build
```

Then register it — `registry.register(MyConnector(settings))`, or add it to
`_register_optional` if it depends on an optional extra.

Three rules worth following:

1. **One segment per addressable unit.** If your source has pages, timestamps or
   files, that is one segment each. It is the only way a chunk's address stays
   unambiguous.
2. **Raise `IngestionError` with an actionable message.** The pipeline collects it
   into `IngestionReport.errors` and keeps going, so one bad file never aborts a
   directory — but only if the message tells the user what to do.
3. **Put metadata in `metadata`, not in the text.** Anything that is not prose
   dilutes both the lexical index and the embeddings.
