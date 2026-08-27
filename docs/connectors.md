# Connectors

A connector turns a *source* into `ParsedDocument`s. It does not chunk, embed, or
write to the store, and it does not decide how a citation is addressed beyond
supplying the callback that builds one.

## Available connectors

| Connector | Recognises | Locator | Extra needed |
|---|---|---|---|
| `markdown` | `.md`, `.markdown`, `.mdx`, `.mdown`, `.mkd` | `TextLocator` | — |
| `text` | `.txt`, `.text`, `.rst`, `.log`, `.csv`, `.tsv` | `TextLocator` | — |
| `pdf` | `.pdf` | `PdfLocator` | — |
| `inline` | `inline:<title>` | `TextLocator` | — |

Connectors for websites, GitHub repositories, YouTube transcripts and Notion
exports register themselves through `_register_optional` in
`kb/ingest/registry.py` when their module and dependencies are importable, so a
base install still ingests PDFs, Markdown and text without them.

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

Registration order is precedence order: a more specific connector (a Notion
export zip) must be registered ahead of a more general one.

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
