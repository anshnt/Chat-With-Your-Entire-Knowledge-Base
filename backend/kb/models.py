"""Core domain models.

The most important type here is :class:`Locator`. A citation is only useful if a
reader can *land on the exact place the claim came from*, and every source type
needs a different kind of address to do that:

===============  ==========================================================
Source           Address that actually jumps somewhere
===============  ==========================================================
PDF              page number (+ char span for highlighting)
Markdown / text  heading path + line range
Website          URL + a ``#:~:text=`` scroll-to-text fragment
GitHub           blob URL + ``#L10-L20``
YouTube          video id + start seconds (``?t=93``)
Notion export    page title path + line range
===============  ==========================================================

So ``Locator`` is a discriminated union with a ``deep_link()`` method per
variant. Retrieval carries the locator along with the chunk, and the UI turns it
into a link. Nothing else in the system needs to know how PDFs differ from
YouTube.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal
from urllib.parse import quote, urlencode, urlparse, urlunparse

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utcnow() -> datetime:
    """Timezone-aware UTC now. Used everywhere instead of ``datetime.utcnow``."""
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    """Short, sortable-ish, human-greppable identifier, e.g. ``doc_9f2a1c...``."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def content_hash(text: str) -> str:
    """Stable content fingerprint, used to deduplicate chunks and documents."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SourceType(str, Enum):
    """Where a document came from. Determines which connector parsed it."""

    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"
    WEB = "web"
    GITHUB = "github"
    YOUTUBE = "youtube"
    NOTION = "notion"


class ChunkKind(str, Enum):
    """What kind of content a chunk holds. Chunkers and rerankers use this."""

    PROSE = "prose"
    CODE = "code"
    TABLE = "table"
    HEADING = "heading"
    TRANSCRIPT = "transcript"
    LIST = "list"


# --------------------------------------------------------------------------- #
# Locators
# --------------------------------------------------------------------------- #


class _BaseLocator(BaseModel):
    """Shared behaviour for every locator variant."""

    model_config = ConfigDict(frozen=True)

    def deep_link(self) -> str | None:
        """A URL that scrolls/seeks to this exact position, when one exists."""
        return None

    def label(self) -> str:
        """Short human-readable position, e.g. ``p. 12`` or ``2:14``."""
        return ""


class PdfLocator(_BaseLocator):
    """Position inside a PDF: a 1-based page, optionally a char span on it."""

    kind: Literal["pdf"] = "pdf"
    page: int = Field(ge=1, description="1-based page number")
    page_count: int | None = Field(default=None, ge=1)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    file_url: str | None = Field(
        default=None, description="Served URL of the PDF, used to build the viewer link"
    )

    def deep_link(self) -> str | None:
        if not self.file_url:
            return None
        # PDF open parameters (RFC 8118 / Adobe): #page=N is honoured by
        # Chrome's built-in viewer, Firefox pdf.js, and Safari.
        return f"{self.file_url}#page={self.page}"

    def label(self) -> str:
        if self.page_count:
            return f"p. {self.page} / {self.page_count}"
        return f"p. {self.page}"


class TextLocator(BaseModel):
    """Position inside a plain-text, Markdown, or Notion document."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["text"] = "text"
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    heading_path: list[str] = Field(
        default_factory=list, description="Ancestor headings, outermost first"
    )
    file_path: str | None = None

    def deep_link(self) -> str | None:
        if not self.file_path:
            return None
        return f"{self.file_path}#L{self.line_start}-L{self.line_end}"

    def label(self) -> str:
        if self.heading_path:
            return " › ".join(self.heading_path[-2:])
        if self.line_start == self.line_end:
            return f"line {self.line_start}"
        return f"lines {self.line_start}-{self.line_end}"


class WebLocator(BaseModel):
    """Position inside a web page, addressed by a scroll-to-text fragment.

    Chromium, Edge and Safari implement the Text Fragments spec
    (``#:~:text=start,end``), so the browser scrolls to and highlights the quote
    without us needing to control the page.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["web"] = "web"
    url: str
    css_selector: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    quote_prefix: str | None = Field(
        default=None, description="First few words of the chunk, for the text fragment"
    )
    quote_suffix: str | None = Field(
        default=None, description="Last few words of the chunk, for the text fragment"
    )

    def deep_link(self) -> str | None:
        if not self.quote_prefix:
            return self.url
        base = self.url.split("#")[0]
        start = quote(self.quote_prefix.strip(), safe="")
        if self.quote_suffix and self.quote_suffix.strip() != self.quote_prefix.strip():
            end = quote(self.quote_suffix.strip(), safe="")
            return f"{base}#:~:text={start},{end}"
        return f"{base}#:~:text={start}"

    def label(self) -> str:
        if self.heading_path:
            return " › ".join(self.heading_path[-2:])
        host = urlparse(self.url).netloc
        return host or self.url


class GitHubLocator(BaseModel):
    """Position inside a file in a GitHub repository."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["github"] = "github"
    repo: str = Field(description="``owner/name``")
    ref: str = Field(default="HEAD", description="Branch, tag, or commit SHA")
    path: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    symbol: str | None = Field(default=None, description="Enclosing function/class, if known")

    def deep_link(self) -> str | None:
        anchor = (
            f"#L{self.line_start}"
            if self.line_start == self.line_end
            else f"#L{self.line_start}-L{self.line_end}"
        )
        return f"https://github.com/{self.repo}/blob/{self.ref}/{self.path}{anchor}"

    def label(self) -> str:
        base = f"{self.path}:{self.line_start}"
        return f"{base} ({self.symbol})" if self.symbol else base


class YouTubeLocator(BaseModel):
    """Position inside a YouTube transcript, addressed by start time."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["youtube"] = "youtube"
    video_id: str
    start_seconds: float = Field(ge=0)
    end_seconds: float | None = Field(default=None, ge=0)

    def deep_link(self) -> str | None:
        params = urlencode({"v": self.video_id, "t": f"{int(self.start_seconds)}s"})
        return f"https://www.youtube.com/watch?{params}"

    def label(self) -> str:
        return _format_timestamp(self.start_seconds)


class NotionLocator(BaseModel):
    """Position inside a page of a Notion export."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["notion"] = "notion"
    page_path: list[str] = Field(default_factory=list, description="Notion page hierarchy")
    notion_page_id: str | None = None
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    heading_path: list[str] = Field(default_factory=list)

    def deep_link(self) -> str | None:
        if not self.notion_page_id:
            return None
        return f"https://www.notion.so/{self.notion_page_id.replace('-', '')}"

    def label(self) -> str:
        if self.heading_path:
            return " › ".join(self.heading_path[-2:])
        if self.page_path:
            return self.page_path[-1]
        return f"lines {self.line_start}-{self.line_end}"


Locator = Annotated[
    PdfLocator | TextLocator | WebLocator | GitHubLocator | YouTubeLocator | NotionLocator,
    Field(discriminator="kind"),
]

LOCATOR_TYPES: dict[str, type[BaseModel]] = {
    "pdf": PdfLocator,
    "text": TextLocator,
    "web": WebLocator,
    "github": GitHubLocator,
    "youtube": YouTubeLocator,
    "notion": NotionLocator,
}


def parse_locator(payload: dict[str, Any]) -> Locator:
    """Rebuild a locator from its serialised form (used when reading the store)."""
    kind = payload.get("kind")
    cls = LOCATOR_TYPES.get(str(kind))
    if cls is None:
        raise ValueError(f"unknown locator kind: {kind!r}")
    return cls.model_validate(payload)  # type: ignore[return-value]


def _format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


# --------------------------------------------------------------------------- #
# Documents and chunks
# --------------------------------------------------------------------------- #


class Document(BaseModel):
    """A single ingested source: one PDF, one web page, one repo file, one video."""

    id: str = Field(default_factory=lambda: new_id("doc"))
    collection: str = "default"
    source_type: SourceType
    title: str
    uri: str = Field(description="Canonical origin: file path, URL, repo path, or video URL")
    content_hash: str = ""
    byte_size: int = 0
    token_estimate: int = 0
    n_chunks: int = 0
    language: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("title")
    @classmethod
    def _title_not_blank(cls, v: str) -> str:
        v = v.strip()
        return v or "Untitled"


class Chunk(BaseModel):
    """A retrievable unit of text plus the address it came from."""

    id: str = Field(default_factory=lambda: new_id("chk"))
    document_id: str
    collection: str = "default"
    ordinal: int = Field(ge=0, description="Position of this chunk within its document")
    text: str
    kind: ChunkKind = ChunkKind.PROSE
    locator: Locator
    token_estimate: int = 0
    content_hash: str = ""
    # Denormalised for display and for lexical boosting; avoids a join per hit.
    document_title: str = ""
    source_type: SourceType | None = None
    heading_context: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.content_hash:
            object.__setattr__(self, "content_hash", content_hash(self.text))
        if not self.token_estimate:
            object.__setattr__(self, "token_estimate", estimate_tokens(self.text))

    def deep_link(self) -> str | None:
        return self.locator.deep_link()

    def position_label(self) -> str:
        """Position within the document, with a redundant title prefix removed.

        A leading heading identical to the document title is dropped, because
        "Design Doc — Design Doc › Retrieval" reads as a bug even though both
        halves are correct.
        """
        pos = self.locator.label()
        title = self.document_title.strip()
        if pos and title and pos.startswith(title):
            pos = pos[len(title) :].lstrip(" ›").strip()
        return pos

    def citation_label(self) -> str:
        """``Design Doc — p. 4``: what a citation chip shows."""
        pos = self.position_label()
        title = self.document_title.strip()
        return f"{title} — {pos}" if pos else title


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) so we never need a tokenizer here."""
    return max(1, len(text) // 4) if text else 0


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


class RetrievalStrategy(str, Enum):
    """Which retrievers contribute candidates."""

    LEXICAL = "lexical"
    DENSE = "dense"
    HYBRID = "hybrid"


class FusionMethod(str, Enum):
    """How lexical and dense rankings are combined."""

    RRF = "rrf"
    """Reciprocal Rank Fusion: rank-based, scale-free, the robust default."""

    WEIGHTED = "weighted"
    """Min-max normalise each score list, then take a weighted sum."""

    MAX = "max"
    """Take the best normalised score per document (high recall, noisier)."""


class ScoredChunk(BaseModel):
    """A chunk with the scores that got it here.

    Keeping the per-retriever scores (rather than collapsing to one number) is
    what makes the retrieval debuggable — and it's what the evaluation harness
    reports on.
    """

    chunk: Chunk
    score: float = Field(description="Final score after fusion/reranking")
    lexical_score: float | None = None
    dense_score: float | None = None
    lexical_rank: int | None = None
    dense_rank: int | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    retrievers: list[str] = Field(default_factory=list)

    @property
    def id(self) -> str:
        return self.chunk.id

    def explain(self) -> str:
        """One-line score breakdown for logs and the debug panel."""
        parts = [f"final={self.score:.4f}"]
        if self.lexical_score is not None:
            parts.append(f"bm25={self.lexical_score:.4f}@{self.lexical_rank}")
        if self.dense_score is not None:
            parts.append(f"dense={self.dense_score:.4f}@{self.dense_rank}")
        if self.fusion_score is not None:
            parts.append(f"fused={self.fusion_score:.4f}")
        if self.rerank_score is not None:
            parts.append(f"rerank={self.rerank_score:.4f}")
        return " ".join(parts)


class RetrievalRequest(BaseModel):
    """Everything that determines what comes back from a search."""

    query: str = Field(min_length=1)
    collection: str = "default"
    top_k: int = Field(default=8, ge=1, le=200)
    candidate_k: int = Field(
        default=50, ge=1, le=1000, description="Candidates fetched per retriever before fusion"
    )
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    fusion: FusionMethod = FusionMethod.RRF
    lexical_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1, description="RRF damping constant")
    use_mmr: bool = Field(default=False, description="Diversify results with MMR")
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    rerank: bool = Field(default=False)
    rerank_top_n: int = Field(default=30, ge=1, le=200)
    source_types: list[SourceType] | None = None
    document_ids: list[str] | None = None
    min_score: float | None = None

    @field_validator("query")
    @classmethod
    def _strip_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v


class RetrievalResult(BaseModel):
    """Search results plus the timings and counts needed to explain them."""

    query: str
    results: list[ScoredChunk]
    strategy: RetrievalStrategy
    fusion: FusionMethod | None = None
    reranked: bool = False
    lexical_candidates: int = 0
    dense_candidates: int = 0
    fused_candidates: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)

    @property
    def chunks(self) -> list[Chunk]:
        return [r.chunk for r in self.results]

    def total_ms(self) -> float:
        return round(sum(self.timings_ms.values()), 2)


# --------------------------------------------------------------------------- #
# Answers
# --------------------------------------------------------------------------- #


class AnswerCitation(BaseModel):
    """A source the answer cites, resolved to a clickable position.

    ``marker`` is the number the reader sees in the text (``[2]``). It indexes
    the prompt's source list, not the corpus — the model is never shown a chunk
    id, because it will happily invent one that looks plausible.
    """

    marker: int = Field(ge=1, description="The [n] shown in the answer text")
    chunk_id: str
    document_id: str
    document_title: str
    label: str = Field(description="Position within the source, e.g. 'p. 12'")
    deep_link: str | None = None
    snippet: str = ""
    source_type: SourceType | None = None
    #: The serialised locator. Carried alongside ``deep_link`` because a client
    #: needs the structured position for decisions the link cannot express — a
    #: PDF page can be embedded in place, a website has to open in a new tab —
    #: and reverse-engineering that from a URL string is guesswork.
    locator: dict[str, Any] = Field(default_factory=dict)
    heading_context: str = ""

    @classmethod
    def from_chunk(cls, marker: int, chunk: Chunk, *, snippet_chars: int = 320) -> AnswerCitation:
        body = chunk.text.strip()
        return cls(
            marker=marker,
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_title=chunk.document_title,
            label=chunk.position_label(),
            deep_link=chunk.deep_link(),
            snippet=body if len(body) <= snippet_chars else f"{body[:snippet_chars].rstrip()}…",
            source_type=chunk.source_type,
            locator=chunk.locator.model_dump(mode="json"),
            heading_context=chunk.heading_context,
        )


class SupportVerdict(str, Enum):
    """Whether a cited source actually backs the claim it is attached to."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    UNCITED = "uncited"
    """The sentence makes a factual claim but cites nothing."""

    NOT_A_CLAIM = "not_a_claim"
    """Framing, transitions, questions — nothing to verify."""


class AnswerSentence(BaseModel):
    """One sentence of an answer, with the citations attached to it.

    Sentence-level granularity is what makes verification meaningful: an answer
    is not uniformly true or false, and "paragraph 2 is unsupported" is not
    actionable while "this clause is unsupported" is.
    """

    text: str
    citation_markers: list[int] = Field(default_factory=list)
    char_start: int = 0
    char_end: int = 0
    # Filled in by the verification stage.
    support_score: float | None = None
    verdict: SupportVerdict | None = None
    supporting_quote: str | None = None
    verification_note: str | None = None

    @property
    def is_cited(self) -> bool:
        return bool(self.citation_markers)


class Answer(BaseModel):
    """A generated answer, its sources, and the evidence for trusting it."""

    query: str
    text: str
    citations: list[AnswerCitation] = Field(default_factory=list)
    sentences: list[AnswerSentence] = Field(default_factory=list)
    generator: str = ""
    model: str = ""
    context_chunks: int = 0
    context_tokens: int = 0
    refused: bool = Field(
        default=False, description="True when the context did not support an answer"
    )
    faithfulness: float | None = Field(
        default=None, description="Share of claim sentences that are supported"
    )
    verified: bool = False
    timings_ms: dict[str, float] = Field(default_factory=dict)
    retrieval: RetrievalResult | None = None

    def citation_for(self, marker: int) -> AnswerCitation | None:
        return next((c for c in self.citations if c.marker == marker), None)

    def unsupported_sentences(self) -> list[AnswerSentence]:
        """Sentences whose cited source does not support them, or that cite nothing."""
        return [
            s
            for s in self.sentences
            if s.verdict in (SupportVerdict.UNSUPPORTED, SupportVerdict.UNCITED)
        ]

    def flagged_sentences(self) -> list[AnswerSentence]:
        """Every sentence a reader should not take on trust.

        Wider than :meth:`unsupported_sentences`: it includes ``partial``, because
        "the source is related but does not quite say this" is exactly the case a
        reader most needs pointed out — it is the one that reads as fine.
        """
        return [
            s
            for s in self.sentences
            if s.verdict
            in (
                SupportVerdict.UNSUPPORTED,
                SupportVerdict.UNCITED,
                SupportVerdict.PARTIAL,
            )
        ]

    def total_ms(self) -> float:
        return round(sum(self.timings_ms.values()), 2)


class IngestionReport(BaseModel):
    """What an ingestion run actually did — surfaced by CLI and API alike."""

    documents: list[Document] = Field(default_factory=list)
    chunks_created: int = 0
    documents_skipped: int = 0
    duplicates_skipped: int = 0
    errors: list[dict[str, str]] = Field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def documents_created(self) -> int:
        return len(self.documents)

    def merge(self, other: IngestionReport) -> IngestionReport:
        return IngestionReport(
            documents=self.documents + other.documents,
            chunks_created=self.chunks_created + other.chunks_created,
            documents_skipped=self.documents_skipped + other.documents_skipped,
            duplicates_skipped=self.duplicates_skipped + other.duplicates_skipped,
            errors=self.errors + other.errors,
            elapsed_ms=self.elapsed_ms + other.elapsed_ms,
        )


class CollectionStats(BaseModel):
    """Corpus summary: what's in here, by source and by size."""

    collection: str
    n_documents: int = 0
    n_chunks: int = 0
    n_embedded: int = 0
    total_tokens: int = 0
    by_source_type: dict[str, int] = Field(default_factory=dict)
    embedding_model: str | None = None
    embedding_dim: int | None = None


def with_text_fragment(url: str, snippet: str) -> str:
    """Attach a scroll-to-text fragment to ``url`` (used by the web connector)."""
    parsed = urlparse(url)
    stripped = urlunparse(parsed._replace(fragment=""))
    return f"{stripped}#:~:text={quote(snippet.strip(), safe='')}"
