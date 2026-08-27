"""YouTube transcript connector.

Transcripts arrive as hundreds of two-second cues, which is the wrong unit for
retrieval — each is too short to be a meaningful chunk, and none of them is a
whole thought. So cues are grouped into time windows, and *each window is its own
segment*, which means every chunk carries a start time and a citation links to
``?t=93s``.

Two problems specific to spoken text:

**No sentence boundaries.** Auto-generated transcripts have no punctuation at all,
so the recursive chunker has nothing to split on and produces arbitrary cuts.
Grouping by *time* instead sidesteps that entirely: a 90-second window is a
coherent unit regardless of punctuation.

**Windows must overlap in time, not in characters.** A sentence spanning a window
boundary would otherwise be split across two chunks with neither containing it.
The overlap is expressed in seconds so the repeated span is a real span of speech.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse

from kb.chunking.base import ChunkDraft
from kb.config import Settings
from kb.errors import IngestionError, MissingDependencyError
from kb.ingest.base import ParsedDocument, Segment
from kb.models import ChunkKind, Locator, SourceType, YouTubeLocator

log = logging.getLogger(__name__)

#: Share of a window repeated at the start of the next one.
WINDOW_OVERLAP_RATIO = 0.15

_ID_RE = re.compile(r"^[\w-]{11}$")
_URL_HOSTS = ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be")

#: Filler that auto-captioning inserts and that carries no retrieval signal.
_NOISE_RE = re.compile(
    r"\[(?:music|applause|laughter|silence|inaudible|foreign|sound effects?)\]", re.I
)


@dataclass(slots=True)
class Cue:
    """One transcript cue."""

    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


class YouTubeConnector:
    """Ingests a video's transcript, addressed by timestamp."""

    name = "youtube"
    source_type = SourceType.YOUTUBE

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def can_handle(self, source: str) -> bool:
        if source.startswith(("yt:", "youtube:")):
            return True
        if not source.startswith(("http://", "https://")):
            return False
        return urlparse(source).netloc.lower() in _URL_HOSTS

    # ------------------------------------------------------------------ #

    def parse(self, source: str, **options: Any) -> Iterable[ParsedDocument]:
        video_id = extract_video_id(source)
        if video_id is None:
            raise IngestionError(f"could not extract a video id from {source!r}")

        languages = options.get("languages") or ["en", "en-US", "en-GB"]
        window_seconds = float(
            options.get("window_seconds", self.settings.transcript_chunk_seconds)
        )

        cues = self._fetch_transcript(video_id, languages)
        if not cues:
            raise IngestionError(
                f"no transcript available for {video_id}. The video may have captions "
                "disabled, or none in the requested languages."
            )

        windows = group_into_windows(cues, window_seconds)
        if not windows:
            raise IngestionError(f"transcript for {video_id} produced no usable text")

        segments = [
            Segment(
                text=text,
                build_locator=_youtube_locator_factory(video_id, start, end),
                kind=ChunkKind.TRANSCRIPT,
                metadata={"start_seconds": round(start, 2), "end_seconds": round(end, 2)},
            )
            for text, start, end in windows
        ]

        title = str(options.get("title") or f"YouTube transcript {video_id}")
        total = windows[-1][2]
        return [
            ParsedDocument(
                title=title,
                uri=f"https://www.youtube.com/watch?v={video_id}",
                source_type=self.source_type,
                segments=segments,
                raw_text="\n\n".join(s.text for s in segments),
                byte_size=sum(len(s.text.encode("utf-8")) for s in segments),
                metadata={
                    "video_id": video_id,
                    "n_cues": len(cues),
                    "duration_seconds": round(total, 1),
                    "window_seconds": window_seconds,
                },
            )
        ]

    def _fetch_transcript(self, video_id: str, languages: Sequence[str]) -> list[Cue]:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError as exc:
            raise MissingDependencyError("youtube-transcript-api", "youtube") from exc

        try:
            raw = _call_transcript_api(YouTubeTranscriptApi, video_id, list(languages))
        except Exception as exc:
            raise IngestionError(f"could not fetch the transcript for {video_id}: {exc}") from exc

        cues: list[Cue] = []
        for item in raw:
            text = clean_cue(_cue_field(item, "text"))
            if not text:
                continue
            cues.append(
                Cue(
                    text=text,
                    start=float(_cue_field(item, "start") or 0.0),
                    duration=float(_cue_field(item, "duration") or 0.0),
                )
            )
        return cues


def _call_transcript_api(api: Any, video_id: str, languages: list[str]) -> Any:
    """Call whichever API shape the installed version exposes.

    ``youtube-transcript-api`` changed from a classmethod (``get_transcript``) to
    an instance method (``fetch``) between major versions, and pinning would mean
    breaking on either side of the change.
    """
    if hasattr(api, "get_transcript"):
        return api.get_transcript(video_id, languages=languages)
    instance = api()
    fetched = instance.fetch(video_id, languages=languages)
    return getattr(fetched, "snippets", fetched)


def _cue_field(item: Any, name: str) -> Any:
    """Read a field from either a dict cue or an object cue."""
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def extract_video_id(source: str) -> str | None:
    """Pull the 11-character video id out of any of YouTube's URL shapes."""
    if source.startswith(("yt:", "youtube:")):
        candidate = source.split(":", 1)[1]
        return candidate if _ID_RE.match(candidate) else None
    if _ID_RE.match(source):
        return source

    parsed = urlparse(source)
    host = parsed.netloc.lower()
    if host.endswith("youtu.be"):
        candidate = parsed.path.lstrip("/").split("/")[0]
        return candidate if _ID_RE.match(candidate) else None

    if "youtube.com" in host:
        query = parse_qs(parsed.query)
        if "v" in query and _ID_RE.match(query["v"][0]):
            return query["v"][0]
        # /embed/ID, /shorts/ID, /live/ID
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in ("embed", "shorts", "live", "v"):
            return parts[1] if _ID_RE.match(parts[1]) else None
    return None


def clean_cue(text: Any) -> str:
    """Strip caption noise markers and collapse whitespace."""
    if not text:
        return ""
    cleaned = _NOISE_RE.sub(" ", str(text))
    cleaned = cleaned.replace("\n", " ").replace("&amp;#39;", "'").replace("&amp;quot;", '"')
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def group_into_windows(
    cues: Sequence[Cue], window_seconds: float, overlap_ratio: float = WINDOW_OVERLAP_RATIO
) -> list[tuple[str, float, float]]:
    """Group cues into overlapping time windows.

    Returns ``(text, start_seconds, end_seconds)``. Overlap is measured in
    *seconds*, not characters, so the repeated span is a real span of speech —
    a sentence crossing a boundary ends up whole in one of the two windows.
    """
    if not cues or window_seconds <= 0:
        return []

    overlap = max(0.0, window_seconds * overlap_ratio)
    windows: list[tuple[str, float, float]] = []
    index = 0

    while index < len(cues):
        window_start = cues[index].start
        limit = window_start + window_seconds
        parts: list[str] = []
        cursor = index
        while cursor < len(cues) and cues[cursor].start < limit:
            parts.append(cues[cursor].text)
            cursor += 1
        if cursor == index:  # a single cue longer than the window
            parts.append(cues[index].text)
            cursor = index + 1

        window_end = cues[cursor - 1].end or limit
        text = re.sub(r"\s{2,}", " ", " ".join(parts)).strip()
        if text:
            windows.append((text, window_start, window_end))

        if cursor >= len(cues):
            break
        # Step the next window back by the overlap, without ever standing still.
        resume_at = window_end - overlap
        next_index = cursor
        while next_index > index + 1 and cues[next_index - 1].start >= resume_at:
            next_index -= 1
        index = max(next_index, index + 1)

    return windows


def _youtube_locator_factory(video_id: str, start: float, end: float) -> Any:
    def build(draft: ChunkDraft) -> Locator:  # noqa: ARG001 - the window owns the time
        return YouTubeLocator(video_id=video_id, start_seconds=start, end_seconds=end)

    return build
