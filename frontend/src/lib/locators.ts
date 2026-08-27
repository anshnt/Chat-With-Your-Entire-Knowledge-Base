/**
 * Turning a locator into something a reader can click.
 *
 * The backend already computes `deep_link`, and that is what the UI uses. This
 * module exists for the two things the backend cannot decide:
 *
 * - **Whether the link opens in place or in a new tab.** A PDF page or a chunk
 *   of local Markdown belongs in the side pane; a website or a GitHub permalink
 *   belongs in a new tab. Getting that wrong is the difference between a source
 *   viewer and a page that keeps navigating away from itself.
 * - **How to describe the destination.** "p. 12 of report.pdf" and "2:14 in the
 *   video" are the same field with different words, and a citation chip that
 *   says "text" is useless.
 */

import type { AnswerCitation, Citation, Locator, SourceType } from "./types";

/** The fields both citation shapes share, which is all this module needs. */
type AnyCitation = Pick<Citation | AnswerCitation, "locator" | "document_title">;

export type OpenMode = "pane" | "external";

/**
 * Whether a deep link will actually resolve in a browser.
 *
 * A `TextLocator` for a local Markdown file produces `docs/architecture.md#L154`,
 * which is a perfectly good address for an editor and a dead link in a browser.
 * Rendering it anyway is the exact failure this project criticises elsewhere: a
 * link that looks like evidence and goes nowhere. So the UI checks first and
 * shows the passage instead.
 */
export function isFollowable(link: string | null): boolean {
  if (!link) return false;
  // Absolute URLs, and paths the API serves itself (uploaded files).
  return /^https?:\/\//.test(link) || link.startsWith("/files/");
}

/** Where a citation should open. */
export function openMode(locator: Locator): OpenMode {
  switch (locator.kind) {
    case "pdf":
      // Served by the API from its uploads directory, so it can be framed.
      return locator.file_url ? "pane" : "external";
    case "text":
    case "notion":
      // A local path is not fetchable by the browser; show the chunk instead.
      return "pane";
    case "web":
    case "github":
    case "youtube":
      return "external";
  }
}

/** Human description of the destination, for a tooltip or aria-label. */
export function describeDestination(citation: AnyCitation): string {
  const { locator, document_title: title } = citation;
  switch (locator.kind) {
    case "pdf":
      return locator.page_count
        ? `page ${locator.page} of ${locator.page_count} in ${title}`
        : `page ${locator.page} of ${title}`;
    case "text":
      return locator.line_start === locator.line_end
        ? `line ${locator.line_start} of ${title}`
        : `lines ${locator.line_start}-${locator.line_end} of ${title}`;
    case "web":
      return `the quoted passage on ${hostOf(locator.url ?? "")}`;
    case "github":
      return `${locator.path}:${locator.line_start}` +
        (locator.symbol ? ` (${locator.symbol})` : "") +
        ` in ${locator.repo}`;
    case "youtube":
      return `${formatTimestamp(locator.start_seconds ?? 0)} in the video`;
    case "notion":
      return (locator.page_path ?? []).join(" › ") || title;
  }
}

/** A short icon-ish glyph per source type, for the citation chip. */
export function sourceGlyph(sourceType: SourceType | null): string {
  switch (sourceType) {
    case "pdf":
      return "PDF";
    case "web":
    case "html":
      return "WEB";
    case "github":
      return "GIT";
    case "youtube":
      return "VID";
    case "notion":
      return "NTN";
    case "markdown":
      return "MD";
    default:
      return "TXT";
  }
}

export function formatTimestamp(seconds: number): string {
  const total = Math.floor(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return hours > 0 ? `${hours}:${pad(minutes)}:${pad(secs)}` : `${minutes}:${pad(secs)}`;
}

export function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

/**
 * Split answer text into citation markers and the prose between them.
 *
 * Done here rather than with `dangerouslySetInnerHTML` so a marker becomes a
 * real interactive element — focusable, keyboard-activatable — instead of styled
 * text. The answer is model output; injecting it as HTML would be the obvious
 * way to introduce an XSS hole into an otherwise safe app.
 */
export type AnswerSegment =
  | { kind: "text"; value: string }
  | { kind: "markers"; markers: number[]; raw: string };

const MARKER_RE = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

export function segmentAnswer(text: string): AnswerSegment[] {
  const segments: AnswerSegment[] = [];
  let cursor = 0;

  for (const match of text.matchAll(MARKER_RE)) {
    const start = match.index ?? 0;
    if (start > cursor) {
      segments.push({ kind: "text", value: text.slice(cursor, start) });
    }
    const markers = match[1]!
      .split(",")
      .map((part) => Number.parseInt(part.trim(), 10))
      .filter((value) => Number.isFinite(value));
    segments.push({ kind: "markers", markers, raw: match[0] });
    cursor = start + match[0].length;
  }

  if (cursor < text.length) {
    segments.push({ kind: "text", value: text.slice(cursor) });
  }
  return segments;
}
