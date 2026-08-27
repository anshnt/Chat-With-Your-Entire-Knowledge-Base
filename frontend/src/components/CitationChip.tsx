/**
 * An inline citation marker.
 *
 * A real `<button>`, not styled text: markers are the primary navigation in this
 * UI, so they have to be reachable by keyboard and announced to a screen reader.
 * The label says where the link goes ("page 12 of report.pdf"), because "[2]" on
 * its own is meaningless read aloud.
 */

import { renderInline } from "../lib/inline";
import type { AnswerCitation } from "../lib/types";
import { formatTimestamp, hostOf, isFollowable, sourceGlyph } from "../lib/locators";

interface Props {
  markers: number[];
  citations: AnswerCitation[];
  activeChunkId: string | null;
  onSelect: (citation: AnswerCitation) => void;
}

export function CitationChip({ markers, citations, activeChunkId, onSelect }: Props) {
  const resolved = markers
    .map((marker) => citations.find((c) => c.marker === marker))
    .filter((c): c is AnswerCitation => Boolean(c));

  if (resolved.length === 0) {
    // A marker the backend could not resolve should not render as a dead chip:
    // a citation that leads nowhere looks like evidence.
    return null;
  }

  return (
    <span className="inline-flex gap-0.5 align-baseline">
      {resolved.map((citation) => {
        const isActive = citation.chunk_id === activeChunkId;
        return (
          <button
            key={citation.marker}
            type="button"
            onClick={() => onSelect(citation)}
            title={describe(citation)}
            aria-label={`Citation ${citation.marker}: ${describe(citation)}`}
            className={[
              "inline-flex min-w-[1.35rem] items-center justify-center rounded",
              "px-1 text-[0.7rem] font-semibold leading-tight transition-colors",
              isActive
                ? "bg-blue-600 text-white"
                : "bg-blue-500/15 text-blue-700 hover:bg-blue-500/30 dark:text-blue-300",
            ].join(" ")}
          >
            {citation.marker}
          </button>
        );
      })}
    </span>
  );
}

function describe(citation: AnswerCitation): string {
  const parts = [citation.document_title];
  if (citation.label) parts.push(citation.label);
  return parts.join(" — ");
}

/**
 * The expanded source card shown under an answer.
 *
 * Deliberately shows the snippet inline. A citation you have to click to check
 * is one most readers will not check, which defeats the point.
 */
export function CitationCard({
  citation,
  isActive,
  onSelect,
}: {
  citation: AnswerCitation;
  isActive: boolean;
  onSelect: (citation: AnswerCitation) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onSelect(citation)}
      className={[
        "group w-full rounded-lg border p-3 text-left transition-colors",
        isActive
          ? "border-blue-500 bg-blue-500/5"
          : "border-slate-200 hover:border-slate-300 hover:bg-slate-100/60 dark:border-slate-800 dark:hover:border-slate-700 dark:hover:bg-slate-900",
      ].join(" ")}
    >
      <div className="flex items-start gap-2">
        <span className="mt-0.5 inline-flex min-w-[1.5rem] items-center justify-center rounded bg-blue-500/15 px-1 text-[0.7rem] font-semibold text-blue-700 dark:text-blue-300">
          {citation.marker}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-2">
            <span className="truncate text-sm font-medium">{citation.document_title}</span>
            <span className="shrink-0 rounded bg-slate-200 px-1 text-[0.6rem] font-semibold uppercase tracking-wide text-slate-600 dark:bg-slate-800 dark:text-slate-400">
              {sourceGlyph(citation.source_type)}
            </span>
          </div>
          {citation.label && (
            <div className="mt-0.5 text-xs text-slate-500">{citation.label}</div>
          )}
          <p className="mt-1.5 line-clamp-3 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
            {renderInline(citation.snippet)}
          </p>
          {citation.deep_link && (
            <div
              className={[
                "mt-1.5 truncate font-mono text-[0.65rem]",
                isFollowable(citation.deep_link)
                  ? "text-blue-600/80 dark:text-blue-400/80"
                  : "text-slate-400",
              ].join(" ")}
            >
              {prettyLink(citation.deep_link)}
            </div>
          )}
        </div>
      </div>
    </button>
  );
}

/** Shorten a deep link for display without losing the part that matters. */
function prettyLink(link: string): string {
  if (link.includes("#:~:text=")) {
    const [base] = link.split("#:~:text=");
    return `${hostOf(base ?? link)} → quoted passage`;
  }
  if (link.includes("&t=")) {
    const seconds = Number.parseInt(link.split("&t=")[1] ?? "0", 10);
    return `youtube.com → ${formatTimestamp(seconds)}`;
  }
  if (link.startsWith("http")) {
    try {
      const url = new URL(link);
      return `${url.host}${url.pathname}${url.hash}`;
    } catch {
      return link;
    }
  }
  return link;
}
