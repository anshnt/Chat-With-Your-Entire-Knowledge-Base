/**
 * The source pane: where a citation actually lands.
 *
 * This is the component the whole `Locator` design exists to serve, and the rule
 * it follows is that **the reader should never have to leave to check a claim**.
 * So:
 *
 * - a PDF citation embeds the PDF at the cited page, in place;
 * - a local text or Notion citation shows the chunk with its neighbours, since
 *   the browser cannot fetch a local path and a chunk out of context reads badly;
 * - a web, GitHub or YouTube citation opens externally — but the pane still shows
 *   the quoted text first, so the claim can be checked without the round trip.
 */

import { useEffect, useState } from "react";

import { api, ApiError } from "../lib/api";
import { renderInline } from "../lib/inline";
import { describeDestination, isFollowable, openMode, sourceGlyph } from "../lib/locators";
import type { AnswerCitation, ChunkView } from "../lib/types";

interface Props {
  citation: AnswerCitation | null;
  onClose: () => void;
}

export function SourcePane({ citation, onClose }: Props) {
  const [context, setContext] = useState<ChunkView[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!citation) {
      setContext(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);

    api
      .chunkContext(citation.chunk_id, 1)
      .then((response) => {
        if (!cancelled) setContext(response.chunks);
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(cause instanceof ApiError ? cause.message : "Could not load the source");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [citation]);

  // Escape closes the pane. It overlays content on narrow screens, so leaving a
  // reader without a keyboard exit would trap them.
  useEffect(() => {
    if (!citation) return;
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [citation, onClose]);

  if (!citation) {
    return (
      <aside className="hidden h-full flex-col items-center justify-center gap-2 border-l border-slate-200 p-6 text-center text-sm text-slate-500 lg:flex dark:border-slate-800">
        <p className="max-w-[18rem]">
          Click a citation to see the source it came from, at the exact page, line
          range or timestamp.
        </p>
      </aside>
    );
  }

  const mode = openMode(citation.locator);
  const pdfUrl =
    citation.locator.kind === "pdf" && citation.locator.file_url
      ? `${citation.locator.file_url}#page=${citation.locator.page ?? 1}`
      : null;

  return (
    <aside
      aria-label="Source"
      className="flex h-full flex-col border-l border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900"
    >
      <header className="flex items-start gap-2 border-b border-slate-200 p-3 dark:border-slate-800">
        <span className="mt-0.5 shrink-0 rounded bg-slate-200 px-1.5 py-0.5 text-[0.6rem] font-semibold uppercase tracking-wide text-slate-600 dark:bg-slate-800 dark:text-slate-400">
          {sourceGlyph(citation.source_type)}
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="truncate text-sm font-medium">{citation.document_title}</h2>
          <p className="mt-0.5 text-xs text-slate-500">{describeDestination(citation)}</p>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close the source pane"
          className="shrink-0 rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600 dark:hover:bg-slate-800"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden>
            <path
              d="M4 4l8 8M12 4l-8 8"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              fill="none"
            />
          </svg>
        </button>
      </header>

      <div className="scrollbar-thin min-h-0 flex-1 overflow-y-auto">
        {/* The cited text first, always. Checking a claim should not require a
            network round trip to another site. */}
        <section className="border-b border-slate-200 p-3 dark:border-slate-800">
          <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Cited passage
          </h3>
          <blockquote className="border-l-2 border-blue-500 pl-3 text-sm leading-relaxed">
            {renderInline(citation.snippet)}
          </blockquote>
          {isFollowable(citation.deep_link) ? (
            <a
              href={citation.deep_link ?? undefined}
              target={mode === "external" ? "_blank" : undefined}
              rel={mode === "external" ? "noreferrer noopener" : undefined}
              className="mt-2 inline-flex items-center gap-1 text-xs font-medium text-blue-600 hover:underline dark:text-blue-400"
            >
              {mode === "external" ? "Open the source" : "Open the file"}
              <span aria-hidden>→</span>
            </a>
          ) : (
            citation.deep_link && (
              <p className="mt-2 font-mono text-[0.65rem] text-slate-400">
                {citation.deep_link}
              </p>
            )
          )}
        </section>

        {pdfUrl && (
          <section className="border-b border-slate-200 dark:border-slate-800">
            <h3 className="px-3 pt-3 text-xs font-semibold uppercase tracking-wide text-slate-500">
              Page {citation.locator.page}
            </h3>
            <iframe
              src={pdfUrl}
              title={`${citation.document_title}, page ${citation.locator.page}`}
              className="mt-2 h-[28rem] w-full border-0"
            />
          </section>
        )}

        <section className="p-3">
          <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">
            In context
          </h3>
          {loading && <p className="text-xs text-slate-500">Loading…</p>}
          {error && <p className="text-xs text-red-600 dark:text-red-400">{error}</p>}
          {context && (
            <div className="space-y-2">
              {context.map((chunk) => {
                const isFocus = chunk.id === citation.chunk_id;
                return (
                  <div
                    key={chunk.id}
                    className={[
                      "rounded border p-2 text-xs leading-relaxed",
                      isFocus
                        ? "border-blue-500 bg-blue-500/5"
                        : "border-slate-200 text-slate-500 dark:border-slate-800",
                    ].join(" ")}
                  >
                    <div className="mb-1 flex items-baseline justify-between gap-2 text-[0.65rem] text-slate-400">
                      <span>{chunk.citation.label || `chunk ${chunk.ordinal}`}</span>
                      <span>{chunk.kind}</span>
                    </div>
                    <p className="whitespace-pre-wrap">{renderInline(chunk.text)}</p>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      </div>
    </aside>
  );
}
