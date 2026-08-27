/**
 * Minimal inline-Markdown rendering for answer text.
 *
 * The extractive generator quotes source sentences **verbatim**, which is
 * exactly what makes it trustworthy — and means a sentence lifted from a
 * Markdown file arrives with its `**bold**` and `` `code` `` markers intact.
 * Showing those raw is the difference between "quoted from the source" and
 * "sloppy".
 *
 * A full Markdown renderer would be the wrong tool: it would add a dependency,
 * and block-level constructs (headings, lists, tables) have no meaning inside a
 * single answer sentence. So this handles the three inline forms that actually
 * appear and leaves everything else as literal text.
 *
 * Crucially it returns React elements, never HTML. The answer is model output;
 * routing it through `dangerouslySetInnerHTML` would be the obvious way to put
 * an XSS hole in an otherwise safe app.
 */

import type { ReactNode } from "react";

type Token =
  | { kind: "text"; value: string }
  | { kind: "code"; value: string }
  | { kind: "strong"; value: string }
  | { kind: "em"; value: string };

// Ordered longest-delimiter-first, so `**` is not mistaken for two `*`.
const PATTERN = /(`[^`]+`)|(\*\*[^*]+\*\*)|(__[^_]+__)|(\*[^*\s][^*]*\*)|(_[^_\s][^_]*_)/g;

export function tokenizeInline(text: string): Token[] {
  const tokens: Token[] = [];
  let cursor = 0;

  for (const match of text.matchAll(PATTERN)) {
    const start = match.index ?? 0;
    if (start > cursor) tokens.push({ kind: "text", value: text.slice(cursor, start) });

    const [, code, strongStar, strongUnderscore, emStar, emUnderscore] = match;
    if (code) tokens.push({ kind: "code", value: code.slice(1, -1) });
    else if (strongStar) tokens.push({ kind: "strong", value: strongStar.slice(2, -2) });
    else if (strongUnderscore) {
      tokens.push({ kind: "strong", value: strongUnderscore.slice(2, -2) });
    } else if (emStar) tokens.push({ kind: "em", value: emStar.slice(1, -1) });
    else if (emUnderscore) tokens.push({ kind: "em", value: emUnderscore.slice(1, -1) });

    cursor = start + match[0].length;
  }

  if (cursor < text.length) tokens.push({ kind: "text", value: text.slice(cursor) });
  return tokens;
}

/** Render inline Markdown as React elements. Never returns HTML. */
export function renderInline(text: string): ReactNode[] {
  return tokenizeInline(text).map((token, index) => {
    switch (token.kind) {
      case "code":
        return (
          <code
            key={index}
            className="rounded bg-slate-200/70 px-1 py-0.5 font-mono text-[0.85em] dark:bg-slate-800"
          >
            {token.value}
          </code>
        );
      case "strong":
        return (
          <strong key={index} className="font-semibold">
            {token.value}
          </strong>
        );
      case "em":
        return <em key={index}>{token.value}</em>;
      default:
        return <span key={index}>{token.value}</span>;
    }
  });
}

/**
 * Strip inline Markdown to plain text.
 *
 * Used where elements are not an option — a `title` attribute, an `aria-label`,
 * a tooltip — and where showing `**` would be worse than showing nothing.
 */
export function stripInline(text: string): string {
  return tokenizeInline(text)
    .map((token) => token.value)
    .join("");
}
