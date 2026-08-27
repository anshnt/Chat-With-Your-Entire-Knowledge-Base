# Frontend

React + TypeScript + Vite + Tailwind. No charting library, no Markdown library,
no state-management library — the app is small enough that each of those would
add more bundle and indirection than it removes.

```bash
npm install
npm run dev        # http://localhost:5173, proxying /api to :8000
```

The API is **proxied** rather than configured by an environment variable, so the
app is same-origin in development and needs no CORS handling or `.env` to run.
Start the backend with `kb serve` in another shell.

## Layout

| File | Responsibility |
|---|---|
| `lib/types.ts` | The API response shapes this UI actually reads |
| `lib/api.ts` | Typed errors, and SSE parsing that survives chunk boundaries |
| `lib/locators.ts` | Where a citation opens, and how to describe it |
| `lib/verdicts.ts` | Verdict colour, label and underline — defined once |
| `lib/inline.tsx` | Inline-Markdown → React elements (never HTML) |
| `hooks/useAsk.ts` | Streaming, abort-on-new-question, then the verified answer |
| `components/AnswerView.tsx` | The answer, with per-sentence verdicts |
| `components/SourcePane.tsx` | Where a citation lands |
| `components/CorpusMap.tsx` | The scatter plot, hand-drawn SVG |
| `components/IngestPanel.tsx` | Drop a file, paste text, or give a path/URL |

## Decisions worth knowing before changing things

**Types are hand-written, not generated.** The set of fields this UI reads is
much smaller than the API surface, and writing them out makes the contract
explicit: anything not in `types.ts` is not relied on, so the backend is free to
change it.

**SSE parsing buffers across reads.** `fetch` hands back arbitrary byte chunks,
not events — an event can straddle two of them, so a per-chunk
`split("\n\n")` silently truncates the tail of every answer. Only complete
events are emitted and the remainder is carried forward.

**Citations arrive with the terminal event, not mid-stream.** A marker can be
half-emitted, and verification needs whole sentences, so the UI shows raw
streamed text first and swaps in the structured answer when it completes.
Rendering a chip for a partial `[` would be worse than waiting a beat.

**In-flight requests are aborted on a new question**, so a slow answer cannot
land after a newer one and overwrite it.

**Answer text is rendered as React elements, never HTML.** It is model output;
`dangerouslySetInnerHTML` would be the obvious way to introduce an XSS hole.

**Never colour alone.** Each verdict has a distinct underline style *and* a word,
so the distinction survives colour blindness and greyscale. `supported` gets no
decoration at all — marking the normal case hides the exceptions.

**A link that will not resolve is not rendered as a link.** `isFollowable()`
gates that; a local file path is shown as text instead of a dead anchor.

**The map palette matches the server's SVG renderer**, deliberately, so a
committed `corpus-map.svg` and the live app never disagree.

## Checks

```bash
npm run typecheck
npm run lint
npm run build
```

All three run in CI.
