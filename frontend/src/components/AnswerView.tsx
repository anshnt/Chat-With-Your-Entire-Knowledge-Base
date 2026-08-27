/**
 * The answer, with per-sentence verification and inline citations.
 *
 * The layout choice that matters: verification verdicts are rendered *on the
 * sentence*, not in a separate panel. A list of "3 unsupported claims" below an
 * answer is something a reader skips; an underline under the actual clause is
 * not. And `supported` sentences get no decoration at all — marking the normal
 * case draws the eye away from the exceptions, which are the only reason the
 * feature exists.
 */

import { useMemo } from "react";

import { CitationCard, CitationChip } from "./CitationChip";
import { renderInline, stripInline } from "../lib/inline";
import { segmentAnswer } from "../lib/locators";
import type { AnswerCitation, AnswerSentence, AskResponse } from "../lib/types";
import { faithfulnessTone, isFlagged, verdictStyle } from "../lib/verdicts";

interface Props {
  answer: AskResponse;
  activeChunkId: string | null;
  onSelectCitation: (citation: AnswerCitation) => void;
  showDebug: boolean;
}

export function AnswerView({ answer, activeChunkId, onSelectCitation, showDebug }: Props) {
  const flagged = useMemo(
    () => answer.sentences.filter((s) => isFlagged(s.verdict)),
    [answer.sentences],
  );

  return (
    <div className="space-y-4">
      <div className="text-[0.95rem] leading-7">
        {answer.sentences.length > 0 ? (
          answer.sentences.map((sentence, index) => (
            <SentenceView
              key={`${sentence.char_start}-${index}`}
              sentence={sentence}
              citations={answer.citations}
              activeChunkId={activeChunkId}
              onSelectCitation={onSelectCitation}
            />
          ))
        ) : (
          <p>{renderInline(answer.answer)}</p>
        )}
      </div>

      {answer.refused && (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/5 p-3 text-sm">
          <strong className="font-medium">Not in the knowledge base.</strong> The retrieved
          sources do not answer this. Try rephrasing, or ingest a source that covers it.
        </div>
      )}

      {flagged.length > 0 && <FlaggedClaims sentences={flagged} />}

      {answer.citations.length > 0 && (
        <section aria-label="Sources">
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Sources
          </h3>
          <div className="grid gap-2 sm:grid-cols-2">
            {answer.citations.map((citation) => (
              <CitationCard
                key={citation.marker}
                citation={citation}
                isActive={citation.chunk_id === activeChunkId}
                onSelect={onSelectCitation}
              />
            ))}
          </div>
        </section>
      )}

      <AnswerFooter answer={answer} />
      {showDebug && <RetrievalDebug answer={answer} />}
    </div>
  );
}

function SentenceView({
  sentence,
  citations,
  activeChunkId,
  onSelectCitation,
}: {
  sentence: AnswerSentence;
  citations: AnswerCitation[];
  activeChunkId: string | null;
  onSelectCitation: (citation: AnswerCitation) => void;
}) {
  const style = verdictStyle(sentence.verdict);
  const flagged = isFlagged(sentence.verdict);
  const segments = segmentAnswer(sentence.text);

  const title = flagged && style
    ? `${style.label}: ${sentence.verification_note ?? style.explanation}`
    : undefined;
  // Markdown markers are stripped for the accessible label: a screen reader
  // announcing "asterisk asterisk" is worse than no annotation.
  const plain = stripInline(sentence.text);

  return (
    <span
      className={style?.underline ? `${style.underline} ` : ""}
      title={title}
      // Only annotate what a reader needs told. Announcing "supported" on every
      // sentence would make the answer unreadable with a screen reader.
      {...(flagged && style ? { "aria-label": `${style.label}: ${plain}` } : {})}
    >
      {segments.map((segment, index) =>
        segment.kind === "text" ? (
          <span key={index}>{renderInline(segment.value)}</span>
        ) : (
          <CitationChip
            key={index}
            markers={segment.markers}
            citations={citations}
            activeChunkId={activeChunkId}
            onSelect={onSelectCitation}
          />
        ),
      )}{" "}
    </span>
  );
}

function FlaggedClaims({ sentences }: { sentences: AnswerSentence[] }) {
  return (
    <section
      aria-label="Claims to check"
      className="rounded-lg border border-slate-200 bg-slate-100/50 p-3 dark:border-slate-800 dark:bg-slate-900/50"
    >
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
        Check before trusting
      </h3>
      <ul className="space-y-2">
        {sentences.map((sentence, index) => {
          const style = verdictStyle(sentence.verdict);
          return (
            <li key={index} className="flex gap-2 text-xs">
              <span
                className={`mt-0.5 shrink-0 rounded px-1.5 py-0.5 font-medium ${style?.badge ?? ""}`}
              >
                {style?.label}
              </span>
              <div className="min-w-0">
                <p className="text-slate-700 dark:text-slate-300">
                  {renderInline(sentence.text)}
                </p>
                {sentence.verification_note && (
                  <p className="mt-0.5 text-slate-500">{sentence.verification_note}</p>
                )}
                {sentence.supporting_quote && (
                  <p className="mt-0.5 border-l-2 border-slate-300 pl-2 italic text-slate-500 dark:border-slate-700">
                    closest match: “{sentence.supporting_quote}”
                  </p>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function AnswerFooter({ answer }: { answer: AskResponse }) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500">
      <span>
        {answer.generator}
        {answer.model && answer.model !== answer.generator ? ` (${answer.model})` : ""}
      </span>
      <span aria-hidden>·</span>
      <span>
        {answer.context_chunks} chunks, ~{answer.context_tokens.toLocaleString()} tokens
      </span>
      <span aria-hidden>·</span>
      <span>{answer.total_ms.toFixed(0)} ms</span>
      {answer.faithfulness !== null && (
        <>
          <span aria-hidden>·</span>
          <span className={faithfulnessTone(answer.faithfulness)}>
            faithfulness {(answer.faithfulness * 100).toFixed(0)}%
            {answer.flagged_count > 0 && ` · ${answer.flagged_count} flagged`}
          </span>
        </>
      )}
      {answer.verified && answer.faithfulness === null && (
        <>
          <span aria-hidden>·</span>
          <span>nothing to verify</span>
        </>
      )}
    </div>
  );
}

/**
 * The retrieval that produced the answer.
 *
 * Exposed in the UI, not hidden behind a flag in a log, because "why did it say
 * that" is almost always a retrieval question — and the per-stage scores are the
 * only way to tell a generation problem from a retrieval one.
 */
function RetrievalDebug({ answer }: { answer: AskResponse }) {
  const retrieval = answer.retrieval;
  if (!retrieval) return null;

  return (
    <details className="rounded-lg border border-slate-200 text-xs dark:border-slate-800">
      <summary className="cursor-pointer select-none px-3 py-2 font-medium text-slate-600 dark:text-slate-400">
        Retrieval · {retrieval.strategy}
        {retrieval.fusion ? ` + ${retrieval.fusion}` : ""}
        {retrieval.reranked ? " + rerank" : ""} · {retrieval.hits.length} of{" "}
        {retrieval.fused_candidates} candidates · {retrieval.total_ms.toFixed(1)} ms
      </summary>
      <div className="border-t border-slate-200 px-3 py-2 dark:border-slate-800">
        <div className="mb-2 flex flex-wrap gap-3 text-slate-500">
          <span>lexical: {retrieval.lexical_candidates}</span>
          <span>dense: {retrieval.dense_candidates}</span>
          {Object.entries(retrieval.timings_ms).map(([stage, ms]) => (
            <span key={stage}>
              {stage.replace(/_ms$/, "")}: {ms.toFixed(1)} ms
            </span>
          ))}
        </div>
        <ol className="space-y-1">
          {retrieval.hits.map((hit, index) => (
            <li key={hit.citation.chunk_id} className="flex gap-2">
              <span className="w-4 shrink-0 text-right text-slate-400">{index + 1}</span>
              <div className="min-w-0">
                <div className="truncate">
                  {hit.citation.document_title}
                  {hit.citation.label && (
                    <span className="text-slate-500"> — {hit.citation.label}</span>
                  )}
                </div>
                <div className="font-mono text-[0.65rem] text-slate-500">
                  {formatScores(hit.scores)}
                </div>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </details>
  );
}

function formatScores(scores: AskResponse["retrieval"] extends null
  ? never
  : NonNullable<AskResponse["retrieval"]>["hits"][number]["scores"]): string {
  const parts = [`final=${scores.final.toFixed(4)}`];
  if (scores.lexical !== null) {
    parts.push(`bm25=${scores.lexical.toFixed(3)}@${scores.lexical_rank ?? "?"}`);
  }
  if (scores.dense !== null) {
    parts.push(`dense=${scores.dense.toFixed(3)}@${scores.dense_rank ?? "?"}`);
  }
  if (scores.fusion !== null) parts.push(`fused=${scores.fusion.toFixed(4)}`);
  if (scores.rerank !== null) parts.push(`rerank=${scores.rerank.toFixed(3)}`);
  return parts.join(" ");
}
