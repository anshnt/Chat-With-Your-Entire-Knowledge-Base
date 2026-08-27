/**
 * The application shell.
 *
 * Three panes on a wide screen: controls, the conversation, and the source. The
 * source pane is the point of the whole app — a citation has to land somewhere,
 * and if checking a claim means leaving the page, most readers will not.
 */

import { useCallback, useEffect, useState } from "react";

import { AnswerView } from "./components/AnswerView";
import { CorpusMap } from "./components/CorpusMap";
import { IngestPanel } from "./components/IngestPanel";
import { SourcePane } from "./components/SourcePane";
import { useAsk } from "./hooks/useAsk";
import { api, ApiError } from "./lib/api";
import type {
  AnswerCitation,
  CollectionStats,
  CorpusMapResponse,
  HealthResponse,
} from "./lib/types";

type Tab = "chat" | "corpus" | "map";

const EXAMPLES = [
  "how does reciprocal rank fusion combine rankings?",
  "why is a cross-encoder too slow to run over the whole corpus?",
  "what stops a wrong figure from being reported as supported?",
];

export function App() {
  const [tab, setTab] = useState<Tab>("chat");
  const [collection] = useState("default");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [stats, setStats] = useState<CollectionStats | null>(null);
  const [corpusMap, setCorpusMap] = useState<CorpusMapResponse | null>(null);
  const [mapError, setMapError] = useState<string | null>(null);
  const [activeCitation, setActiveCitation] = useState<AnswerCitation | null>(null);
  const [question, setQuestion] = useState("");
  const [showDebug, setShowDebug] = useState(true);
  const [strategy, setStrategy] = useState<"hybrid" | "lexical" | "dense">("hybrid");
  const [rerank, setRerank] = useState(true);

  const { turns, busy, ask, stop, clear } = useAsk({
    collection,
    strategy,
    rerank,
    verify: true,
    includeRetrieval: true,
  });

  const refresh = useCallback(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    api
      .stats(collection)
      .then((response) => setStats(response.stats))
      .catch(() => setStats(null));
  }, [collection]);

  useEffect(refresh, [refresh]);

  // The map is a whole-corpus computation, so it is loaded on demand rather than
  // eagerly — and only once per corpus change, since the server caches it too.
  useEffect(() => {
    if (tab !== "map" || corpusMap) return;
    setMapError(null);
    api
      .corpusMap(collection)
      .then(setCorpusMap)
      .catch((cause: unknown) =>
        setMapError(cause instanceof ApiError ? cause.message : "Could not load the map"),
      );
  }, [tab, corpusMap, collection]);

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    void ask(question);
    setQuestion("");
  };

  return (
    <div className="flex h-dvh flex-col">
      <Header health={health} stats={stats} tab={tab} onTab={setTab} />

      <div className="grid min-h-0 flex-1 lg:grid-cols-[18rem_1fr_22rem]">
        <nav
          aria-label="Controls"
          className="scrollbar-thin hidden overflow-y-auto border-r border-slate-200 p-4 lg:block dark:border-slate-800"
        >
          <Controls
            strategy={strategy}
            onStrategy={setStrategy}
            rerank={rerank}
            onRerank={setRerank}
            showDebug={showDebug}
            onShowDebug={setShowDebug}
            stats={stats}
            health={health}
            onClear={clear}
            hasTurns={turns.length > 0}
          />
          <hr className="my-4 border-slate-200 dark:border-slate-800" />
          <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Add sources
          </h2>
          <IngestPanel
            collection={collection}
            onIngested={() => {
              refresh();
              setCorpusMap(null);
            }}
          />
        </nav>

        <main className="scrollbar-thin min-h-0 overflow-y-auto">
          {tab === "chat" && (
            <div className="mx-auto max-w-3xl space-y-6 p-4 sm:p-6">
              {turns.length === 0 && <EmptyState onPick={(q) => void ask(q)} />}
              {turns.map((turn) => (
                <article key={turn.id} className="space-y-3">
                  <h2 className="text-sm font-semibold text-slate-500">{turn.question}</h2>
                  {turn.error ? (
                    <p className="rounded-lg border border-red-500/40 bg-red-500/5 p-3 text-sm text-red-700 dark:text-red-400">
                      {turn.error}
                    </p>
                  ) : turn.answer ? (
                    <AnswerView
                      answer={turn.answer}
                      activeChunkId={activeCitation?.chunk_id ?? null}
                      onSelectCitation={setActiveCitation}
                      showDebug={showDebug}
                    />
                  ) : (
                    /* Raw streamed text: citations and verdicts arrive with the
                       terminal event, because a half-emitted marker cannot be
                       rendered as a chip. */
                    <p className="whitespace-pre-wrap text-[0.95rem] leading-7 text-slate-700 dark:text-slate-300">
                      {turn.streaming || "…"}
                      <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-slate-400 align-middle" />
                    </p>
                  )}
                </article>
              ))}
            </div>
          )}

          {tab === "corpus" && <CorpusTab collection={collection} />}

          {tab === "map" && (
            <div className="p-4 sm:p-6">
              {mapError && (
                <p className="rounded-lg border border-red-500/40 bg-red-500/5 p-3 text-sm text-red-700 dark:text-red-400">
                  {mapError}
                </p>
              )}
              {!mapError && !corpusMap && (
                <p className="text-sm text-slate-500">Projecting the corpus…</p>
              )}
              {corpusMap && <CorpusMap map={corpusMap} />}
            </div>
          )}
        </main>

        <SourcePane citation={activeCitation} onClose={() => setActiveCitation(null)} />
      </div>

      {tab === "chat" && (
        <form
          onSubmit={submit}
          className="border-t border-slate-200 bg-white p-3 dark:border-slate-800 dark:bg-slate-900"
        >
          <div className="mx-auto flex max-w-3xl gap-2">
            <label htmlFor="question" className="sr-only">
              Your question
            </label>
            <input
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask your knowledge base…"
              autoComplete="off"
              className="min-w-0 flex-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm placeholder:text-slate-400 dark:border-slate-700 dark:bg-slate-950"
            />
            {busy ? (
              <button
                type="button"
                onClick={stop}
                className="rounded-md border border-slate-300 px-3 py-2 text-sm font-medium dark:border-slate-700"
              >
                Stop
              </button>
            ) : (
              <button
                type="submit"
                disabled={!question.trim()}
                className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50 hover:bg-blue-500"
              >
                Ask
              </button>
            )}
          </div>
        </form>
      )}
    </div>
  );
}

function Header({
  health,
  stats,
  tab,
  onTab,
}: {
  health: HealthResponse | null;
  stats: CollectionStats | null;
  tab: Tab;
  onTab: (tab: Tab) => void;
}) {
  const tabs: Array<[Tab, string]> = [
    ["chat", "Chat"],
    ["corpus", "Corpus"],
    ["map", "Map"],
  ];
  return (
    <header className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-slate-200 px-4 py-2.5 dark:border-slate-800">
      <h1 className="text-sm font-semibold">Chat With Your Entire Knowledge Base</h1>
      <nav className="flex gap-1" aria-label="Views">
        {tabs.map(([id, label]) => (
          <button
            key={id}
            type="button"
            onClick={() => onTab(id)}
            aria-current={tab === id ? "page" : undefined}
            className={[
              "rounded px-2.5 py-1 text-xs font-medium transition-colors",
              tab === id
                ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
                : "text-slate-600 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800",
            ].join(" ")}
          >
            {label}
          </button>
        ))}
      </nav>
      <div className="ml-auto flex flex-wrap items-center gap-x-3 text-xs text-slate-500">
        {stats && (
          <span>
            {stats.n_documents} docs · {stats.n_chunks} chunks
            {stats.n_chunks > 0 && stats.n_embedded < stats.n_chunks && (
              <span className="text-amber-600 dark:text-amber-400">
                {" "}
                · {stats.n_chunks - stats.n_embedded} unembedded
              </span>
            )}
          </span>
        )}
        {health ? (
          <span title={`embedder ${health.embedding_model}, dim ${health.embedding_dim}`}>
            {health.generator} · {health.retrieval_strategy}
            {health.reranker ? ` · ${health.reranker}` : ""}
          </span>
        ) : (
          <span className="text-red-600 dark:text-red-400">API unreachable</span>
        )}
      </div>
    </header>
  );
}

function Controls({
  strategy,
  onStrategy,
  rerank,
  onRerank,
  showDebug,
  onShowDebug,
  stats,
  health,
  onClear,
  hasTurns,
}: {
  strategy: "hybrid" | "lexical" | "dense";
  onStrategy: (value: "hybrid" | "lexical" | "dense") => void;
  rerank: boolean;
  onRerank: (value: boolean) => void;
  showDebug: boolean;
  onShowDebug: (value: boolean) => void;
  stats: CollectionStats | null;
  health: HealthResponse | null;
  onClear: () => void;
  hasTurns: boolean;
}) {
  return (
    <div className="space-y-3">
      <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
        Retrieval
      </h2>
      {/* The retrieval knobs are in the UI on purpose: asking the same question
          under lexical, dense and hybrid is how you develop an intuition for
          what fusion is actually doing. */}
      <div className="flex gap-1" role="group" aria-label="Retrieval strategy">
        {(["hybrid", "lexical", "dense"] as const).map((value) => (
          <button
            key={value}
            type="button"
            onClick={() => onStrategy(value)}
            aria-pressed={strategy === value}
            className={[
              "flex-1 rounded px-2 py-1 text-xs font-medium capitalize transition-colors",
              strategy === value
                ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
                : "border border-slate-300 hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800",
            ].join(" ")}
          >
            {value}
          </button>
        ))}
      </div>

      <Toggle label="Rerank" checked={rerank} onChange={onRerank} />
      <Toggle label="Show retrieval detail" checked={showDebug} onChange={onShowDebug} />

      {hasTurns && (
        <button
          type="button"
          onClick={onClear}
          className="w-full rounded border border-slate-300 px-2 py-1 text-xs font-medium hover:bg-slate-100 dark:border-slate-700 dark:hover:bg-slate-800"
        >
          Clear conversation
        </button>
      )}

      {stats && stats.n_chunks === 0 && (
        <p className="rounded border border-amber-500/40 bg-amber-500/5 p-2 text-xs">
          The corpus is empty. Add a source below to get started.
        </p>
      )}

      {health?.verifier && (
        <p className="text-xs text-slate-500">
          Citations are checked by <code>{health.verifier}</code>. Claims their source
          does not support are flagged in the answer.
        </p>
      )}
    </div>
  );
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-2 text-xs">
      <span>{label}</span>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="h-3.5 w-3.5 rounded border-slate-400"
      />
    </label>
  );
}

function EmptyState({ onPick }: { onPick: (question: string) => void }) {
  return (
    <div className="space-y-3 py-8 text-center">
      <p className="text-sm text-slate-500">
        Ask a question. Every claim comes back with a citation that jumps to the
        exact page, line or timestamp it came from.
      </p>
      <div className="flex flex-col items-center gap-1.5">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => onPick(example)}
            className="rounded-full border border-slate-300 px-3 py-1 text-xs text-slate-600 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-slate-800"
          >
            {example}
          </button>
        ))}
      </div>
    </div>
  );
}

function CorpusTab({ collection }: { collection: string }) {
  const [documents, setDocuments] = useState<
    Array<{ id: string; title: string; source_type: string; n_chunks: number }>
  >([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .documents(collection, 200)
      .then((response) => setDocuments(response.documents))
      .catch((cause: unknown) =>
        setError(cause instanceof ApiError ? cause.message : "Could not load documents"),
      );
  }, [collection]);

  if (error) {
    return (
      <p className="m-4 rounded-lg border border-red-500/40 bg-red-500/5 p-3 text-sm text-red-700 dark:text-red-400">
        {error}
      </p>
    );
  }

  return (
    <div className="p-4 sm:p-6">
      <table className="w-full text-sm">
        <thead className="text-left text-xs uppercase tracking-wide text-slate-500">
          <tr>
            <th className="pb-2 font-semibold">Title</th>
            <th className="pb-2 font-semibold">Source</th>
            <th className="pb-2 text-right font-semibold">Chunks</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
          {documents.map((document) => (
            <tr key={document.id}>
              <td className="max-w-0 truncate py-2 pr-4">{document.title}</td>
              <td className="py-2 pr-4 text-xs text-slate-500">{document.source_type}</td>
              <td className="py-2 text-right text-xs text-slate-500">{document.n_chunks}</td>
            </tr>
          ))}
          {documents.length === 0 && (
            <tr>
              <td colSpan={3} className="py-6 text-center text-sm text-slate-500">
                No documents yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
