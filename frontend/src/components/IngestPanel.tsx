/**
 * Ingestion: drop a file, paste text, or give a path/URL.
 *
 * The one interesting decision is that **errors are shown per source, not as a
 * single failure**. The backend reports `IngestionReport.errors` alongside the
 * documents it *did* create, because one unreadable file in a directory should
 * not abort the rest — so the UI has to show a partial success as a partial
 * success rather than collapsing it to "failed".
 */

import { useCallback, useRef, useState } from "react";

import { api, ApiError } from "../lib/api";

interface Props {
  collection: string;
  onIngested: () => void;
}

type Result =
  | { kind: "idle" }
  | { kind: "busy"; what: string }
  | { kind: "done"; documents: number; chunks: number; errors: string[] }
  | { kind: "failed"; message: string };

export function IngestPanel({ collection, onIngested }: Props) {
  const [result, setResult] = useState<Result>({ kind: "idle" });
  const [source, setSource] = useState("");
  const [pasted, setPasted] = useState("");
  const [title, setTitle] = useState("");
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const run = useCallback(
    async (what: string, action: () => Promise<{ documents_created: number; chunks_created: number; errors: unknown[] }>) => {
      setResult({ kind: "busy", what });
      try {
        const response = await action();
        setResult({
          kind: "done",
          documents: response.documents_created,
          chunks: response.chunks_created,
          errors: response.errors.map(describeError),
        });
        if (response.documents_created > 0) onIngested();
      } catch (cause) {
        setResult({
          kind: "failed",
          message: cause instanceof ApiError ? cause.message : "Ingestion failed",
        });
      }
    },
    [onIngested],
  );

  const uploadFiles = useCallback(
    async (files: FileList | File[]) => {
      const list = Array.from(files);
      if (list.length === 0) return;
      setResult({ kind: "busy", what: `uploading ${list.length} file(s)` });
      let documents = 0;
      let chunks = 0;
      const errors: string[] = [];
      for (const file of list) {
        try {
          const response = await api.uploadFile(file, collection);
          documents += response.documents_created;
          chunks += response.chunks_created;
          errors.push(...response.errors.map(describeError));
        } catch (cause) {
          errors.push(
            `${file.name}: ${cause instanceof ApiError ? cause.message : "upload failed"}`,
          );
        }
      }
      setResult({ kind: "done", documents, chunks, errors });
      if (documents > 0) onIngested();
    },
    [collection, onIngested],
  );

  return (
    <div className="space-y-4">
      <div
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          void uploadFiles(event.dataTransfer.files);
        }}
        className={[
          "rounded-lg border-2 border-dashed p-6 text-center transition-colors",
          dragging
            ? "border-blue-500 bg-blue-500/5"
            : "border-slate-300 dark:border-slate-700",
        ].join(" ")}
      >
        <p className="text-sm text-slate-600 dark:text-slate-400">
          Drop PDFs, Markdown or text files here
        </p>
        <button
          type="button"
          onClick={() => fileInput.current?.click()}
          className="mt-2 rounded-md bg-slate-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-700 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-slate-300"
        >
          Choose files
        </button>
        <input
          ref={fileInput}
          type="file"
          multiple
          className="hidden"
          onChange={(event) => {
            if (event.target.files) void uploadFiles(event.target.files);
            event.target.value = "";
          }}
        />
        <p className="mt-2 text-xs text-slate-500">
          Uploaded files are kept, so PDF citations can link back to the page.
        </p>
      </div>

      <form
        className="space-y-2"
        onSubmit={(event) => {
          event.preventDefault();
          if (!source.trim()) return;
          void run(source, () => api.ingestSource(source.trim(), collection));
        }}
      >
        <label htmlFor="ingest-source" className="block text-xs font-medium text-slate-600 dark:text-slate-400">
          Or a path, directory, or URL
        </label>
        <div className="flex gap-2">
          <input
            id="ingest-source"
            value={source}
            onChange={(event) => setSource(event.target.value)}
            placeholder="./docs · anshnt/kb · https://example.com/docs"
            className="min-w-0 flex-1 rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-sm placeholder:text-slate-400 dark:border-slate-700 dark:bg-slate-900"
          />
          <button
            type="submit"
            disabled={!source.trim() || result.kind === "busy"}
            className="rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50 hover:bg-blue-500"
          >
            Ingest
          </button>
        </div>
      </form>

      <form
        className="space-y-2"
        onSubmit={(event) => {
          event.preventDefault();
          if (!pasted.trim()) return;
          void run("pasted text", () =>
            api.ingestText(pasted, title.trim() || "Pasted document", collection),
          );
          setPasted("");
          setTitle("");
        }}
      >
        <label htmlFor="ingest-text" className="block text-xs font-medium text-slate-600 dark:text-slate-400">
          Or paste text
        </label>
        <input
          value={title}
          onChange={(event) => setTitle(event.target.value)}
          placeholder="Title"
          className="w-full rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-sm placeholder:text-slate-400 dark:border-slate-700 dark:bg-slate-900"
        />
        <textarea
          id="ingest-text"
          value={pasted}
          onChange={(event) => setPasted(event.target.value)}
          rows={5}
          placeholder="Paste Markdown or plain text…"
          className="w-full rounded-md border border-slate-300 bg-white px-2.5 py-1.5 font-mono text-xs placeholder:text-slate-400 dark:border-slate-700 dark:bg-slate-900"
        />
        <button
          type="submit"
          disabled={!pasted.trim() || result.kind === "busy"}
          className="rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50 hover:bg-blue-500"
        >
          Add document
        </button>
      </form>

      <ResultView result={result} />
    </div>
  );
}

function ResultView({ result }: { result: Result }) {
  if (result.kind === "idle") return null;

  if (result.kind === "busy") {
    return <p className="text-xs text-slate-500">Ingesting {result.what}…</p>;
  }

  if (result.kind === "failed") {
    return (
      <p className="rounded-md border border-red-500/40 bg-red-500/5 p-2 text-xs text-red-700 dark:text-red-400">
        {result.message}
      </p>
    );
  }

  return (
    <div className="space-y-1 text-xs">
      {result.documents > 0 ? (
        <p className="text-emerald-700 dark:text-emerald-400">
          Added {result.documents} document{result.documents === 1 ? "" : "s"} (
          {result.chunks} chunks).
        </p>
      ) : (
        <p className="text-slate-500">
          No new documents — unchanged content is skipped rather than duplicated.
        </p>
      )}
      {/* Per-source errors, because one bad file does not abort the rest. */}
      {result.errors.length > 0 && (
        <ul className="space-y-0.5 text-amber-700 dark:text-amber-400">
          {result.errors.map((message, index) => (
            <li key={index}>{message}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function describeError(error: unknown): string {
  if (error && typeof error === "object") {
    const record = error as { source?: string; error?: string };
    const where = record.source ? `${record.source}: ` : "";
    return `${where}${record.error ?? "failed"}`;
  }
  return String(error);
}
