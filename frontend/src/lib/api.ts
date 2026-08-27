/**
 * API client.
 *
 * Two things here beyond `fetch`:
 *
 * - **Typed errors.** Every API failure carries a stable `code`, so the UI
 *   branches on that rather than on message text.
 * - **SSE parsing that survives chunk boundaries.** `fetch` streams arbitrary
 *   byte chunks, not events: an event can be split across two reads, so a naive
 *   `split("\n\n")` per chunk drops the tail of every answer. The reader keeps a
 *   buffer and only emits complete events.
 */

import type {
  AskResponse,
  ChunkView,
  CollectionStats,
  CorpusMapResponse,
  DocumentRecord,
  HealthResponse,
  SearchResponse,
} from "./types";

export class ApiError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly status: number,
    public readonly details: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch (cause) {
    // A network failure is the single most likely error in development: the API
    // is simply not running. Saying so beats "Failed to fetch".
    throw new ApiError(
      "network_error",
      "Could not reach the API. Is `kb serve` running on port 8000?",
      0,
      { cause: String(cause) },
    );
  }

  if (!response.ok) {
    let code = `http_${response.status}`;
    let message = response.statusText || "Request failed";
    let details: Record<string, unknown> = {};
    try {
      const body = await response.json();
      if (typeof body?.code === "string") code = body.code;
      if (typeof body?.message === "string") message = body.message;
      if (body?.details) details = body.details;
      // FastAPI validation errors use `detail`, not our envelope.
      else if (Array.isArray(body?.detail)) {
        message = body.detail.map((d: { msg?: string }) => d.msg).join("; ");
        code = "validation_error";
      }
    } catch {
      /* a non-JSON error body is not worth failing over */
    }
    throw new ApiError(code, message, response.status, details);
  }

  return (await response.json()) as T;
}

export interface AskOptions {
  collection?: string;
  topK?: number;
  strategy?: "lexical" | "dense" | "hybrid";
  rerank?: boolean;
  useMmr?: boolean;
  verify?: boolean;
  includeRetrieval?: boolean;
}

function askBody(query: string, options: AskOptions): string {
  return JSON.stringify({
    query,
    collection: options.collection ?? "default",
    top_k: options.topK,
    strategy: options.strategy,
    rerank: options.rerank,
    use_mmr: options.useMmr,
    verify: options.verify,
    include_retrieval: options.includeRetrieval ?? true,
  });
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),

  stats: (collection = "default") =>
    request<{ stats: CollectionStats; collections: string[] }>(
      `/api/collections/${encodeURIComponent(collection)}/stats`,
    ),

  documents: (collection = "default", limit = 100) =>
    request<{ documents: DocumentRecord[]; total: number }>(
      `/api/documents?collection=${encodeURIComponent(collection)}&limit=${limit}`,
    ),

  documentChunks: (documentId: string) =>
    request<{ document: DocumentRecord; chunks: ChunkView[] }>(
      `/api/documents/${encodeURIComponent(documentId)}/chunks`,
    ),

  chunkContext: (chunkId: string, window = 1) =>
    request<{ focus_chunk_id: string; chunks: ChunkView[] }>(
      `/api/chunks/${encodeURIComponent(chunkId)}/context?window=${window}`,
    ),

  search: (query: string, options: AskOptions = {}) =>
    request<SearchResponse>("/api/search", {
      method: "POST",
      body: askBody(query, options),
    }),

  ask: (query: string, options: AskOptions = {}) =>
    request<AskResponse>("/api/ask", {
      method: "POST",
      body: askBody(query, options),
    }),

  corpusMap: (collection = "default", method = "auto", maxPoints = 2000) =>
    request<CorpusMapResponse>(
      `/api/collections/${encodeURIComponent(collection)}/map` +
        `?method=${method}&max_points=${maxPoints}`,
    ),

  ingestText: (text: string, title: string, collection = "default") =>
    request<{ documents_created: number; chunks_created: number; errors: unknown[] }>(
      "/api/ingest",
      { method: "POST", body: JSON.stringify({ text, title, collection }) },
    ),

  ingestSource: (source: string, collection = "default", options: Record<string, unknown> = {}) =>
    request<{ documents_created: number; chunks_created: number; errors: unknown[] }>(
      "/api/ingest",
      { method: "POST", body: JSON.stringify({ source, collection, options }) },
    ),

  async uploadFile(file: File, collection = "default") {
    const form = new FormData();
    form.append("file", file);
    form.append("collection", collection);
    const response = await fetch("/api/ingest/upload", { method: "POST", body: form });
    if (!response.ok) {
      throw new ApiError("upload_failed", `Upload failed: ${response.statusText}`, response.status);
    }
    return (await response.json()) as {
      documents_created: number;
      chunks_created: number;
      errors: unknown[];
    };
  },
};

export type SseEvent =
  | { type: "delta"; text: string }
  | { type: "done"; answer: AskResponse }
  | { type: "error"; message: string };

/**
 * Stream an answer as Server-Sent Events.
 *
 * The buffering is the point. `fetch` hands back arbitrary byte chunks, and an
 * SSE event can straddle two of them — so parsing each chunk independently
 * silently truncates. Only complete `\n\n`-terminated events are emitted, and
 * anything left in the buffer is carried into the next read.
 */
export async function* streamAsk(
  query: string,
  options: AskOptions = {},
  signal?: AbortSignal,
): AsyncGenerator<SseEvent> {
  const response = await fetch("/api/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: askBody(query, options),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new ApiError(
      "stream_failed",
      `Could not open the answer stream (${response.status})`,
      response.status,
    );
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary = buffer.indexOf("\n\n");
      while (boundary !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const event = parseSseBlock(block);
        if (event) yield event;
        boundary = buffer.indexOf("\n\n");
      }
    }
    // A final event with no trailing blank line still has to be delivered.
    const tail = parseSseBlock(buffer);
    if (tail) yield tail;
  } finally {
    reader.releaseLock();
  }
}

function parseSseBlock(block: string): SseEvent | null {
  let event = "";
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!event || dataLines.length === 0) return null;

  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(dataLines.join("\n"));
  } catch {
    return null;
  }

  if (event === "delta" && typeof payload.text === "string") {
    return { type: "delta", text: payload.text };
  }
  if (event === "done" && payload.answer) {
    return { type: "done", answer: payload.answer as AskResponse };
  }
  if (event === "error") {
    return { type: "error", message: String(payload.message ?? "Generation failed") };
  }
  return null;
}
