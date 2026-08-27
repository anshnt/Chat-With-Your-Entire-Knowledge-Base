/**
 * Types mirroring the API's response models.
 *
 * Hand-written rather than generated, deliberately: the set of fields this UI
 * actually reads is much smaller than the API surface, and writing them out
 * makes the contract the frontend depends on explicit. Anything not listed here
 * is not relied on, so the API is free to change it.
 */

export type SourceType =
  | "pdf"
  | "markdown"
  | "text"
  | "html"
  | "web"
  | "github"
  | "youtube"
  | "notion";

export type SupportVerdict =
  | "supported"
  | "partial"
  | "unsupported"
  | "uncited"
  | "not_a_claim";

/** A source position. The `kind` discriminates which fields are meaningful. */
export interface Locator {
  kind: "pdf" | "text" | "web" | "github" | "youtube" | "notion";
  page?: number;
  page_count?: number;
  line_start?: number;
  line_end?: number;
  heading_path?: string[];
  file_path?: string;
  file_url?: string;
  url?: string;
  repo?: string;
  ref?: string;
  path?: string;
  symbol?: string;
  video_id?: string;
  start_seconds?: number;
  page_path?: string[];
  notion_page_id?: string;
}

export interface Citation {
  chunk_id: string;
  document_id: string;
  document_title: string;
  source_type: SourceType | null;
  label: string;
  deep_link: string | null;
  locator: Locator;
  snippet: string;
  heading_context: string;
}

export interface ScoreBreakdown {
  final: number;
  lexical: number | null;
  dense: number | null;
  lexical_rank: number | null;
  dense_rank: number | null;
  fusion: number | null;
  rerank: number | null;
  retrievers: string[];
}

export interface SearchHit {
  citation: Citation;
  scores: ScoreBreakdown;
  text: string;
}

export interface SearchResponse {
  query: string;
  hits: SearchHit[];
  strategy: "lexical" | "dense" | "hybrid";
  fusion: "rrf" | "weighted" | "max" | null;
  reranked: boolean;
  lexical_candidates: number;
  dense_candidates: number;
  fused_candidates: number;
  timings_ms: Record<string, number>;
  total_ms: number;
}

export interface AnswerCitation {
  marker: number;
  chunk_id: string;
  document_id: string;
  document_title: string;
  label: string;
  deep_link: string | null;
  snippet: string;
  source_type: SourceType | null;
  /** The structured position. Needed for decisions the URL cannot express —
   *  whether to embed in place or open externally. */
  locator: Locator;
  heading_context: string;
}

export interface AnswerSentence {
  text: string;
  citation_markers: number[];
  char_start: number;
  char_end: number;
  support_score: number | null;
  verdict: SupportVerdict | null;
  supporting_quote: string | null;
  verification_note: string | null;
}

export interface AskResponse {
  query: string;
  answer: string;
  citations: AnswerCitation[];
  sentences: AnswerSentence[];
  generator: string;
  model: string;
  refused: boolean;
  verified: boolean;
  faithfulness: number | null;
  unsupported_count: number;
  flagged_count: number;
  context_chunks: number;
  context_tokens: number;
  timings_ms: Record<string, number>;
  total_ms: number;
  retrieval: SearchResponse | null;
}

export interface DocumentRecord {
  id: string;
  collection: string;
  source_type: SourceType;
  title: string;
  uri: string;
  n_chunks: number;
  token_estimate: number;
  created_at: string;
}

export interface ChunkView {
  id: string;
  document_id: string;
  ordinal: number;
  text: string;
  kind: string;
  citation: Citation;
  token_estimate: number;
}

export interface CollectionStats {
  collection: string;
  n_documents: number;
  n_chunks: number;
  n_embedded: number;
  total_tokens: number;
  by_source_type: Record<string, number>;
  embedding_model: string | null;
  embedding_dim: number | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  embedding_model: string;
  embedding_dim: number;
  retrieval_strategy: string;
  reranker: string | null;
  generator: string;
  generation_model: string;
  verifier: string | null;
  connectors: string[];
}

export interface MapPoint {
  chunk_id: string;
  document_id: string;
  document_title: string;
  x: number;
  y: number;
  cluster: number;
  source_type: string | null;
  kind: string;
  snippet: string;
  heading: string;
  tokens: number;
  retrievals: number;
}

export interface MapCluster {
  id: number;
  label: string;
  terms: string[];
  size: number;
  coherence: number;
  retrieved_share: number;
}

export interface CorpusMapResponse {
  collection: string;
  method: string;
  n_chunks: number;
  n_plotted: number;
  sampled: boolean;
  explained_variance: number | null;
  retrieval_coverage: number;
  notes: string[];
  elapsed_ms: number;
  clusters: MapCluster[];
  points: MapPoint[];
}

/** A message in the chat transcript. */
export interface ChatTurn {
  id: string;
  question: string;
  /** Streamed text, before the final answer arrives. */
  streaming: string;
  answer: AskResponse | null;
  error: string | null;
  done: boolean;
}
