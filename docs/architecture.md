# Architecture

## The shape of the system

```
sources ──→ connectors ──→ segments ──→ chunkers ──→ chunks + locators
                                                          │
                                                          ├──→ FTS5 index (BM25)
                                                          └──→ float32 vectors
                                                          
query ──┬──→ BM25         ──┐
        └──→ dense cosine ──┴──→ fuse ──→ rerank ──→ MMR ──→ top-k ──→ answer
```

Five layers, each with one job:

| Layer | Module | Responsibility |
|---|---|---|
| Connectors | `kb/ingest/` | Turn a source into text plus the means to address positions in it |
| Chunking | `kb/chunking/` | Split text on natural boundaries, tracking position |
| Storage | `kb/store/` | One SQLite file: documents, chunks, BM25 index, vectors |
| Retrieval | `kb/retrieval/` | Lexical + dense candidates, fusion, diversification |
| Interfaces | `kb/api/`, `kb/cli.py` | HTTP and terminal, both over the same façade |

Everything above the storage layer goes through `KnowledgeBase`
(`kb/knowledge_base.py`). That is deliberate: the evaluation harness measures the
same code path that serves a real query, which is the only way a retrieval
benchmark means anything.

## Locators: why citations actually land somewhere

A citation is only useful if the reader can get to the exact place the claim came
from. Every source type needs a different kind of address for that, so `Locator`
is a discriminated union and each variant knows how to build its own link:

| Variant | Address | `deep_link()` |
|---|---|---|
| `PdfLocator` | page, char span | `report.pdf#page=12` |
| `TextLocator` | line range, heading path | `notes.md#L88-L104` |
| `WebLocator` | URL, quote | `example.com/p#:~:text=RRF%20consumes,not%20scores` |
| `GitHubLocator` | repo, ref, path, lines | `github.com/o/r/blob/main/f.py#L10-L20` |
| `YouTubeLocator` | video id, start seconds | `youtube.com/watch?v=ID&t=93s` |
| `NotionLocator` | page path, line range | page URL |

The web variant uses the [Text Fragments][text-fragments] spec, so the browser
scrolls to and highlights the quote on a page we do not control.

The payoff is in what *doesn't* need to change. Chunking, retrieval, fusion,
reranking and generation never learn what a page or a timestamp is — they only
ever call `locator.deep_link()`. Adding a source type is a new `Locator` variant
plus a connector, and nothing else.

[text-fragments]: https://wicg.github.io/scroll-to-text-fragment/

## Connectors and segments

A connector produces `ParsedDocument`s made of `Segment`s. A segment is a run of
text plus a `build_locator` callback:

```python
@dataclass
class Segment:
    text: str
    build_locator: Callable[[ChunkDraft], Locator]
```

This is the seam that lets one chunker serve every source. A PDF yields one
segment per page whose callback stamps that page number; a YouTube transcript
yields one segment per time window whose callback stamps that start time.
Chunking happens *inside* a segment, so a chunk can never straddle a page
boundary and therefore never has an ambiguous address.

## Chunking

Two chunkers, chosen by source:

**`RecursiveChunker`** tries the largest natural boundary that fits — sections,
paragraphs, sentences, clauses, words — and only hard-cuts on pathological input.
A chunk that ends mid-sentence embeds worse *and* reads badly as a citation, so
the boundary quality is not cosmetic.

**`MarkdownChunker`** additionally:

- keeps the full ancestor heading path per chunk, so a citation reads
  `Architecture › Retrieval › Fusion` instead of `line 412`;
- never splits inside a fenced code block, and marks such chunks `CODE`;
- keeps tables whole and marks them `TABLE`;
- **prefixes each chunk with its heading path.** A chunk that reads "It defaults
  to 60." is unretrievable on its own. Under `Retrieval › Fusion › RRF` it is
  not. This measurably helps both BM25 and dense retrieval.

Overlap is taken as whole trailing *sentences* rather than a fixed character
count, so the repeated span always reads as prose, and is skipped inside code
fences where it would produce unbalanced backticks.

## Storage

One SQLite file holds everything:

| Table | Contents |
|---|---|
| `documents` | One row per ingested source; unique on `(collection, content_hash)` |
| `chunks` | Retrievable units, with the serialised locator |
| `chunks_fts` | FTS5 external-content index over chunks → BM25 |
| `embeddings` | `float32` little-endian blobs, plus a stored norm |
| `retrieval_events` | Every retrieval, for the heatmap and for mining eval queries |

Keeping the lexical and dense views in one transactional store means they cannot
drift apart — there is no second system to reconcile after a failed write. FTS5
triggers keep the index in sync with `chunks` automatically, including on delete.

**Why an exact vector scan.** 50k chunks × 1024 dims is ~200 MB, and a full
normalised matrix multiply lands in single-digit milliseconds under numpy's BLAS.
At this project's scale that is not a compromise — it is faster than an ANN index
would be after paying build time, and it has no recall loss and no tuning
surface. The assembled matrix is cached and invalidated by a write counter. The
seam for swapping in an approximate index is `DenseRetriever.search`, and nothing
above it would change.

**Filtering happens before top-k selection**, not after. Post-filtering a
fixed-size result set silently loses recall exactly when the filter is
selective — which is the case that matters.

## Retrieval

### Why hybrid

Dense retrieval fails on exact identifiers, error codes, rare proper nouns and
anything the embedding model never saw. BM25 fails on paraphrase. The failures
are not correlated, which is what makes combining them worth more than tuning
either.

The important detail is the *order*: two candidate lists of `candidate_k` (50 by
default) are fused, and only then truncated to `top_k` (8). The chunk BM25 ranks
40th and the vectors rank 35th is often the right answer, and it is invisible to
either retriever alone at k=8.

### Lexical

BM25 via FTS5, with two non-obvious pieces of work:

**Query sanitisation.** FTS5's MATCH grammar treats `"`, `*`, `:`, `^`, `AND/OR/NOT`
and parentheses as operators, so a raw question like `What is "hybrid search"? (RRF)`
is a syntax error rather than a query. Terms are extracted and re-quoted rather
than escaped in place.

**Recall under an exact-match engine.** A strict AND over every term returns
nothing for most natural-language questions; a plain OR ranks a stopword match
alongside a full match. The compromise: OR over content terms, prefix variants
for the longer ones (so `retriev` reaches `retrieving` beyond what the porter
stemmer catches), and a small stoplist — small on purpose, because an aggressive
one strips meaning from short queries like "who is on call".

### Fusion

**Reciprocal Rank Fusion is the default because it consumes ranks, not scores.**
BM25 scores are unbounded and corpus-dependent; cosine scores live in [-1, 1].
Any weighted sum of the two needs per-query normalisation, and min-max
normalisation over a *truncated* candidate list is unstable — one outlier at rank
1 compresses everything below it.

```
RRF(d) = Σᵢ wᵢ / (k + rankᵢ(d))
```

`k = 60` means rank 1 and rank 2 differ by ~1.6%, so one retriever's
confident-but-wrong top hit cannot outvote a chunk both retrievers agree on.

`weighted` and `max` fusion are also implemented, because they are the right
answer when the score scales *are* comparable — and because having all three
lets the evaluation harness demonstrate the difference instead of asserting it.

**Provider-aware weight defaults.** The offline hashing embedder is a
lexical-overlap model, not a semantic one, so leaving dense weighted above BM25
lets the weaker signal win. When the provider is `hashing` and the user has not
set the weights, they flip to favour BM25 (0.65 / 0.35). Set either weight
explicitly and that inference is skipped.

### MMR

Top-k by relevance alone has a specific failure mode that hurts RAG badly: the k
best chunks are often k near-copies. Overlapping chunks, a doc and its changelog,
the same paragraph in two exports — all rank together, and the generator ends up
with one fact repeated k times instead of k facts.

```
MMR = argmax_{d ∉ S} [ λ · rel(d, q) − (1 − λ) · max_{s ∈ S} sim(d, s) ]
```

λ ≈ 0.7 is a good default for question answering. Similarity uses stored vectors
when available and falls back to token Jaccard, so diversification still works
on a collection that has not been embedded.

## Score provenance

`ScoredChunk` keeps every stage's score rather than collapsing to one number:

```
final=0.0328 bm25=6.9412@2 dense=0.7431@1 fused=0.0328 rerank=0.91
```

This is not a debug convenience — it is the difference between a ranking you can
explain and one you have to trust. The CLI prints it, the API returns it, and the
evaluation harness reports on it.

## Offline by default

The default embedder (`HashingEmbedder`) and generator are deterministic local
implementations. This is a design decision, not a placeholder:

- `git clone && pytest` works with no keys, no network, no model download;
- CI needs no secrets, so it runs on forks and pull requests;
- retrieval assertions can be **exact** rather than approximate, because the same
  text produces the same vector on every machine.

`HashingEmbedder` is signed feature hashing over word unigrams, word bigrams and
character 4-grams, with sublinear term-frequency damping. It has no idea that
"car" relates to "automobile" and does not pretend to. What it is: fast,
dependency-free, reproducible, and strong enough that the full pipeline — fusion,
MMR, reranking, evaluation — can be exercised end to end. Set
`KB_EMBEDDING_PROVIDER=voyage` for real semantics; no other code changes.

## Error handling

`KBError` subclasses carry a stable `code` and an `http_status`, so the HTTP layer
maps failures without string-matching messages and clients branch on codes.

Ingestion degrades rather than aborting: one unreadable file in a directory is
reported in `IngestionReport.errors` while the rest of the directory succeeds, and
a document whose embedding step fails is still **searchable over BM25** — `kb embed`
resumes the job later. That resumability matters when embedding a large corpus
against a rate-limited API.

## Reranking

Fusion optimises recall at `candidate_k`; the generator only ever sees `top_k`.
Reranking is the stage that converts one into the other. It is the highest-
leverage single addition to a naive pipeline, for a structural reason: a
cross-encoder concatenates the query and the passage into one sequence and runs
attention across both, so it can condition on the query *while reading* the
passage. Comparing two independently-computed embeddings cannot express that, no
matter how good the embeddings are. The cost is O(candidates) model calls, which
is exactly why the stage sits after fusion — tens of candidates, not tens of
thousands of chunks.

All four providers implement `Reranker.score(query, candidates) -> list[float]`
and inherit the reordering, provenance and failure handling from the base class:

| Provider | Kind | Notes |
|---|---|---|
| `lexical` *(default)* | Offline cross-features | No keys, no network, deterministic |
| `cross_encoder` | Local model | `ms-marco-MiniLM-L-6-v2`, ~90 MB, CPU |
| `cohere` / `voyage` | Hosted API | Best quality, worst latency |
| `llm` | Listwise | Sees all candidates at once, orders them |

### The offline reranker is not a stub

It computes the query-passage interaction features a first-stage retriever
structurally cannot, because BM25 scores terms independently and a bi-encoder
never sees the pair:

1. **IDF-weighted term coverage** — how much of the query's *information* the
   passage covers, not how many words. IDF is computed over the candidate set
   itself, so it adapts per query with no corpus statistics to maintain.
2. **Proximity** — the width of the narrowest window containing all matched
   terms, via a k-way merge. `fusion … 400 words … ranks` and `fusion of ranks`
   are identical under BM25 and very different here.
3. **Exact phrase match** — the longest contiguous run of query terms. The
   clearest signal a bag-of-words model throws away.
4. **First-match position** — passages that answer immediately beat passages that
   mention the topic near the end.
5. **Heading match** — a hit in the heading path means the *section* is about the
   query, not just one sentence.

Weights are chosen for interpretability, not fitted. The point is a strong,
explainable baseline that `kb eval` can measure a hosted cross-encoder
*against* — so "is the upgrade worth its latency on this corpus" is a
measurement rather than an assumption.

### Failure is not an option the pipeline exposes

Every failure mode degrades to the fused order and keeps every candidate:

- an exception in `score` → fused order, warning logged;
- a wrong-length score list → fused order, warning logged;
- a hosted provider returning only its top *n* → the omitted candidates are
  ranked below the returned ones rather than dropped, preserving the recall
  fusion worked for;
- a listwise LLM returning duplicates, hallucinated indices, omissions or prose →
  the parser yields a complete permutation regardless;
- a missing optional dependency or API key → the offline reranker, with a
  warning.

Ties break on `fusion_score`, so a reranker that cannot separate two passages
leaves the retrieval order intact instead of shuffling it.

### Score quality, not just ordering

RRF scores are compressed by construction — `k = 60` means rank 1 and rank 2
differ by ~1.6% — which makes a `min_score` threshold useless. Rerank scores
separate:

```
fusion only:   0.0164  0.0161  0.0159
+ reranking:   1.7448  0.8685  0.6778
```

Both stages' scores are kept on every result (`fusion_score` and `rerank_score`),
so the ranking stays explainable after reranking rather than becoming a black box.
