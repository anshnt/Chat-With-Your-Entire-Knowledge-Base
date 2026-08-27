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

## Answer generation

The `Generator` base class owns everything that determines whether a citation is
real; a provider implementation is only the part that produces text. That split
is deliberate — the shared half is where the correctness lives.

### Context packing

Chunks go into the prompt in rank order until the token budget is spent, and a
chunk is included **whole or not at all**. A truncated chunk yields a citation
pointing at text the model was never shown, which is worse than one fewer
source. The top-ranked chunk is always included, even if it alone exceeds the
budget, because returning nothing would be worse still.

Sources are labelled with a *marker* (`[1]`, `[2]`) rather than a chunk id. A
model shown `chk_9f2a1c…` will cheerfully produce one that looks just like it;
a small integer range is easy to constrain in the prompt and trivial to
validate on the way out.

### The three things that make a citation real

**Markers are validated against the sources actually supplied.** The model's
output is untrusted input. An invented `[9]` when six sources were given is
stripped from the text — not rendered as a dead chip — and logged, because a
model inventing source numbers is a signal about the prompt.

**Markers attach to the sentence they belong to.** Citations are written *after*
the claim: `…defaults to 60. [1]`. A sentence splitter that breaks on the period
puts `[1]` at the head of the *next* sentence, silently attributing the citation
to the wrong claim. Since verification is per-sentence, that would make every
verdict meaningless, so `_reattach_trailing_markers` moves a leading marker run
back onto the sentence it follows.

**Empty retrieval refuses.** With no context, the generator says so. Answering
from parametric knowledge at that point is the worst thing a grounded system can
do, because the output is indistinguishable from a grounded answer.

### The extractive generator

It does not write prose — it selects the sentences from the retrieved chunks that
best answer the question and cites each to its chunk. This is a design choice,
not a placeholder: an extractive answer is **trivially faithful**, since every
sentence is verbatim from a source. There is no mechanism by which it can
hallucinate.

That makes it the right default for a system about verified citations:

- it needs no key, network or model, so the demo, the tests and CI exercise the
  real end-to-end path including citation resolution;
- it is deterministic, so verification tests can assert exact verdicts;
- degrading to it when a provider is unavailable cannot introduce an unsupported
  claim, which is unusual for a fallback;
- it gives the evaluation harness a real floor to measure an LLM against.

Selection details that matter:

- **A relevance floor.** A sentence must cover at least 25% of the query's IDF
  mass. Without it, the chunk-rank prior alone is enough to produce a confident,
  fully-cited answer about reranking when asked the capital of France.
- **MMR over sentences.** Relevance minus redundancy against what is already
  selected. Without the redundancy term the answer becomes one fact restated from
  four overlapping chunks — the characteristic failure of naive extractive
  summarisation.
- **The heading prefix is stripped before quoting.** `Retrieval › Fusion` is
  valuable for retrieval and reads as a fragment in an answer.
- **Table rows and code fences are skipped.** They score well on term overlap and
  read as noise.

### Streaming

`Generator.stream` yields `(delta, None)` while generating, then `("", answer)`
once. Citations cannot be resolved until the text is complete — a marker may be
mid-emission — so the finished answer with its citations arrives as a separate
terminal event rather than being patched in as it goes. The SSE endpoint maps
that directly onto `delta` / `done` / `error` events.

## Citation verification

A citation nobody checked is decoration. The failure this stage exists to catch
is not a model inventing nonsense — it is a model *correctly completing a fact
that is not in the corpus* and attaching a citation to a chunk that does not say
it. The answer reads perfectly, the chip links to a real page, and the claim is
unsourced. No amount of retrieval quality prevents that; only checking does.

Verdicts are per **sentence**, because an answer is not uniformly true or false,
and "paragraph 2 is unsupported" is not actionable while "this clause is
unsupported" is. That granularity is also why the generation layer takes care to
attach citation markers to the sentence they follow — a misattributed marker
would make every verdict meaningless.

### Faithfulness

The share of *claim* sentences that come out supported. `not_a_claim` sentences
are excluded from the denominator: counting "Here is what the sources say:" as a
verified fact would inflate the score, and a metric you can raise by adding
filler is worthless. Claim classification is deliberately conservative —
over-classifying prose as a claim only makes the score stricter, while
under-classifying hides real failures.

A refusal scores `None`, not zero. Punishing the system for correctly declining
to answer would push it toward answering anyway, which is the opposite of what
this stage is for.

### The offline verifier

Textual entailment is the right frame, but a full NLI model is a large dependency
for a check that runs on every sentence of every answer. The lexical verifier
approximates the useful part and is explicit about its limits:

1. **IDF-weighted content coverage** of the claim against the chunk, with IDF
   computed over the chunk's own sentences — a word appearing in every sentence
   carries little evidence that *this* sentence is supported.
2. **Best-sentence alignment**, which becomes the `supporting_quote`.
3. **Number agreement**, the highest-value check in the file. The characteristic
   RAG failure is a fluent paraphrase with a wrong figure: "defaults to 50" cited
   to a chunk saying 60 scores near-perfectly on word overlap and is exactly the
   error a reader cannot spot. The check is *sentence-scoped*, not chunk-scoped,
   because a chunk-level check passes a claim of "50" against a chunk that says
   "defaults to 60" and, separately, "recall at 50" — a real case, found by
   writing a test fixture that accidentally contained the decoy.
4. **Negation agreement**. A claim and a chunk that disagree on negation share
   almost all their words, so without this a flat contradiction reads as strongly
   supported. Contrast is distinguished from negation: "combines ranks, **not**
   raw scores" and "combines ranks **rather than** raw scores" mean the same
   thing, and a keyword check calls them contradictory. Contrastive
   constructions are stripped first, then negation is detected only where it
   attaches to a verb (`does not support`) or is inherently negative (`never`,
   `cannot`, `without`, `fails to`).

**Gates, not scores.** A number contradiction or a negation flip caps the support
score below the unsupported boundary rather than merely subtracting from it. Some
signals are dispositive: the claim is wrong however well the rest of its words
line up, and that guarantee should not depend on the threshold a caller picks.

What it cannot catch: a claim that is a genuine semantic inference from the
chunk, and a paraphrase with no lexical overlap. Both push the score *down*, so
the failure mode is a false "unsupported" rather than a false "supported" — the
safe direction, and the reason falling back to this verifier when a hosted judge
is unavailable makes the check stricter rather than laxer.

### The LLM judge

The prompt asks a strict entailment question — *"does this text state this claim,
or can it be deduced directly from it?"* — not "is this a good citation?". A
model asked to judge quality rates plausible claims highly; consistency with the
source is exactly the failure being hunted.

Three prompt decisions carry most of the reliability:

- **A supporting quote is required.** Forcing the judge to point at a sentence
  suppresses "yes, because it seems right". If it cannot quote, it cannot claim
  support.
- **The default is no.** Uncertainty resolves to unsupported. A verifier that
  guesses "supported" is worse than none, because it launders the exact failure it
  was added to catch. An unparseable reply is likewise treated as unsupported.
- **Numbers are called out explicitly**, for the reason above.

Multiple markers on one sentence mean "any of these supports it", so support is
aggregated with `max` — not the mean, which would punish a correct citation for
being listed beside a weaker one. The judge loop short-circuits on a confident
yes, keeping the common case to one call.

## Corpus visualisation

A map of unlabelled dots is a screensaver. What makes one useful is answering
questions you cannot ask any other way: does this corpus cover what people
actually ask, are two documents saying the same thing, and is there a region
nobody's queries ever reach.

### Projection

Three methods, chosen by what is installed: **UMAP** (best structure, heavy
dependency), **t-SNE** (via scikit-learn), then a ~30-line numpy **PCA**. PCA is a
first-class fallback rather than an error because a map that exists is worth more
than a perfect map that does not — and it is honest about being linear: the
response reports `explained_variance`, and the CLI warns when it is low.

Two details that matter more than the method:

**Determinism.** A map whose points move on every reload cannot be compared
across runs. Every method is seeded, and the PCA sign convention is fixed — each
component's largest-magnitude loading is forced positive — because eigenvector
signs are otherwise arbitrary and would mirror the plot at random.

**Aspect-preserving normalisation.** Coordinates are scaled into `[0,1]²` by the
*larger* axis range, not per axis. Independent scaling would distort exactly the
distances the projection exists to show.

PCA uses SVD rather than an eigendecomposition of the covariance matrix: better
conditioned, and it avoids materialising a `dim × dim` matrix, which for
1536-dimensional embeddings is the expensive part.

### Clustering

k-means in numpy, with k-means++ seeding. The dependency-free path matters
because this is the feature most likely to be looked at first, on a fresh clone.

- **k-means++ seeding**, because random seeding routinely puts two centroids
  inside one dense topic and none in another, which no amount of iteration
  recovers from.
- **Empty clusters are reseeded** to the point furthest from its centroid. An
  empty cluster produces NaN centroids, which would poison the whole map.
- **Distances via the expanded form** `‖a−b‖² = ‖a‖² + ‖b‖² − 2a·b`, so the whole
  assignment step is one matrix multiply.
- **k defaults to `√(n/2)`**, clamped — better than a fixed default, which gives
  8 clusters for 12 chunks and 8 for 12,000.

### Automatic cluster labels

The obvious approach — the cluster's most frequent terms — produces *"the, and,
retrieval"* for every cluster. What works is a **contrast** statistic: terms
frequent *inside* the cluster and rare *outside* it. That is log-odds with an
informative Dirichlet prior (Monroe et al.), which is the standard tool for
exactly this question and is well-behaved on small clusters where a raw frequency
ratio explodes.

Two refinements on top:

- A term must appear in **at least two chunks** of the cluster. A term repeated
  many times in one chunk is that chunk's vocabulary, not the cluster's, and
  without this a single verbose passage names the whole region.
- Ties break toward terms spread across more of the cluster.

Each cluster also reports **coherence** — the mean similarity of its members to
the centroid. A low value means the cluster is a grab-bag and its label should not
be trusted, which is worth surfacing rather than hiding behind a confident-looking
name.

### Retrieval counts are the point

Every chunk carries how often it has been retrieved, from the `retrieval_events`
table. `coverage()` is the share of chunks retrieved at least once, and it is the
most useful number on the page: it says how much of the corpus is doing any work
at all. A cluster with 0% retrieval is either redundant or unreachable by how
people actually ask — both actionable, and neither visible without logging
retrievals.

In the SVG, never-retrieved points are drawn **hollow** rather than in a different
colour, so the distinction survives greyscale printing and colour blindness. Size
encodes retrieval count on a log scale, because one chunk retrieved 40 times would
otherwise dwarf everything and the interesting distinction is between zero, a few,
and many.

### Caching

A corpus map touches every vector and every chunk's text, so it is cached against
the store's write counter — the same invalidation signal the vector matrix uses.
It is recomputed exactly when the corpus changes and never otherwise (a cached
rebuild is ~0.015 ms against ~70 ms cold).

Above `max_points` chunks the map is **sampled with even spacing, not randomly**:
even spacing over the store's ordering keeps the sample stable between calls, so
the map does not reshuffle on reload, and spreads it across documents since chunks
are stored in ingestion order. The response always says how many were sampled.
