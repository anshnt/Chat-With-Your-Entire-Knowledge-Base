# Retrieval evaluation

Without measurement, every retrieval change is a vibe. This is the part of the
project that makes "hybrid beats dense" a number you can reproduce rather than a
claim you have to take on faith — including when the number says something
inconvenient.

```bash
kb eval generate -o eval/golden.yaml          # bootstrap a set from the corpus
kb eval run eval/golden.yaml --sweep full     # compare configurations
kb eval run eval/golden.yaml --report ./out   # markdown + JSON + SVG charts
```

## Golden sets

A golden set maps a question to the sources that answer it. Two practical
problems make that harder than it sounds.

**Chunk ids are not stable across re-ingestion.** They are generated per run, so
a golden set keyed on them is worthless the moment the chunker changes — which is
exactly when you most want to measure. A `GoldenQuery` can therefore specify its
expectation three ways, resolved against the live corpus at evaluation time:

| Field | Stability | Notes |
|---|---|---|
| `chunk_ids` | Brittle | Exact, fine within a single ingestion run |
| `document_ids` / `document_titles` | Survives re-chunking | Coarser: every chunk of the document counts |
| `must_contain` | Survives re-ingestion | A text snippet; any chunk containing it is relevant |

`must_contain` is the durable option and the default for generated sets. Matching
is whitespace-insensitive, because chunk text keeps its newlines while a snippet
taken from a sentence is space-joined — matching them literally silently drops
every snippet that crosses a line break.

**Relevance is not binary.** `2` means "directly answers", `1` means "related and
useful context". Binary relevance cannot express "this chunk is adjacent to the
answer", which is most of what a retriever actually gets wrong.

## Metrics

Implemented from their definitions rather than pulled from a library, because the
edge cases are where evaluation harnesses quietly lie.

| Metric | What it answers |
|---|---|
| `hit_rate@k` | Did *any* relevant chunk make the context? The floor for RAG — below this, generation quality is irrelevant |
| `recall@k` | What share of the relevant chunks made it? |
| `precision@k` | What share of the context is worth its tokens? |
| `ndcg@k` | Are the best chunks *first*, not merely present? |
| `mrr` | How far down is the first relevant hit? |
| `map` | Are *all* the relevant chunks early, not just one? |

Four decisions that keep the numbers honest:

- **A query with no relevant documents is excluded, not scored 0.** Scoring it
  zero drags the mean down by an amount that depends on how many unanswerable
  questions happen to be in the set, making runs incomparable. Excluded queries
  are counted and reported.
- **`recall@k` divides by `min(len(relevant), k)`.** Dividing by the raw count
  means a query with 20 relevant chunks can never exceed 0.4 at k=8, so the
  metric measures the golden set's shape rather than the retriever.
- **`precision@k` divides by `k`, not by the number of results returned.** A
  retriever returning 2 results, both relevant, has not achieved precision 1.0 at
  k=8 — it has failed to fill the context window.
- **`ndcg@k` normalises against the ideal *achievable at k*.** Otherwise the
  ceiling moves with the golden set.

## Sweeps

`--sweep` runs the same questions through the same corpus with one variable
changed, which is what makes a difference attributable rather than suggestive.

| Preset | Compares |
|---|---|
| `strategies` | lexical vs dense vs hybrid |
| `fusion` | RRF vs weighted vs max |
| `rerank` | fusion only vs reranked |
| `mmr` | diversification off vs λ=0.7 vs λ=0.5 |
| `full` | lexical, dense, hybrid, hybrid+rerank |

The report table shows deltas against the first configuration, and every run is
produced by the same `KnowledgeBase` that serves a live query — so the numbers
describe the code path that actually runs.

## Generating a golden set

Hand-writing 200 questions is the reason most projects have no evaluation.
`kb eval generate` bootstraps one from the corpus: pick a chunk, derive a
question it answers, record the chunk as the expected source. The label is
correct **by construction** — the question came from that chunk.

Care taken in the generator, because a bad question is worse than no question:

- Markdown inline formatting is stripped, or you get questions like *"what is
  (RRF)`?"*.
- Subjects spanning a clause boundary are rejected. "X scores pairs jointly **and
  is** more accurate" would otherwise yield *"what is X scores pairs jointly
  and?"*.
- Non-referential subjects are rejected: *"what is there?"*, *"what can it
  do?"* are unanswerable however good the retriever.
- Snippets are taken from the **middle** of a sentence. Openings are formulaic
  and repeat across a corpus, so a snippet from the start matches many chunks and
  silently inflates recall.
- Questions per document are **capped**, so one long file cannot dominate the set
  and turn the metrics into a measure of how well retrieval handles a single file.

### Synthetic questions are easier than real ones

They are phrased in the corpus's own vocabulary, so they systematically
over-reward lexical retrieval. This is not a small effect — see below. Use
generated sets to catch regressions and compare configurations; use
`eval/golden-paraphrase.yaml` (hand-written, deliberately paraphrased) or
`kb eval mine` (real logged queries) to judge absolute quality.

## What the numbers actually say here

Both sets below run against this repository's own documentation with the default
offline embedder, measured at the commit that added this file:

```bash
kb --data-dir /tmp/kbeval ingest ./docs ./README.md
kb --data-dir /tmp/kbeval eval generate -o /tmp/kbeval/golden.yaml --per-document 12
kb --data-dir /tmp/kbeval eval run /tmp/kbeval/golden.yaml --sweep full
kb --data-dir /tmp/kbeval eval run eval/golden-paraphrase.yaml --sweep full
```

(These figures will drift as the docs change — this file is itself part of the
corpus being measured. The shape of the result is the durable part, not the third
decimal place.)

**Generated set**, 15 of 19 queries scored — questions in the corpus's own words:

| configuration | hit_rate@5 | recall@5 | ndcg@5 | mrr | mean ms |
|---|---|---|---|---|---|
| lexical | **1.000** | **1.000** | 0.844 | **0.793** | 1.2 |
| dense | 0.600 | 0.600 | 0.485 | 0.447 | 2.7 |
| hybrid | 0.867 | 0.867 | 0.726 | 0.696 | 5.3 |
| hybrid + rerank | 0.933 | 0.933 | 0.818 | 0.787 | 10.2 |

**Paraphrase set**, 18 of 18 scored — questions worded differently from the
passages that answer them:

| configuration | hit_rate@5 | recall@5 | ndcg@5 | mrr | mean ms |
|---|---|---|---|---|---|
| lexical | 0.778 | 0.722 | 0.652 | **0.659** | 2.4 |
| dense | 0.500 | 0.444 | 0.292 | 0.291 | 3.1 |
| hybrid | **0.833** | **0.778** | **0.659** | 0.651 | 7.3 |
| hybrid + rerank | 0.778 | 0.778 | 0.636 | 0.588 | 15.4 |

Four things worth reading off that, none of them flattering by default.

**1. The generated set is not a quality measure — it is a vocabulary measure.**
BM25 scores a perfect 1.000 hit rate on questions derived from the corpus's own
sentences, and hybrid *loses* to it. On paraphrased questions the ordering
inverts: hybrid takes the hit rate (+0.055) and recall (+0.056). A project
reporting only synthetic numbers is reporting how well its retriever matches
strings it was handed.

**2. Hybrid earns its keep on recall, not on ranking.** On the paraphrase set it
gains 0.055 hit rate and 0.056 recall over BM25 while nDCG barely moves (+0.007)
and MRR slips (−0.008). That is exactly the expected shape: fusion surfaces
chunks BM25 alone never returns, and putting them in the *right order* is the
reranker's job, not fusion's.

**3. The offline reranker helps on one set and hurts on the other, for a
legible reason.** It gains 0.092 nDCG over plain hybrid on the generated set and
loses 0.023 on the paraphrase set. It scores query-term coverage, proximity and
exact phrase — precisely the signals that are *present* when a question reuses the
passage's wording and *absent* when it does not. The default is `lexical` because
it needs no API key; this measurement is the argument for switching to
`cross_encoder` in real use, and it is the kind of thing you cannot know without
measuring.

**4. Dense retrieval is weak because the default embedder is not semantic.**
`HashingEmbedder` is word and character n-gram hashing — it has no idea that
"furniture" relates to "headers" or that "trust score" means "faithfulness". Set
`KB_EMBEDDING_PROVIDER=voyage` and the dense and hybrid rows are the ones that
move; the lexical row will not.

Four queries in the generated run were **excluded rather than scored zero**: the
snippet they expected no longer appears intact in any single chunk, because
chunking split the sentence it came from. That is the failure mode the exclusion
rule exists for — scoring them zero would have shown a phantom 20% regression.

## Using it in CI

```bash
kb eval run eval/golden-paraphrase.yaml --metric ndcg@5 --fail-under 0.55
```

Exits non-zero when the headline metric falls below the threshold, so a retrieval
regression fails the build instead of being discovered in production.

## Reports

`--report DIR` writes three artefacts:

- `evaluation.md` — commits into the repo, so a retrieval change arrives in a PR
  with its numbers attached and the diff shows what moved.
- `evaluation.json` — machine-readable, for thresholds and trend tracking.
- `evaluation-metrics.svg`, `evaluation-ranks.svg` — hand-authored SVG (no
  plotting dependency, deterministic output) with colours that read on both light
  and dark backgrounds.

The rank histogram is the most diagnostic single artefact: a long tail at ranks
8-10 says the reranker is the problem, while a spike in the "not retrieved"
bucket says retrieval is.

Per query, the report also lists **where retrieval found nothing relevant** —
those are the cases where no amount of generation quality helps, and they are the
list of things to go and fix.
