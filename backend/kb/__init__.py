"""kb — chat with your entire knowledge base.

A retrieval-augmented generation system built around four ideas that separate it
from a chatbot wrapper:

1. **Hybrid retrieval** — BM25 (SQLite FTS5) fused with dense vectors via
   Reciprocal Rank Fusion, then diversified with MMR.
2. **Reranking** — a cross-encoder / listwise stage over the fused candidates.
3. **Citation verification** — every sentence of the answer is checked against
   the chunk it cites, and unsupported claims are flagged, not hidden.
4. **Retrieval evaluation** — Recall@k, nDCG@k, MRR and MAP over a golden set,
   so retrieval changes are measured instead of vibed.

Everything runs offline: the default embedder and generator are deterministic
local implementations, so the test suite and CI need no API keys.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
