"""Shared fixtures.

Every fixture is offline and deterministic: the hashing embedder produces the
same vectors on every machine, so retrieval assertions can be exact rather than
approximate. That is the whole reason the default provider is what it is.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from kb.config import Settings
from kb.knowledge_base import KnowledgeBase
from kb.store import SQLiteStore

CORPUS: dict[str, str] = {
    "retrieval.md": """# Retrieval

## Hybrid search

Hybrid search combines BM25 lexical matching with dense vector retrieval.
Lexical matching finds exact identifiers and rare terms; dense retrieval finds
paraphrase. Neither alone is sufficient.

## Reciprocal rank fusion

Reciprocal Rank Fusion combines ranked lists using ranks rather than scores.
The damping constant k defaults to 60, which means rank one and rank two differ
by roughly one and a half percent.

## Maximal marginal relevance

MMR trades relevance for diversity so the top results are not near-duplicates
of each other.
""",
    "evaluation.md": """# Evaluation

## Metrics

Recall at k measures how many relevant chunks reached the candidate set.
Normalised discounted cumulative gain rewards ranking the best chunk first.
Mean reciprocal rank considers only the first relevant result.

## Golden sets

A golden set maps each question to the chunk identifiers that answer it.
""",
    "storage.txt": """The store keeps documents, chunks, the full text index and the dense
vectors in a single SQLite file. Vectors are little-endian float32 blobs.
Cosine similarity is a dot product over normalised rows.
""",
}


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """Settings pointed at a throwaway data directory."""
    return Settings(
        data_dir=tmp_path / "data",
        embedding_dim=256,
        chunk_size=600,
        chunk_overlap=80,
        min_chunk_size=60,
        rerank_enabled=False,
    )


@pytest.fixture
def store(tmp_settings: Settings) -> Iterator[SQLiteStore]:
    instance = SQLiteStore(tmp_settings.db_path)
    yield instance
    instance.close()


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """A small on-disk corpus of Markdown and text files."""
    directory = tmp_path / "corpus"
    directory.mkdir()
    for name, body in CORPUS.items():
        (directory / name).write_text(body, encoding="utf-8")
    return directory


@pytest.fixture
def empty_kb(tmp_settings: Settings) -> Iterator[KnowledgeBase]:
    instance = KnowledgeBase(tmp_settings)
    yield instance
    instance.close()


@pytest.fixture
def kb(empty_kb: KnowledgeBase, corpus_dir: Path) -> KnowledgeBase:
    """A knowledge base with the sample corpus ingested and embedded."""
    report = empty_kb.ingest(str(corpus_dir))
    assert not report.errors, report.errors
    assert report.chunks_created > 0
    return empty_kb
