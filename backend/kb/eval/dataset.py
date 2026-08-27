"""Golden sets.

A golden set maps a question to the chunk ids (or documents) that answer it. Two
practical problems make this harder than it sounds, and both are handled here.

**Chunk ids are not stable across re-ingestion.** They are generated per
ingestion run, so a golden set keyed on them becomes worthless the moment the
chunker changes — which is exactly when you most want to measure. So a
:class:`GoldenQuery` may specify its expected sources three ways, and they are
resolved against the live corpus at evaluation time:

* ``chunk_ids`` — exact, brittle, fine within a single run
* ``document_ids`` / ``document_titles`` — stable across re-chunking, coarser
* ``must_contain`` — a text snippet; any chunk containing it is relevant. Stable
  across re-chunking *and* re-ingestion, and by far the most durable option.

**Relevance is not binary.** ``2`` means "directly answers", ``1`` means
"related and useful context". Binary relevance cannot express "this chunk is
adjacent to the answer", which is most of what a retriever actually gets wrong.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from kb.errors import EvaluationError
from kb.models import SourceType

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_match(text: str) -> str:
    """Collapse whitespace and lowercase, for snippet matching.

    Chunk text keeps its newlines; a snippet taken from a sentence is
    space-joined. Matching them literally silently fails whenever the snippet
    crosses a line break — which excluded 7 of 16 questions in the first real run
    and looked like a corpus problem rather than a bug.
    """
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


class GoldenQuery(BaseModel):
    """One evaluation question and the sources that should answer it."""

    id: str = ""
    query: str = Field(min_length=1)
    #: Exact chunk ids. Precise but invalidated by re-chunking.
    chunk_ids: list[str] = Field(default_factory=list)
    #: Documents whose chunks count as relevant. Survives re-chunking.
    document_ids: list[str] = Field(default_factory=list)
    document_titles: list[str] = Field(default_factory=list)
    #: Text a chunk must contain to be relevant. Survives re-ingestion entirely.
    must_contain: list[str] = Field(default_factory=list)
    #: Per-source grades, keyed by chunk id, document id/title, or snippet.
    grades: dict[str, int] = Field(default_factory=dict)
    #: Retrieval scope restrictions, so a query can test a filter path.
    source_types: list[SourceType] | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _require_an_expectation(self) -> GoldenQuery:
        if not (self.chunk_ids or self.document_ids or self.document_titles or self.must_contain):
            raise ValueError(
                f"golden query {self.query!r} specifies no expected sources; "
                "set chunk_ids, document_ids, document_titles or must_contain"
            )
        return self

    def grade_for(self, key: str) -> int:
        """Relevance grade for a key, defaulting to 2 ("directly answers")."""
        return self.grades.get(key, 2)


class GoldenSet(BaseModel):
    """A named collection of evaluation queries."""

    name: str = "golden"
    collection: str = "default"
    description: str = ""
    queries: list[GoldenQuery] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self) -> Iterator[GoldenQuery]:  # type: ignore[override]
        return iter(self.queries)

    @model_validator(mode="after")
    def _assign_ids(self) -> GoldenSet:
        """Give every query a stable id so results can be joined across runs."""
        for index, query in enumerate(self.queries, start=1):
            if not query.id:
                query.id = f"q{index:03d}"
        seen: set[str] = set()
        for query in self.queries:
            if query.id in seen:
                raise ValueError(f"duplicate golden query id: {query.id}")
            seen.add(query.id)
        return self

    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path: str | Path) -> GoldenSet:
        """Load from YAML, JSON, or JSONL, chosen by extension."""
        file_path = Path(path).expanduser()
        if not file_path.is_file():
            raise EvaluationError(f"golden set not found: {file_path}")
        text = file_path.read_text(encoding="utf-8")
        suffix = file_path.suffix.lower()

        if suffix == ".jsonl":
            queries = [json.loads(line) for line in text.splitlines() if line.strip()]
            return cls(name=file_path.stem, queries=queries)
        try:
            payload: Any = yaml.safe_load(text) if suffix in (".yaml", ".yml") else json.loads(text)
        except (yaml.YAMLError, json.JSONDecodeError) as exc:
            raise EvaluationError(f"could not parse {file_path.name}: {exc}") from exc

        if isinstance(payload, list):
            return cls(name=file_path.stem, queries=payload)
        if not isinstance(payload, dict):
            raise EvaluationError(f"{file_path.name} must contain a mapping or a list")
        payload.setdefault("name", file_path.stem)
        return cls.model_validate(payload)

    def save(self, path: str | Path) -> None:
        file_path = Path(path).expanduser()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json", exclude_defaults=False)
        if file_path.suffix.lower() in (".yaml", ".yml"):
            file_path.write_text(
                yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
        else:
            file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def tagged(self, tag: str) -> GoldenSet:
        """A sub-set carrying ``tag`` — for slicing results by question type."""
        return GoldenSet(
            name=f"{self.name}:{tag}",
            collection=self.collection,
            description=self.description,
            queries=[q for q in self.queries if tag in q.tags],
        )

    def all_tags(self) -> list[str]:
        return sorted({tag for query in self.queries for tag in query.tags})


class ResolvedQuery(BaseModel):
    """A golden query with its expectations resolved to live chunk ids."""

    query: GoldenQuery
    relevance: dict[str, int] = Field(default_factory=dict)
    unresolved: list[str] = Field(
        default_factory=list, description="Expectations that matched nothing in the corpus"
    )

    @property
    def is_usable(self) -> bool:
        return bool(self.relevance)


def resolve_golden_set(
    golden: GoldenSet, store: Any, *, collection: str | None = None
) -> tuple[list[ResolvedQuery], list[str]]:
    """Resolve every query's expectations against the live corpus.

    Returns the resolved queries and a list of warnings. Unresolvable
    expectations are reported loudly rather than silently producing a zero: a
    golden set that has drifted from the corpus would otherwise look like a
    retrieval regression, which is the most expensive kind of false alarm.
    """
    target = collection or golden.collection
    warnings: list[str] = []
    resolved: list[ResolvedQuery] = []

    documents = store.list_documents(target, limit=100_000)
    titles_to_ids: dict[str, list[str]] = {}
    for document in documents:
        titles_to_ids.setdefault(document.title.strip().lower(), []).append(document.id)
    known_document_ids = {d.id for d in documents}

    # Loaded once: snippet matching needs the chunk text, and re-reading per
    # query would make a 200-question set quadratic in corpus size.
    chunk_index = [
        (chunk.id, chunk.document_id, normalize_for_match(chunk.text))
        for batch in store.iter_chunks(target)
        for chunk in batch
    ]
    chunks_by_document: dict[str, list[str]] = {}
    for chunk_id, document_id, _ in chunk_index:
        chunks_by_document.setdefault(document_id, []).append(chunk_id)
    known_chunk_ids = {c[0] for c in chunk_index}

    for query in golden.queries:
        relevance: dict[str, int] = {}
        unresolved: list[str] = []

        for chunk_id in query.chunk_ids:
            if chunk_id in known_chunk_ids:
                relevance[chunk_id] = max(relevance.get(chunk_id, 0), query.grade_for(chunk_id))
            else:
                unresolved.append(f"chunk_id {chunk_id}")

        for document_id in query.document_ids:
            if document_id not in known_document_ids:
                unresolved.append(f"document_id {document_id}")
                continue
            grade = query.grade_for(document_id)
            for chunk_id in chunks_by_document.get(document_id, []):
                relevance[chunk_id] = max(relevance.get(chunk_id, 0), grade)

        for title in query.document_titles:
            matches = titles_to_ids.get(title.strip().lower(), [])
            if not matches:
                unresolved.append(f"document_title {title!r}")
                continue
            grade = query.grade_for(title)
            for document_id in matches:
                for chunk_id in chunks_by_document.get(document_id, []):
                    relevance[chunk_id] = max(relevance.get(chunk_id, 0), grade)

        for snippet in query.must_contain:
            needle = normalize_for_match(snippet)
            grade = query.grade_for(snippet)
            matched = False
            for chunk_id, _, text in chunk_index:
                if needle and needle in text:
                    relevance[chunk_id] = max(relevance.get(chunk_id, 0), grade)
                    matched = True
            if not matched:
                unresolved.append(f"must_contain {snippet!r}")

        if unresolved:
            warnings.append(
                f"{query.id} ({query.query[:60]!r}): unresolved {', '.join(unresolved)}"
            )
        if not relevance:
            warnings.append(
                f"{query.id}: no expectation resolved to a chunk — excluded from scoring"
            )

        resolved.append(ResolvedQuery(query=query, relevance=relevance, unresolved=unresolved))

    return resolved, warnings


def golden_set_from_pairs(
    pairs: Sequence[tuple[str, str]], *, name: str = "golden", collection: str = "default"
) -> GoldenSet:
    """Build a golden set from ``(question, snippet)`` pairs.

    The quickest way to a usable set by hand, and it uses the most durable
    expectation type.
    """
    return GoldenSet(
        name=name,
        collection=collection,
        queries=[GoldenQuery(query=q, must_contain=[snippet]) for q, snippet in pairs],
    )
