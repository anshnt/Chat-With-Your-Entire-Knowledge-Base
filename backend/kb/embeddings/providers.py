"""Hosted and local embedding providers.

Each provider is imported lazily inside its constructor so that the package
imports cleanly with none of the optional extras installed — a fresh clone runs
the full test suite with only the base dependencies.

Retries use exponential backoff with jitter on the transient statuses
(429/5xx) and fail fast on everything else, because a 401 will not fix itself.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from kb.embeddings.base import Embedder
from kb.errors import MissingDependencyError, ProviderError

_MAX_ATTEMPTS = 4
_RETRY_BASE_DELAY = 0.5


def _with_retries(fn: Any, *, what: str) -> Any:
    """Call ``fn`` with exponential backoff on transient provider failures."""
    last: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if not _is_transient(exc) or attempt == _MAX_ATTEMPTS - 1:
                break
            delay = _RETRY_BASE_DELAY * (2**attempt) * (0.5 + random.random())
            time.sleep(delay)
    raise ProviderError(f"{what} failed: {last}") from last


def _is_transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        return status == 429 or 500 <= status < 600
    text = str(exc).lower()
    return any(
        s in text for s in ("timeout", "timed out", "connection", "429", "503", "overloaded")
    )


class VoyageEmbedder(Embedder):
    """Voyage AI embeddings. Asymmetric: distinct ``input_type`` per side."""

    def __init__(self, model: str = "voyage-3", api_key: str = "", batch_size: int = 64) -> None:
        try:
            import voyageai  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MissingDependencyError("voyageai", "embeddings") from exc
        key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        if not key:
            raise ProviderError("VOYAGE_API_KEY is not set")
        self._client = voyageai.Client(api_key=key)
        self.model = model
        self.batch_size = batch_size
        self.dim = _probe_dim(lambda: self._call(["dimension probe"], "document"))

    def _call(self, texts: Sequence[str], input_type: str) -> np.ndarray:
        result = _with_retries(
            lambda: self._client.embed(list(texts), model=self.model, input_type=input_type),
            what="voyage embed",
        )
        return np.asarray(result.embeddings, dtype="float32")

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return _batched_embed(texts, self.batch_size, lambda b: self._call(b, "document"), self.dim)

    def embed_query(self, text: str) -> np.ndarray:
        return self._call([text], "query")[0]


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings. Symmetric, so queries and documents share a path."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        batch_size: int = 64,
        dimensions: int | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MissingDependencyError("openai", "embeddings") from exc
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ProviderError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=key)
        self.model = model
        self.batch_size = batch_size
        self._dimensions = dimensions
        self.dim = dimensions or _probe_dim(lambda: self._call(["dimension probe"]))

    def _call(self, texts: Sequence[str]) -> np.ndarray:
        kwargs: dict[str, Any] = {"model": self.model, "input": list(texts)}
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions
        result = _with_retries(
            lambda: self._client.embeddings.create(**kwargs), what="openai embed"
        )
        return np.asarray([d.embedding for d in result.data], dtype="float32")

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return _batched_embed(texts, self.batch_size, self._call, self.dim)


class LocalEmbedder(Embedder):
    """sentence-transformers, running locally on CPU or GPU."""

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MissingDependencyError("sentence-transformers", "local") from exc
        self._model = SentenceTransformer(model, device=device)
        self.model = model
        self.batch_size = batch_size
        self.dim = int(self._model.get_sentence_embedding_dimension())
        # E5/BGE-family models expect these prefixes; passing them is what makes
        # the asymmetric variants perform as advertised.
        self.query_prefix = query_prefix or _default_query_prefix(model)
        self.document_prefix = document_prefix or _default_document_prefix(model)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        payload = [f"{self.document_prefix}{t}" for t in texts]
        vectors = self._model.encode(
            payload,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        vector = self._model.encode(
            f"{self.query_prefix}{text}",
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vector, dtype="float32")


def _default_query_prefix(model: str) -> str:
    lowered = model.lower()
    if "e5" in lowered:
        return "query: "
    if "bge" in lowered and "en" in lowered:
        return "Represent this sentence for searching relevant passages: "
    return ""


def _default_document_prefix(model: str) -> str:
    return "passage: " if "e5" in model.lower() else ""


def _batched_embed(texts: Sequence[str], batch_size: int, call: Any, dim: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, dim), dtype="float32")
    parts = [call(list(texts[i : i + batch_size])) for i in range(0, len(texts), batch_size)]
    return np.vstack(parts).astype("float32")


def _probe_dim(call: Any) -> int:
    """Discover a provider's output width with one throwaway call."""
    vectors = call()
    return int(np.asarray(vectors).shape[-1])
