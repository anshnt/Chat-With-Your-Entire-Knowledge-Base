"""Settings.

Everything is overridable by environment variable with the ``KB_`` prefix, e.g.
``KB_EMBEDDING_PROVIDER=voyage``. The defaults are deliberately chosen so that a
fresh clone works with no API keys and no network: the ``hashing`` embedder and
``extractive`` generator are deterministic local implementations. Point the
provider settings at a real service when you want real quality.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from kb.models import FusionMethod, RetrievalStrategy


class EmbeddingProvider(str, Enum):
    HASHING = "hashing"
    """Deterministic local bag-of-ngrams embedder. No network, no keys. The
    default so that tests and CI are hermetic."""

    VOYAGE = "voyage"
    OPENAI = "openai"
    LOCAL = "local"
    """sentence-transformers, runs on CPU/GPU locally."""


class RerankProvider(str, Enum):
    NONE = "none"
    LEXICAL = "lexical"
    """Offline reranker: query-term coverage + proximity + position priors."""

    CROSS_ENCODER = "cross_encoder"
    COHERE = "cohere"
    VOYAGE = "voyage"
    LLM = "llm"
    """Listwise reranking by an LLM."""


class VerifyProvider(str, Enum):
    NONE = "none"
    LEXICAL = "lexical"
    """Offline entailment approximation: coverage, alignment, number/entity and
    negation agreement. No keys, deterministic."""

    LLM = "llm"
    """A language model acting as a strict entailment judge."""


class GenerationProvider(str, Enum):
    EXTRACTIVE = "extractive"
    """Offline generator: selects and stitches the most query-relevant sentences
    from retrieved chunks, with real citations. Always available."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class ChunkStrategy(str, Enum):
    RECURSIVE = "recursive"
    """Split on the largest natural boundary that fits: sections, paragraphs,
    sentences, then characters."""

    SENTENCE = "sentence"
    FIXED = "fixed"
    SEMANTIC = "semantic"
    """Group adjacent sentences by embedding similarity, break at drops."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # ---------------- storage ----------------
    data_dir: Path = Field(default=Path("./data"))
    db_filename: str = "kb.db"

    # ---------------- chunking ----------------
    chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = Field(default=1200, ge=100, le=8000, description="Target chars per chunk")
    chunk_overlap: int = Field(default=180, ge=0, le=2000)
    min_chunk_size: int = Field(default=120, ge=0, description="Chunks below this are merged")
    code_chunk_size: int = Field(default=1600, ge=200, le=8000)
    transcript_chunk_seconds: float = Field(default=90.0, gt=0)

    # ---------------- embeddings ----------------
    embedding_provider: EmbeddingProvider = EmbeddingProvider.HASHING
    embedding_model: str = Field(default="", description="Provider-specific model name")
    embedding_dim: int = Field(default=512, ge=16, le=8192)
    embedding_batch_size: int = Field(default=64, ge=1, le=512)
    embedding_cache: bool = True

    # ---------------- retrieval ----------------
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    fusion_method: FusionMethod = FusionMethod.RRF
    top_k: int = Field(default=8, ge=1, le=200)
    candidate_k: int = Field(default=50, ge=1, le=1000)
    rrf_k: int = Field(default=60, ge=1)
    lexical_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    use_mmr: bool = False
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)

    # ---------------- reranking ----------------
    rerank_provider: RerankProvider = RerankProvider.LEXICAL
    rerank_model: str = ""
    rerank_enabled: bool = True
    rerank_top_n: int = Field(default=30, ge=1, le=200)

    # ---------------- generation ----------------
    generation_provider: GenerationProvider = GenerationProvider.EXTRACTIVE
    generation_model: str = ""
    generation_max_tokens: int = Field(default=1500, ge=64, le=16000)
    generation_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    context_token_budget: int = Field(
        default=6000, ge=256, description="Max tokens of retrieved context in the prompt"
    )

    # ---------------- verification ----------------
    verify_citations: bool = True
    verify_provider: VerifyProvider = VerifyProvider.LEXICAL
    verify_model: str = ""
    verification_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Support score below which a claim is unsupported"
    )

    # ---------------- ingestion ----------------
    max_document_bytes: int = Field(default=64 * 1024 * 1024, ge=1024)
    web_crawl_max_pages: int = Field(default=25, ge=1, le=1000)
    web_crawl_max_depth: int = Field(default=2, ge=0, le=10)
    web_request_timeout: float = Field(default=20.0, gt=0)
    web_user_agent: str = (
        "kb-chat/0.1 (+https://github.com/anshnt/Chat-With-Your-Entire-Knowledge-Base)"
    )
    github_max_file_bytes: int = Field(default=512 * 1024, ge=1024)

    # ---------------- api ----------------
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    log_level: str = "INFO"

    # ---------------- credentials ----------------
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    voyage_api_key: str = ""
    cohere_api_key: str = ""

    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand(cls, v: object) -> object:
        return Path(str(v)).expanduser() if v else v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @model_validator(mode="after")
    def _fill_provider_defaults(self) -> Settings:
        if not self.embedding_model:
            self.embedding_model = _DEFAULT_EMBEDDING_MODELS[self.embedding_provider]
        if not self.rerank_model:
            self.rerank_model = _DEFAULT_RERANK_MODELS.get(self.rerank_provider, "")
        if not self.generation_model:
            self.generation_model = _DEFAULT_GENERATION_MODELS.get(self.generation_provider, "")
        # The offline hashing embedder is a lexical-overlap model, not a semantic
        # one, so leaving dense weighted above BM25 lets the weaker signal win.
        # Flip the default when the user has not set the weights themselves.
        explicit = self.model_fields_set
        if (
            self.embedding_provider is EmbeddingProvider.HASHING
            and "lexical_weight" not in explicit
            and "dense_weight" not in explicit
        ):
            self.lexical_weight, self.dense_weight = 0.65, 0.35

        # Keep fusion weights meaningful: normalise so they sum to 1.
        total = self.lexical_weight + self.dense_weight
        if total > 0:
            self.lexical_weight /= total
            self.dense_weight /= total
        # Inherit keys from the conventional unprefixed variables too, so an
        # existing shell that already exports ANTHROPIC_API_KEY just works.
        for field, env in (
            ("anthropic_api_key", "ANTHROPIC_API_KEY"),
            ("openai_api_key", "OPENAI_API_KEY"),
            ("voyage_api_key", "VOYAGE_API_KEY"),
            ("cohere_api_key", "COHERE_API_KEY"),
        ):
            if not getattr(self, field):
                setattr(self, field, os.environ.get(env, ""))
        return self

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_filename

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.uploads_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)


_DEFAULT_EMBEDDING_MODELS: dict[EmbeddingProvider, str] = {
    EmbeddingProvider.HASHING: "hashing-ngram-v1",
    EmbeddingProvider.VOYAGE: "voyage-3",
    EmbeddingProvider.OPENAI: "text-embedding-3-small",
    EmbeddingProvider.LOCAL: "sentence-transformers/all-MiniLM-L6-v2",
}

_DEFAULT_RERANK_MODELS: dict[RerankProvider, str] = {
    RerankProvider.NONE: "",
    RerankProvider.LEXICAL: "lexical-coverage-v1",
    RerankProvider.CROSS_ENCODER: "cross-encoder/ms-marco-MiniLM-L-6-v2",
    RerankProvider.COHERE: "rerank-english-v3.0",
    RerankProvider.VOYAGE: "rerank-2",
    RerankProvider.LLM: "",
}

_DEFAULT_GENERATION_MODELS: dict[GenerationProvider, str] = {
    GenerationProvider.EXTRACTIVE: "extractive-v1",
    GenerationProvider.ANTHROPIC: os.environ.get("KB_ANTHROPIC_MODEL", "claude-sonnet-4-5"),
    GenerationProvider.OPENAI: "gpt-4o-mini",
}

# Native dimensionality of each hosted embedding model, used to validate that a
# collection is not queried with vectors of a different width than it was built with.
KNOWN_EMBEDDING_DIMS: dict[str, int] = {
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-code-3": 1024,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings — used by tests that patch the environment."""
    get_settings.cache_clear()
