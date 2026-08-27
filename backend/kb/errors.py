"""Exception hierarchy for kb.

Every error carries a stable ``code`` so the HTTP layer can map failures to
status codes without string-matching messages.
"""

from __future__ import annotations


class KBError(Exception):
    """Base class for every error raised by kb."""

    code: str = "kb_error"
    http_status: int = 500

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


class ConfigurationError(KBError):
    """Raised when settings are missing or mutually inconsistent."""

    code = "configuration_error"
    http_status = 500


class MissingDependencyError(ConfigurationError):
    """An optional extra is required for the requested provider."""

    code = "missing_dependency"

    def __init__(self, package: str, extra: str) -> None:
        super().__init__(
            f"{package!r} is not installed. Install it with: pip install 'kb-chat[{extra}]'",
            details={"package": package, "extra": extra},
        )


class NotFoundError(KBError):
    """A requested entity does not exist."""

    code = "not_found"
    http_status = 404


class ValidationError(KBError):
    """Caller-supplied input is invalid."""

    code = "validation_error"
    http_status = 422


class IngestionError(KBError):
    """A source could not be read, parsed, or chunked."""

    code = "ingestion_error"
    http_status = 400


class UnsupportedSourceError(IngestionError):
    """No registered connector can handle the given source."""

    code = "unsupported_source"


class ProviderError(KBError):
    """An upstream provider (embeddings, reranker, LLM) failed."""

    code = "provider_error"
    http_status = 502


class RetrievalError(KBError):
    """Retrieval could not be completed."""

    code = "retrieval_error"
    http_status = 500


class EvaluationError(KBError):
    """An evaluation run could not be completed."""

    code = "evaluation_error"
    http_status = 400
