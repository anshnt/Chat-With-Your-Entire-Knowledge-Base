"""Dependency wiring for the API.

The knowledge base is a process-level singleton: it owns an embedding model that
may be expensive to load and a SQLite connection pool that is cheap to share.
Construction is lazy so importing the app (for tests, or for ``--reload``) does
not touch the disk or load a model.
"""

from __future__ import annotations

import threading

from kb.config import Settings, get_settings
from kb.knowledge_base import KnowledgeBase

_lock = threading.Lock()
_instance: KnowledgeBase | None = None


def get_knowledge_base() -> KnowledgeBase:
    """FastAPI dependency returning the shared :class:`KnowledgeBase`."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = KnowledgeBase(get_settings())
    return _instance


def set_knowledge_base(instance: KnowledgeBase | None) -> None:
    """Override the singleton — used by tests to inject a temporary corpus."""
    global _instance
    with _lock:
        _instance = instance


def get_api_settings() -> Settings:
    return get_settings()
