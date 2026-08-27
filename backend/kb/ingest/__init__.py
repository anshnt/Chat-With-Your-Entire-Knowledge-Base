"""Ingestion: connectors, registry, and the parse→chunk→embed pipeline."""

from kb.ingest.base import Connector, ParsedDocument, Segment
from kb.ingest.pipeline import IngestionPipeline, build_pipeline
from kb.ingest.registry import ConnectorRegistry, default_registry

__all__ = [
    "Connector",
    "ConnectorRegistry",
    "IngestionPipeline",
    "ParsedDocument",
    "Segment",
    "build_pipeline",
    "default_registry",
]
