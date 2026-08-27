"""Ingestion endpoints."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile

from kb.api.deps import get_knowledge_base
from kb.api.schemas import IngestRequest, IngestResponse
from kb.errors import ValidationError
from kb.knowledge_base import KnowledgeBase
from kb.models import IngestionReport

router = APIRouter(prefix="/api", tags=["ingest"])


def _to_response(report: IngestionReport) -> IngestResponse:
    return IngestResponse(
        documents_created=report.documents_created,
        chunks_created=report.chunks_created,
        documents_skipped=report.documents_skipped,
        duplicates_skipped=report.duplicates_skipped,
        errors=report.errors,
        elapsed_ms=report.elapsed_ms,
        documents=report.documents,
    )


@router.post("/ingest", response_model=IngestResponse, summary="Ingest a source or pasted text")
def ingest(
    payload: IngestRequest, kb: KnowledgeBase = Depends(get_knowledge_base)
) -> IngestResponse:
    """Ingest a path, directory, glob or URL — or a document pasted as ``text``."""
    if payload.text:
        report = kb.ingest_text(
            payload.text,
            title=payload.title or "Pasted document",
            collection=payload.collection,
            embed=payload.embed,
        )
    elif payload.source:
        options: dict[str, Any] = dict(payload.options)
        if payload.title:
            options["title"] = payload.title
        report = kb.ingest(
            payload.source,
            collection=payload.collection,
            embed=payload.embed,
            **options,
        )
    else:
        raise ValidationError("provide either 'source' or 'text'")
    return _to_response(report)


@router.post("/ingest/upload", response_model=IngestResponse, summary="Upload and ingest a file")
async def upload(
    file: UploadFile = File(...),
    collection: str = Form("default"),
    title: str | None = Form(None),
    kb: KnowledgeBase = Depends(get_knowledge_base),
) -> IngestResponse:
    """Store an uploaded file under the data directory, then ingest it.

    The file is kept rather than parsed and discarded because PDF citations link
    back to it — ``/files/{name}#page=12`` only resolves if the bytes are still
    on disk.
    """
    if not file.filename:
        raise ValidationError("upload requires a filename")

    kb.settings.ensure_dirs()
    safe_name = Path(file.filename).name
    destination = kb.settings.uploads_dir / safe_name
    destination = _unique_path(destination)

    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    options: dict[str, Any] = {"file_url": f"/files/{destination.name}"}
    if title:
        options["title"] = title
    report = kb.ingest(str(destination), collection=collection, **options)
    return _to_response(report)


@router.post("/embed", response_model=dict, summary="Embed any chunks still missing vectors")
def embed(
    collection: str = "default",
    kb: KnowledgeBase = Depends(get_knowledge_base),
) -> dict:
    """Backfill embeddings.

    Ingestion is designed to leave a document searchable over BM25 even if
    embedding fails, so this is the resume path.
    """
    embedded = kb.embed_pending(collection=collection)
    return {"embedded": embedded, "collection": collection}


def _unique_path(path: Path) -> Path:
    """Avoid clobbering an existing upload with the same name."""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for i in range(1, 1000):
        candidate = path.with_name(f"{stem}-{i}{suffix}")
        if not candidate.exists():
            return candidate
    raise ValidationError(f"too many files named {path.name}")
