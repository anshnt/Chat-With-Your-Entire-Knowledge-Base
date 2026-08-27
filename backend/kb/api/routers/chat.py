"""Answering endpoints.

Two shapes for the same operation:

* ``POST /api/ask`` — one JSON response with the answer, its citations, and the
  retrieval that produced it.
* ``POST /api/ask/stream`` — Server-Sent Events, because a grounded answer over a
  large context takes seconds and a blank screen for that long is the difference
  between a demo and a tool.

Citations arrive with the terminal ``done`` event rather than incrementally: a
marker may be half-emitted mid-stream, and rendering a citation chip for `[` is
worse than rendering it a beat later.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from kb.api.deps import get_knowledge_base
from kb.api.schemas import AskRequest, AskResponse
from kb.knowledge_base import KnowledgeBase

router = APIRouter(prefix="/api", tags=["chat"])


def _retrieval_overrides(payload: AskRequest) -> dict[str, object]:
    return {
        "collection": payload.collection,
        "top_k": payload.top_k,
        "candidate_k": payload.candidate_k,
        "strategy": payload.strategy,
        "fusion": payload.fusion,
        "rerank": payload.rerank,
        "use_mmr": payload.use_mmr,
        "source_types": payload.source_types,
        "document_ids": payload.document_ids,
    }


@router.post("/ask", response_model=AskResponse, summary="Answer a question with citations")
def ask(payload: AskRequest, kb: KnowledgeBase = Depends(get_knowledge_base)) -> AskResponse:
    """Retrieve and answer.

    The response carries the full retrieval diagnostics alongside the answer, so
    a bad answer can be attributed to retrieval or to generation instead of being
    guessed at.
    """
    answer = kb.ask(payload.query, **_retrieval_overrides(payload))
    return AskResponse.from_answer(answer, include_retrieval=payload.include_retrieval)


@router.post("/ask/stream", summary="Answer a question, streamed as SSE")
def ask_stream(
    payload: AskRequest, kb: KnowledgeBase = Depends(get_knowledge_base)
) -> StreamingResponse:
    """Stream an answer as Server-Sent Events.

    Event types:

    ``delta``  ``{"text": "..."}``     — incremental answer text
    ``done``   ``{"answer": {...}}``   — the finished answer with citations
    ``error``  ``{"message": "..."}``  — generation failed mid-stream
    """

    def events() -> Iterator[str]:
        try:
            for delta, answer in kb.ask_stream(payload.query, **_retrieval_overrides(payload)):
                if answer is not None:
                    body = AskResponse.from_answer(
                        answer, include_retrieval=payload.include_retrieval
                    )
                    yield _sse("done", {"answer": body.model_dump(mode="json")})
                elif delta:
                    yield _sse("delta", {"text": delta})
        except Exception as exc:
            # Headers are already sent, so an HTTP error status is no longer
            # available; the client learns about the failure from the event.
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Stops nginx buffering the stream into a single delivery.
            "X-Accel-Buffering": "no",
        },
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
