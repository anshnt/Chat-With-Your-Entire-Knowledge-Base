"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from kb import __version__
from kb.api.deps import get_knowledge_base
from kb.api.routers import corpus, ingest, search
from kb.api.schemas import ErrorResponse, HealthResponse
from kb.config import Settings, get_settings
from kb.errors import KBError

log = logging.getLogger(__name__)

DESCRIPTION = """
Retrieval-augmented search over PDFs, Markdown, Notion exports, websites,
GitHub repositories and YouTube transcripts.

Every result carries a **locator** — a typed source address — so a citation
links to the exact page, line range or timestamp it came from, and every result
carries its **per-stage scores**, so a ranking can be explained rather than
trusted.
""".strip()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Called by ``kb serve`` and by the test suite."""
    settings = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Warm the knowledge base at startup so the first request does not pay
        # for loading an embedding model.
        knowledge_base = get_knowledge_base()
        log.info(
            "knowledge base ready: %s (embedder=%s, dim=%s)",
            settings.db_path,
            knowledge_base.embedder.model,
            knowledge_base.embedder.dim,
        )
        yield
        knowledge_base.close()

    app = FastAPI(
        title="Chat With Your Entire Knowledge Base",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(KBError)
    async def kb_error_handler(request: Request, exc: KBError) -> JSONResponse:
        """Map domain errors to their declared status codes.

        Every :class:`KBError` carries a stable ``code``, so clients branch on
        that rather than on message text.
        """
        log.info("%s %s -> %s: %s", request.method, request.url.path, exc.code, exc.message)
        return JSONResponse(
            status_code=exc.http_status,
            content=ErrorResponse(
                code=exc.code, message=exc.message, details=exc.details
            ).model_dump(),
        )

    @app.get("/api/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        knowledge_base = get_knowledge_base()
        return HealthResponse(
            version=__version__,
            embedding_model=knowledge_base.embedder.model,
            embedding_dim=knowledge_base.embedder.dim,
            retrieval_strategy=settings.retrieval_strategy.value,
            reranker=(
                getattr(knowledge_base.reranker, "name", None) if knowledge_base.reranker else None
            ),
            generator=knowledge_base.generator.name,
            generation_model=knowledge_base.generator.model,
            verifier=(
                knowledge_base.verifier.name if knowledge_base.verifier is not None else None
            ),
            connectors=knowledge_base.registry.names(),
        )

    app.include_router(search.router)
    app.include_router(ingest.router)
    app.include_router(corpus.router)

    _mount_optional_routers(app)

    # Uploaded files are served so that PDF citations resolve: a locator's
    # ``/files/report.pdf#page=12`` needs the bytes to still be reachable.
    settings.ensure_dirs()
    app.mount("/files", StaticFiles(directory=str(settings.uploads_dir)), name="files")

    return app


def _mount_optional_routers(app: FastAPI) -> None:
    """Include routers from modules that may not be present yet.

    Keeps feature layers (chat, evaluation, visualisation) additive: each one
    ships its own router and is picked up by existing, without this file needing
    to know it is coming.
    """
    for module_name in (
        "kb.api.routers.chat",
        "kb.api.routers.evaluation",
        "kb.api.routers.visualization",
    ):
        try:
            module = __import__(module_name, fromlist=["router"])
        except ImportError:
            continue
        router = getattr(module, "router", None)
        if router is not None:
            app.include_router(router)


app = create_app
"""``uvicorn kb.api.app:app`` expects an application object or factory; the CLI
passes ``factory=True``."""
