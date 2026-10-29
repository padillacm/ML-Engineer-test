"""FastAPI application: routers, error handlers, middleware and the lifespan.

``create_app(settings, container)`` assembles the application. Importing this module reads no
environment variable and loads nothing: ``app = create_app()`` (the Uvicorn target) loads the
settings, configures logging and builds the container (models, clients) once, in the lifespan,
and closes what it built on shutdown. Tests pass ``settings`` and a ``container`` of fakes,
which the application uses as given and does not close.

With ``AUDIT_VIEWER_ENABLED`` on, the lifespan also includes the audit viewer's routes
(``app.api.viewer``: ``/audit``, its files and pdf.js), once the settings are known; off, they
do not exist and answer 404 like any unknown path. They pass through the same middleware as
the API (request id, access line, body limit); only their responses carry the viewer's
Content-Security-Policy.

Middleware, outermost first: request context (request id, access line, unexpected errors),
then the body-size limit, chosen by path (``body_limit``): ``MAX_UPLOAD_MB`` plus the form
allowance (``FORM_ALLOWANCE_BYTES``) for the upload routes, ``JSON_BODY_LIMIT_BYTES`` for
everything else (``/api/ask`` takes a question of at most 2,000 characters; the other routes
take no body), then FastAPI's exception handlers.
"""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.types import Scope

from app.api.dependencies import JobLimiter
from app.api.errors import install_error_handlers
from app.api.extract import router as extract_router
from app.api.health import PROBE_ROUTES
from app.api.health import router as health_router
from app.api.middleware import BodyLimitMiddleware, RequestContextMiddleware, route_path
from app.api.rag import router as rag_router
from app.api.viewer import pdfjs_directory, viewer_router
from app.bootstrap import Container, build_container, resolve_concurrency
from app.observability import configure_logging
from app.settings import Settings, load_settings

MULTIPART_FIELD_LIMIT_BYTES = 1024 * 1024
"""Largest non-file form field the multipart parser accepts: Starlette's ``max_part_size``
default (1 MiB), which FastAPI uses (a test keeps them equal). A longer ``names`` field is
refused by the parser itself, so the body limit needs room for no more than this."""

FORM_ENVELOPE_BYTES = 64 * 1024
"""Room for the multipart envelope (boundaries, part headers) and the ``language`` field."""

FORM_ALLOWANCE_BYTES = MULTIPART_FIELD_LIMIT_BYTES + FORM_ENVELOPE_BYTES
"""Room above ``MAX_UPLOAD_MB`` for the rest of an upload form: a ``names`` field as large as
the parser accepts plus the envelope, so no form the parser would accept (a PDF within the
limit, ``names``, ``language``) is cut by the body limit. The PDF itself is held to
``MAX_UPLOAD_MB`` exactly by the handler, and ``names`` to ``MAX_NAMES`` entries."""

JSON_BODY_LIMIT_BYTES = 64 * 1024
"""Body limit of every route other than the uploads: a 2,000-character question is at most
about 24 KB of JSON even with every character escaped (12 bytes for an escaped astral one)."""

UPLOAD_PATHS = frozenset({"/api/extract", "/api/ingest"})
"""Routes whose body is a multipart form with a PDF."""

TITLE = "Document Intelligence Service"
VERSION = "0.1.0"

_logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    """The application; ``settings`` default to the environment, ``container`` to one built
    from the settings at start-up."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active = settings if settings is not None else load_settings()
        configure_logging(
            active.log_level,
            active.log_format,
            exception_messages=active.log_exception_messages,
        )
        begin = time.perf_counter()
        concurrency = resolve_concurrency(active)
        services = (
            container if container is not None else build_container(active, concurrency=concurrency)
        )
        app.state.settings = active
        app.state.services = services
        app.state.limiter = JobLimiter(
            concurrency.max_concurrent_jobs,
            active.busy_wait_s,
            max_questions=active.max_concurrent_questions,
        )
        if active.audit_viewer_enabled:
            include_viewer(app, active)
        _logger.info(
            "service started (language model %s)",
            "configured" if services.llm_configured else "not configured",
            extra={"duration_ms": round((time.perf_counter() - begin) * 1000, 1)},
        )
        try:
            yield
        finally:
            if container is None:
                services.close()
            _logger.info("service stopped")

    app = FastAPI(title=TITLE, version=VERSION, lifespan=lifespan)
    install_error_handlers(app)
    app.add_middleware(BodyLimitMiddleware, limit=lambda scope: body_limit(app, scope))
    # Added last: the outermost. Successful health probes are access-logged at DEBUG.
    app.add_middleware(RequestContextMiddleware, probe_routes=PROBE_ROUTES)
    app.include_router(health_router)
    # The routers carry their "/api" prefix, so each route's path is its full template (the
    # access line logs it).
    app.include_router(extract_router)
    app.include_router(rag_router)
    return app


def include_viewer(app: FastAPI, settings: Settings) -> None:
    """Include the audit viewer's routes, once: a second start of the same application (the
    lifespan runs again) finds them already there (``app.state.viewer_included``)."""
    if getattr(app.state, "viewer_included", False):
        return
    app.include_router(viewer_router(pdfjs_directory(settings.pdfjs_dir)))
    app.state.viewer_included = True


def body_limit(app: FastAPI, scope: Scope) -> int | None:
    """The body limit in bytes of the request in ``scope``, ``None`` before the lifespan has
    loaded the settings."""
    settings: Settings | None = getattr(app.state, "settings", None)
    if settings is None:
        return None
    if route_path(scope) in UPLOAD_PATHS:
        return settings.max_upload_bytes + FORM_ALLOWANCE_BYTES
    return JSON_BODY_LIMIT_BYTES


app = create_app()
