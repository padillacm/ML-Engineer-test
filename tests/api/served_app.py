"""The application over fakes, built for a real Uvicorn process (``--factory``).

``uvicorn tests.api.served_app:create --factory`` serves ``create_app`` with the HTTP test
settings (``api_settings``) updated from the JSON object in ``SERVED_APP_SETTINGS``, and an OCR
fake of ``SERVED_APP_PAGES`` pages of ``SERVED_APP_PAGE_S`` seconds each that logs one
``PAGE_LINE`` per page recognised, so a test can tell from the server's output how far a job
got. ``SERVED_APP_DRAIN_TIMEOUT_S``, when set, replaces the middleware's ``DRAIN_TIMEOUT_S``, so
a test of the drain's time bound need not wait the real five seconds.
"""

import json
import logging
import os

from fastapi import FastAPI

from app.api import middleware
from app.main import create_app
from tests.api.support import Fakes, PagedOcr, api_settings

PAGE_LINE = "fake page recognised"

_logger = logging.getLogger("tests.served_app")


def create() -> FastAPI:
    """The application Uvicorn serves (called once by ``--factory``)."""
    drain_timeout_s = os.environ.get("SERVED_APP_DRAIN_TIMEOUT_S")
    if drain_timeout_s is not None:
        middleware.DRAIN_TIMEOUT_S = float(drain_timeout_s)
    overrides = json.loads(os.environ.get("SERVED_APP_SETTINGS", "{}"))
    settings = api_settings(**overrides)
    ocr = PagedOcr(
        pages=int(os.environ.get("SERVED_APP_PAGES", "10")),
        page_s=float(os.environ.get("SERVED_APP_PAGE_S", "0.2")),
        on_page=lambda: _logger.info(PAGE_LINE),
    )
    fakes = Fakes(ocr=ocr)
    return create_app(settings=settings, container=fakes.container(settings))
