"""``GET /health`` (liveness) and ``GET /health/ready`` (readiness).

Liveness calls nothing and runs on the event loop, so it answers while every worker thread is
busy. Readiness runs every check of the container (the vector store's ping, whether a language
model is configured; the model itself is never contacted) one after the other in the readiness
threads (``JobLimiter.run_check``), so it never waits behind jobs or questions, and answers 503
naming the failing checks. A check that raises, or takes longer than ``READINESS_TIMEOUT_S``,
is ``unavailable`` (the vector store's ping itself times out after about that long, so the
abandoned check's thread ends soon after).
"""

import logging
from collections.abc import Callable
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel

from app.api.dependencies import JobLimiter, Services, get_limiter, get_services, get_settings
from app.domain.errors import ServiceError
from app.models.schemas import HealthResponse
from app.settings import Settings

UNAVAILABLE = "unavailable"
LIVENESS_PATH = "/health"
READINESS_PATH = "/health/ready"
PROBE_ROUTES = (LIVENESS_PATH, READINESS_PATH)
"""The probe endpoints: their successful requests are access-logged at DEBUG (app.main)."""

router = APIRouter(tags=["health"])
_logger = logging.getLogger(__name__)


class ReadinessResponse(BaseModel):
    """Overall readiness and the status of each check."""

    status: Literal["ready", "not_ready"]
    checks: dict[str, str]


@router.get(LIVENESS_PATH, response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness: the process is up and serving."""
    return HealthResponse(status="ok")


@router.get(
    READINESS_PATH,
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse, "description": "A dependency is not ready"}},
)
async def ready(
    services: Annotated[Services, Depends(get_services)],
    limiter: Annotated[JobLimiter, Depends(get_limiter)],
    settings: Annotated[Settings, Depends(get_settings)],
    response: Response,
) -> ReadinessResponse:
    """Readiness: 200 when every check passes, 503 otherwise."""
    timeout_s = settings.readiness_timeout_s
    checks = {
        name: await _run(name, check, limiter, timeout_s) for name, check in services.readiness
    }
    is_ready = UNAVAILABLE not in checks.values()
    if not is_ready:
        response.status_code = 503
    return ReadinessResponse(status="ready" if is_ready else "not_ready", checks=checks)


async def _run(name: str, check: Callable[[], str], limiter: JobLimiter, timeout_s: float) -> str:
    try:
        return await limiter.run_check(check, timeout_s)
    except TimeoutError:
        _logger.warning(
            "readiness check failed",
            extra={"check": name, "ready": False, "code": "readiness_timeout"},
        )
    except ServiceError as error:
        _logger.warning(
            "readiness check failed",
            extra={
                "check": name,
                "ready": False,
                "code": error.code,
                "category": str(error.category),
            },
        )
    except Exception:
        _logger.exception("readiness check failed", extra={"check": name, "ready": False})
    return UNAVAILABLE
