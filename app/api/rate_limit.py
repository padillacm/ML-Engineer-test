"""Pure ASGI middleware limiting what one client address may ask at once and over time.

Without authentication the client address is the only handle on a client, so the limits are
keyed by it: one token bucket per ``(address, route class)`` and one counter of in-flight jobs
per address. The address is the peer of the connection (ASGI ``scope["client"]``); the
``X-Forwarded-For`` header is never read, because there is no proxy in front of the service and,
without one, trusting the header would let a client choose its own key (and so escape every
limit). A request without a client (an in-process test client, some servers) counts as one
address, ``UNKNOWN_CLIENT``.

``limits(scope)`` (called per request, like ``BodyLimitMiddleware.limit``) says which limits
apply to the request's route class, or ``None`` for an unlimited route (the health probes, the
audit viewer, unknown paths) and before the lifespan has loaded the settings. A bucket holds at
most ``burst`` tokens, refills at ``per_minute / 60`` tokens per second and gives one token to
each admitted request; a request that finds none is answered 429 ``rate_limited`` with
``Retry-After`` set to the seconds until one token has refilled (rounded up, at least 1). The
job cap (``max_jobs``) counts the requests of an address that were admitted and have not ended;
at the cap a request is answered 429 at once with ``Retry-After: 1``. The cap is checked before
the bucket is charged, so a capped client keeps its tokens for when its jobs have ended (a
refused request cost the service nothing). A rate or cap of zero turns that limit off.

The 429 is rendered here, with the uniform error body and the same failure log line the HTTP
error handler writes, because this middleware sits outside FastAPI's exception handlers (as the
early 413 of ``BodyLimitMiddleware`` is); it is sent before any body byte is read and before any
job slot is taken, and the request context middleware outside it gives it the request id and
the access line.

Memory is bounded: a bucket that is full again (its client idle for at least
``burst / per_minute`` minutes) is dropped on the first request after ``PRUNE_INTERVAL_S``
since the last pruning, and a counter is dropped at zero, so the dictionaries hold at most the
addresses seen in the last pruning interval plus one refill time. No lock is needed: the
middleware runs on the event loop's single thread and never awaits between reading and writing
its state. The clock is injectable (``time.monotonic`` by default) so tests can move time.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from starlette.types import ASGIApp, Receive, Scope, Send

from app.api.errors import error_response, log_failure
from app.domain.errors import Category

RATE_LIMITED = "rate_limited"
RATE_MESSAGE = "too many requests from this client; retry later"
CAP_MESSAGE = "too many jobs of this client are in progress; retry later"
UNKNOWN_CLIENT = "unknown"
"""The address of a request whose scope carries no client."""

PRUNE_INTERVAL_S = 60.0
"""How often, at most, the full buckets are dropped (a pass over every bucket)."""

CAP_RETRY_AFTER_S = 1
"""``Retry-After`` of a request refused by the job cap: a job may end at any moment."""

_RATE_LIMITED_STATUS = 429
_SECONDS_PER_MINUTE = 60.0

type Clock = Callable[[], float]
type Limits = Callable[[Scope], "RouteLimits | None"]


@dataclass(frozen=True, slots=True)
class RouteLimits:
    """The limits of one route class.

    ``per_minute`` sustained requests refill a bucket of ``burst`` tokens (0: no rate limit);
    ``max_jobs`` requests of one address may be in flight at once (0: no cap).
    """

    route_class: str
    per_minute: int
    burst: int
    max_jobs: int = 0


class _Bucket:
    """The tokens of one ``(address, route class)`` at ``updated``, with the limits that last
    charged it (what "full" means when pruning)."""

    __slots__ = ("limits", "tokens", "updated")

    def __init__(self, limits: RouteLimits, now: float) -> None:
        self.limits = limits
        self.tokens = float(limits.burst)
        self.updated = now

    def refill(self, limits: RouteLimits, now: float) -> None:
        rate = limits.per_minute / _SECONDS_PER_MINUTE
        self.tokens = min(float(limits.burst), self.tokens + (now - self.updated) * rate)
        self.updated = now
        self.limits = limits

    def full(self, now: float) -> bool:
        rate = self.limits.per_minute / _SECONDS_PER_MINUTE
        return self.tokens + (now - self.updated) * rate >= self.limits.burst


class RateLimitMiddleware:
    """429 ``rate_limited`` for a client over the rate or the job cap of ``limits(scope)``."""

    def __init__(self, app: ASGIApp, limits: Limits, clock: Clock = time.monotonic) -> None:
        self.app = app
        self.limits = limits
        self.clock = clock
        self._buckets: dict[tuple[str, str], _Bucket] = {}
        self._in_flight: dict[str, int] = {}
        self._pruned_at = clock()

    @property
    def tracked(self) -> int:
        """Buckets kept right now (a test of the pruning reads it)."""
        return len(self._buckets)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        limits = self.limits(scope) if scope["type"] == "http" else None
        if limits is None:
            await self.app(scope, receive, send)
            return
        address = client_address(scope)
        now = self.clock()
        self._prune(now)
        refused = self._refuse(address, limits, now)
        if refused is not None:
            message, retry_after_s = refused
            log_failure(RATE_LIMITED, str(Category.INVALID_INPUT), _RATE_LIMITED_STATUS)
            await error_response(
                _RATE_LIMITED_STATUS,
                message,
                RATE_LIMITED,
                {"Retry-After": str(retry_after_s)},
            )(scope, receive, send)
            return
        if limits.max_jobs == 0:
            await self.app(scope, receive, send)
            return
        self._in_flight[address] = self._in_flight.get(address, 0) + 1
        try:
            await self.app(scope, receive, send)
        finally:
            # The request has ended, however it ended: response complete, failed, cancelled.
            left = self._in_flight[address] - 1
            if left > 0:
                self._in_flight[address] = left
            else:
                del self._in_flight[address]

    def _refuse(self, address: str, limits: RouteLimits, now: float) -> tuple[str, int] | None:
        """The message and ``Retry-After`` seconds of a refusal, or ``None`` once the request
        is admitted (and charged a token)."""
        if limits.max_jobs and self._in_flight.get(address, 0) >= limits.max_jobs:
            return CAP_MESSAGE, CAP_RETRY_AFTER_S
        if limits.per_minute == 0:
            return None
        key = (address, limits.route_class)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = self._buckets[key] = _Bucket(limits, now)
        else:
            bucket.refill(limits, now)
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return None
        missing = 1.0 - bucket.tokens
        rate = limits.per_minute / _SECONDS_PER_MINUTE
        return RATE_MESSAGE, max(1, math.ceil(missing / rate))

    def _prune(self, now: float) -> None:
        if now - self._pruned_at < PRUNE_INTERVAL_S:
            return
        self._pruned_at = now
        for key in [key for key, bucket in self._buckets.items() if bucket.full(now)]:
            del self._buckets[key]


def client_address(scope: Scope) -> str:
    """The peer address of the request, ``UNKNOWN_CLIENT`` when the scope carries none."""
    client = scope.get("client")
    if isinstance(client, (tuple, list)) and client and isinstance(client[0], str):
        return client[0]
    return UNKNOWN_CLIENT
