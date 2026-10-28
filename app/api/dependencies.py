"""FastAPI dependencies: the use cases, settings and job limiter of the running application.

The lifespan stores them on ``app.state`` (``services``, ``settings``, ``limiter``); routers
receive them through these functions and never import a concrete adapter or the composition
root. ``Services`` is the structural type of what the routers need from the container.
The functions are ``async`` because they only read attributes: FastAPI would otherwise run
each of them in the thread pool.
"""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Final, Protocol

import anyio
import anyio.abc
import anyio.lowlevel
import anyio.to_thread
from fastapi import Request
from starlette.types import Receive

from app.application.ask_question import AskQuestion
from app.application.extract_names import ExtractNames
from app.application.ingest_document import IngestDocument
from app.domain.cancellation import CANCELLED_MESSAGE, Cancellation
from app.domain.errors import ConfigurationError, DependencyUnavailable, ProcessingFailed
from app.settings import Settings

BUSY_MESSAGE = "the service is busy; retry later"

READINESS_THREADS: Final = 2
"""Readiness checks run at once (a probe runs its checks one after the other)."""


class Services(Protocol):
    """The use cases and readiness checks the HTTP edge calls.

    A readiness check returns its status when it passes and raises when it fails.
    """

    @property
    def extract(self) -> ExtractNames: ...

    @property
    def ingest(self) -> IngestDocument: ...

    @property
    def ask(self) -> AskQuestion: ...

    @property
    def readiness(self) -> Sequence[tuple[str, Callable[[], str]]]: ...


class _Pool:
    """``size`` slots, awaited on the event loop, and ``size`` threads to run the work in."""

    def __init__(self, size: int) -> None:
        self.slots = anyio.Semaphore(size, max_value=size)
        self.threads = anyio.CapacityLimiter(size)

    async def take(self, wait_s: float) -> None:
        """Take a slot, waiting up to ``wait_s`` seconds; ``service_busy`` when none frees up."""
        try:
            self.slots.acquire_nowait()
        except anyio.WouldBlock:
            pass
        else:
            return
        try:
            with anyio.fail_after(wait_s):
                await self.slots.acquire()
        except TimeoutError:
            raise DependencyUnavailable(BUSY_MESSAGE, code="service_busy") from None


class JobLimiter:
    """Where the blocking work of each request runs, and how much of it runs at once.

    Heavy jobs (extraction, ingestion) and questions have ``max_jobs`` and ``max_questions``
    slots: a request waits on the event loop, holding no thread, up to ``wait_s`` seconds for
    a slot of its kind and gets ``DependencyUnavailable`` (code ``service_busy``) when none
    frees up in time; with a slot it runs in a thread of its kind's own pool (as many threads
    as slots) and frees the slot when the work ends, however it ends. Readiness checks
    (``run_check``) have their own threads and wait for no slot. Each pool is a separate AnyIO
    capacity limiter, so none of this work takes the threads of AnyIO's default pool (40),
    which the framework itself uses, and jobs, questions and checks never wait for each other.

    A job runs with a ``Cancellation`` of its own. Given the request's ``receive`` (its body
    already read), ``run_job`` watches it for the client's ``http.disconnect`` (a ``receive``
    that raises counts as one): a request still waiting for a slot then gives up (no slot
    taken) and a running job is cancelled, so the use case stops at its next check. A thread
    cannot be interrupted, so the slot is freed only when the job's thread returns. Either way
    the request ends with ``ProcessingFailed`` (code ``request_cancelled``, logged as 499), also
    when the job completed after its client left, since no response can reach the client, and
    when the wait for a slot timed out after the client had left; a job that failed keeps its
    own error (a genuine failure is worth its 5xx, whoever is still listening). Cancelling the
    request's task (the server's shutdown after its graceful timeout) cancels the job too, so
    its thread returns at the job's next check instead of holding the shutdown.
    """

    def __init__(self, max_jobs: int, wait_s: float, *, max_questions: int) -> None:
        for name, size in (("max_jobs", max_jobs), ("max_questions", max_questions)):
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ConfigurationError(f"{name} must be an integer >= 1, got {size!r}")
        if not 0 <= wait_s < float("inf"):
            raise ConfigurationError(f"wait_s must be a finite number >= 0, got {wait_s!r}")
        self._wait_s = float(wait_s)
        self._jobs = _Pool(max_jobs)
        self._questions = _Pool(max_questions)
        self._readiness_threads = anyio.CapacityLimiter(READINESS_THREADS)

    @property
    def free_slots(self) -> int:
        """Job slots free right now."""
        return self._jobs.slots.value

    @property
    def waiting_jobs(self) -> int:
        """Requests waiting for a job slot right now."""
        return self._jobs.slots.statistics().tasks_waiting

    @property
    def free_question_slots(self) -> int:
        """Question slots free right now."""
        return self._questions.slots.value

    @property
    def waiting_questions(self) -> int:
        """Questions waiting for a slot right now."""
        return self._questions.slots.statistics().tasks_waiting

    async def run_job[T](
        self, job: Callable[[Cancellation], T], receive: Receive | None = None
    ) -> T:
        """Run ``job(cancel)`` in a job thread once a slot is free: ``service_busy`` after
        ``wait_s``, ``request_cancelled`` once the client of ``receive`` disconnects."""
        cancel = Cancellation()
        waiting = anyio.CancelScope()
        results: list[T] = []
        failure: Exception | None = None
        stop = anyio.CancelScope()  # the job's end stops the watcher, leaving the token
        async with anyio.create_task_group() as watcher:
            # Started (not just scheduled) before the job: cancelled from outside, it is
            # already inside the block that cancels the job.
            await watcher.start(_watch, receive, cancel, waiting, stop)
            try:
                results.append(await self._run_job(job, cancel, waiting))
            # Raised again below: raised here, the task group would wrap it in a group.
            except Exception as error:  # noqa: BLE001 - re-raised as is after the group
                failure = error
            finally:
                gone = cancel.cancelled
                stop.cancel()
        # Cancelled from outside, the caller gets its cancellation, not the job's outcome.
        await anyio.lowlevel.checkpoint_if_cancelled()
        if failure is not None:
            raise failure
        if gone:
            raise _cancelled()
        return results[0]

    async def run_question[T](self, question: Callable[[], T]) -> T:
        """Run ``question`` in a question thread once a question slot is free;
        ``service_busy`` after ``wait_s``."""
        await self._questions.take(self._wait_s)
        try:
            return await anyio.to_thread.run_sync(question, limiter=self._questions.threads)
        finally:
            self._questions.slots.release()

    async def run_check[T](self, check: Callable[[], T], timeout_s: float) -> T:
        """Run ``check`` in a readiness thread; ``TimeoutError`` after ``timeout_s`` seconds.

        A check that times out is abandoned: its thread runs on to its own end (the vector
        store's ping has a timeout of about ``timeout_s`` itself), but the probe answers now.
        """
        with anyio.fail_after(timeout_s):
            return await anyio.to_thread.run_sync(
                check, limiter=self._readiness_threads, abandon_on_cancel=True
            )

    async def _run_job[T](
        self, job: Callable[[Cancellation], T], cancel: Cancellation, waiting: anyio.CancelScope
    ) -> T:
        try:
            with waiting:  # cancelled by the watcher when the client leaves
                await self._jobs.take(self._wait_s)
        except DependencyUnavailable:
            if cancel.cancelled:
                raise _cancelled() from None  # busy, but nobody is waiting for the answer
            raise
        if waiting.cancelled_caught:
            raise _cancelled()  # gave up waiting: no slot was taken
        try:
            cancel.raise_if_cancelled()  # the client left as the slot was taken
            return await anyio.to_thread.run_sync(partial(job, cancel), limiter=self._jobs.threads)
        finally:
            self._jobs.slots.release()


async def _watch(
    receive: Receive | None,
    cancel: Cancellation,
    waiting: anyio.CancelScope,
    stop: anyio.CancelScope,
    *,
    task_status: anyio.abc.TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
) -> None:
    """Cancel ``cancel`` and ``waiting`` once the client of ``receive`` disconnects, or once this
    task is cancelled from outside (the request's task is being cancelled); ``stop``, cancelled
    when the job has ended, ends the watch and leaves them as they are.

    Called once the body has been read: the server's ``receive`` then blocks until the client
    disconnects or the response is complete (by then this task has been stopped). A ``receive``
    that raises is taken as a disconnect: no answer can be delivered through it either.
    """
    try:
        with stop:
            task_status.started()
            if receive is None:
                await anyio.sleep_forever()
            else:
                await _wait_for_disconnect(receive)
    finally:
        if not stop.cancelled_caught:  # the client left, or the request is being cancelled
            cancel.cancel()
            waiting.cancel()


async def _wait_for_disconnect(receive: Receive) -> None:
    try:
        while (await receive())["type"] != "http.disconnect":
            pass
    except Exception:  # noqa: BLE001 - the connection is unusable, whatever the error says
        return


def _cancelled() -> ProcessingFailed:
    return ProcessingFailed(CANCELLED_MESSAGE, code="request_cancelled")


async def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


async def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


async def get_limiter(request: Request) -> JobLimiter:
    limiter: JobLimiter = request.app.state.limiter
    return limiter


async def get_extract(request: Request) -> ExtractNames:
    return (await get_services(request)).extract


async def get_ingest(request: Request) -> IngestDocument:
    return (await get_services(request)).ingest


async def get_ask(request: Request) -> AskQuestion:
    return (await get_services(request)).ask
