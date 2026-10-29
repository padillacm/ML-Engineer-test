"""Slots handed out in arrival order, with priority lanes: the NER and embedder pools' admission.

A plain ``threading.Condition`` hands a freed slot to whichever thread takes the lock first, and
that is usually the thread that has just released it and calls again: under load one caller
could hold a pool for many calls in a row while another waited (BENCHMARKS 18.3: a question
waited over two minutes behind 12 passage streams). ``FifoSlots`` hands a freed slot over
directly to the first waiter of its queue (per-waiter hand-off), so a caller that arrives, or
comes back, while others wait queues behind them.

Lanes give a fixed priority order: lane 0 is served first, and a lane's waiters are served only
while no waiter of a higher-priority lane waits. Each lane may also hold at most its own number
of slots, which is how the embedder reserves a slot for queries.
"""

import threading
from collections import deque
from collections.abc import Sequence


class _Waiter:
    """One caller waiting for a slot; ``granted`` is set by whoever hands it a slot."""

    __slots__ = ("granted", "wakeup")

    def __init__(self, lock: threading.Lock) -> None:
        self.granted = False
        self.wakeup = threading.Condition(lock)


class FifoSlots:
    """Up to ``capacity`` slots at once, handed out in arrival order within each lane.

    ``lane_limits[i]`` is the most slots lane ``i`` may hold at once (default: one lane that may
    hold them all). Lane 0 has the highest priority. ``close`` refuses every later or waiting
    ``acquire``; slots already held are released as usual.
    """

    def __init__(self, capacity: int, lane_limits: Sequence[int] | None = None) -> None:
        limits = (capacity,) if lane_limits is None else tuple(lane_limits)
        if capacity < 1 or not limits or not all(1 <= limit <= capacity for limit in limits):
            raise ValueError("capacity and lane limits must be at least 1, limits at most capacity")
        self._lock = threading.Lock()
        self._capacity = capacity
        self._limits = limits
        self._held = [0] * len(limits)
        self._queues: tuple[deque[_Waiter], ...] = tuple(deque() for _ in limits)
        self._closed = False

    def acquire(self, lane: int = 0) -> bool:
        """Take a slot of ``lane``, waiting behind earlier callers; ``False`` once closed.

        Returns ``False`` without a slot when the slots are closed before or while waiting.
        """
        with self._lock:
            if self._closed:
                return False
            waiter = _Waiter(self._lock)
            self._queues[lane].append(waiter)
            self._dispatch()
            try:
                while not waiter.granted and not self._closed:
                    waiter.wakeup.wait()
            except BaseException:  # interrupted while waiting: leave the line (or the slot)
                self._leave(lane, waiter)
                raise
            if self._closed:
                self._leave(lane, waiter)
                return False
            return True

    def release(self, lane: int = 0) -> None:
        """Give back a slot of ``lane``; the first waiter that may take it gets it at once."""
        with self._lock:
            if self._held[lane] == 0:
                raise RuntimeError(f"no slot of lane {lane} is held")
            self._held[lane] -= 1
            self._dispatch()

    def close(self) -> None:
        """Refuse every later ``acquire`` and wake the waiting ones (idempotent)."""
        with self._lock:
            self._closed = True
            for queue in self._queues:
                for waiter in queue:
                    waiter.wakeup.notify()

    def waiting(self, lane: int = 0) -> int:
        """How many callers wait for a slot of ``lane``."""
        with self._lock:
            return len(self._queues[lane])

    def held(self, lane: int = 0) -> int:
        """How many slots of ``lane`` are held."""
        with self._lock:
            return self._held[lane]

    def _dispatch(self) -> None:
        """Hand free slots to the heads of the lanes, in priority order (lock held)."""
        if self._closed:
            return
        for lane, queue in enumerate(self._queues):
            while queue and sum(self._held) < self._capacity:
                if self._held[lane] >= self._limits[lane]:
                    break
                waiter = queue.popleft()
                waiter.granted = True
                self._held[lane] += 1
                waiter.wakeup.notify()
            if queue:
                return  # strict priority: lower lanes wait while this one has waiters

    def _leave(self, lane: int, waiter: _Waiter) -> None:
        """Take a waiter that will not use its slot out of the line, or give its slot back."""
        if waiter.granted:
            self._held[lane] -= 1
            self._dispatch()
        else:
            self._queues[lane].remove(waiter)
