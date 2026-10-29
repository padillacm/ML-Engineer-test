"""``FifoSlots``: capacity, arrival order, lanes and their priority, close, interrupted waits."""

import threading
import time
from collections.abc import Callable

import pytest

from app.adapters import slots as slots_module
from app.adapters.slots import FifoSlots


def wait_until(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while not condition():
        assert time.monotonic() < deadline, "the condition never held"
        time.sleep(0.002)


class Caller:
    """A thread that takes a slot of ``lane``, records it in ``order`` and holds it until told."""

    def __init__(self, slots: FifoSlots, name: str, order: list[str], lane: int = 0) -> None:
        self.slots = slots
        self.name = name
        self.order = order
        self.lane = lane
        self.done = threading.Event()
        self.got: bool | None = None
        self.thread = threading.Thread(target=self._run)

    def _run(self) -> None:
        self.got = self.slots.acquire(self.lane)
        if self.got:
            self.order.append(self.name)
            assert self.done.wait(5), "the test never let the caller finish"
            self.slots.release(self.lane)

    def start_and_queue(self) -> "Caller":
        """Start and return once this caller waits in line (so arrival order is fixed)."""
        before = self.slots.waiting(self.lane)
        self.thread.start()
        wait_until(lambda: self.slots.waiting(self.lane) == before + 1)
        return self

    def has_run(self) -> bool:
        return self.name in self.order

    def finish(self) -> None:
        self.done.set()
        self.thread.join(5)
        assert not self.thread.is_alive()


def test_slots_are_free_up_to_the_capacity() -> None:
    slots = FifoSlots(2)

    assert slots.acquire()
    assert slots.acquire()
    assert slots.held() == 2
    slots.release()
    slots.release()
    assert (slots.held(), slots.waiting()) == (0, 0)


def test_waiters_get_the_slot_in_arrival_order() -> None:
    slots = FifoSlots(1)
    order: list[str] = []
    assert slots.acquire()
    callers = [Caller(slots, name, order).start_and_queue() for name in "abcde"]

    slots.release()
    for caller in callers:  # each one gets the slot only after the one before gives it back
        wait_until(caller.has_run)
        caller.finish()

    assert order == list("abcde")


def test_a_releasing_thread_does_not_overtake_a_waiter() -> None:
    slots = FifoSlots(1)
    order: list[str] = []
    assert slots.acquire()
    waiter = Caller(slots, "waiter", order).start_and_queue()

    slots.release()
    assert slots.held() == 1  # handed over to the waiter before this thread can call again
    again = Caller(slots, "releaser again", order).start_and_queue()
    wait_until(lambda: order == ["waiter"])
    waiter.finish()
    wait_until(lambda: order == ["waiter", "releaser again"])
    again.finish()


def test_a_newcomer_queues_behind_waiters_even_when_a_slot_frees() -> None:
    slots = FifoSlots(2)
    order: list[str] = []
    first = Caller(slots, "first", order)
    first.thread.start()
    wait_until(lambda: order == ["first"])
    assert slots.acquire()
    queued = Caller(slots, "queued", order).start_and_queue()
    late = Caller(slots, "late", order).start_and_queue()

    first.finish()
    wait_until(lambda: order == ["first", "queued"])
    assert slots.waiting() == 1  # "late" still waits: both slots are held
    slots.release()
    wait_until(lambda: order == ["first", "queued", "late"])
    queued.finish()
    late.finish()


def test_a_higher_lane_goes_first_and_each_lane_keeps_its_order() -> None:
    slots = FifoSlots(1, (1, 1))
    order: list[str] = []
    assert slots.acquire(1)
    low = [Caller(slots, f"low{i}", order, lane=1).start_and_queue() for i in range(2)]
    high = [Caller(slots, f"high{i}", order, lane=0).start_and_queue() for i in range(2)]

    slots.release(1)
    for caller in high + low:
        wait_until(caller.has_run)
        caller.finish()

    assert order == ["high0", "high1", "low0", "low1"]


def test_a_lane_never_holds_more_than_its_limit() -> None:
    slots = FifoSlots(3, (3, 2))
    order: list[str] = []
    assert slots.acquire(1)
    assert slots.acquire(1)
    blocked = Caller(slots, "third low", order, lane=1).start_and_queue()

    assert slots.acquire(0)  # the slot the low lane may not take is free for the high lane
    assert (slots.held(0), slots.held(1), slots.waiting(1)) == (1, 2, 1)
    slots.release(0)
    assert order == []
    slots.release(1)
    wait_until(lambda: order == ["third low"])
    blocked.finish()
    slots.release(1)
    assert (slots.held(0), slots.held(1)) == (0, 0)


def test_a_lower_lane_waits_while_a_higher_one_has_waiters() -> None:
    slots = FifoSlots(2, (2, 2))
    order: list[str] = []
    assert slots.acquire(0)
    assert slots.acquire(0)
    low = Caller(slots, "low", order, lane=1).start_and_queue()
    high = Caller(slots, "high", order, lane=0).start_and_queue()

    slots.release(0)
    wait_until(lambda: order == ["high"])
    assert slots.waiting(1) == 1
    slots.release(0)
    wait_until(lambda: order == ["high", "low"])
    high.finish()
    low.finish()


def test_close_refuses_later_calls_and_wakes_waiters() -> None:
    slots = FifoSlots(1, (1, 1))
    order: list[str] = []
    assert slots.acquire(0)
    waiters = [Caller(slots, "low", order, lane=1).start_and_queue()]
    waiters.append(Caller(slots, "high", order, lane=0).start_and_queue())

    slots.close()
    slots.close()  # idempotent
    for waiter in waiters:
        waiter.thread.join(5)
        assert waiter.got is False
    assert slots.acquire(0) is False
    assert slots.acquire(1) is False
    slots.release(0)  # the slot held before the close is given back as usual
    assert (slots.held(0), slots.held(1), slots.waiting(0), slots.waiting(1)) == (0, 0, 0, 0)
    assert order == []


class Interrupted(Exception):
    """Stands for a ``KeyboardInterrupt`` raised in a waiting thread."""


def waiter_class(after_wake: Callable[[], None]) -> type:
    """A ``_Waiter`` whose wait runs ``after_wake`` (lock held) once it has been woken."""

    class Condition(threading.Condition):
        def wait(self, timeout: float | None = None) -> bool:
            woken = super().wait(timeout)
            after_wake()
            return woken

    class Waiter(slots_module._Waiter):
        def __init__(self, lock: threading.Lock) -> None:
            super().__init__(lock)
            self.wakeup = Condition(lock)

    return Waiter


def test_a_slot_granted_but_not_yet_taken_when_closing_is_given_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slots = FifoSlots(1)
    order: list[str] = []
    assert slots.acquire()

    def close_before_the_waiter_runs() -> None:
        slots._closed = True  # what close() does, between the hand-over and the wake-up

    monkeypatch.setattr(slots_module, "_Waiter", waiter_class(close_before_the_waiter_runs))
    waiter = Caller(slots, "waiter", order).start_and_queue()
    slots.release()  # hands the slot over to the waiter

    waiter.thread.join(5)
    assert waiter.got is False
    assert (slots.held(), slots.waiting()) == (0, 0)
    assert order == []


def test_an_interrupted_waiter_leaves_the_line(monkeypatch: pytest.MonkeyPatch) -> None:
    slots = FifoSlots(1)
    assert slots.acquire()

    class Waiter(slots_module._Waiter):
        def __init__(self, lock: threading.Lock) -> None:
            super().__init__(lock)
            self.wakeup = self  # type: ignore[assignment]

        def wait(self) -> None:
            raise Interrupted

    monkeypatch.setattr(slots_module, "_Waiter", Waiter)
    with pytest.raises(Interrupted):
        slots.acquire()

    assert (slots.held(), slots.waiting()) == (1, 0)
    monkeypatch.undo()
    slots.release()
    assert slots.acquire()  # the slot was not lost to the interrupted waiter
    slots.release()


def test_a_waiter_interrupted_after_the_hand_over_passes_the_slot_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slots = FifoSlots(1)
    order: list[str] = []
    outcome: list[BaseException] = []
    assert slots.acquire()

    def interrupt() -> None:
        raise Interrupted

    def interrupted_call() -> None:
        try:
            slots.acquire()
        except Interrupted as error:
            outcome.append(error)

    monkeypatch.setattr(slots_module, "_Waiter", waiter_class(interrupt))
    first = threading.Thread(target=interrupted_call)
    first.start()
    wait_until(lambda: slots.waiting() == 1)
    monkeypatch.undo()
    second = Caller(slots, "second", order).start_and_queue()

    slots.release()  # hands the slot to the first waiter, which is interrupted on waking
    first.join(5)
    wait_until(lambda: order == ["second"])
    second.finish()

    assert len(outcome) == 1
    assert (slots.held(), slots.waiting()) == (0, 0)


def test_releasing_a_slot_that_is_not_held_fails() -> None:
    slots = FifoSlots(2, (2, 1))
    assert slots.acquire(0)

    with pytest.raises(RuntimeError, match="lane 1"):
        slots.release(1)
    slots.release(0)


@pytest.mark.parametrize(
    ("capacity", "limits"), [(0, None), (2, ()), (2, (0,)), (2, (3,)), (2, (2, 0))]
)
def test_invalid_sizes_are_refused(capacity: int, limits: tuple[int, ...] | None) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        FifoSlots(capacity, limits)
