import threading

import pytest

from app.domain.cancellation import CANCELLED_MESSAGE, NEVER_CANCELLED, Cancellation
from app.domain.errors import Category, ProcessingFailed


def test_a_new_token_is_not_cancelled() -> None:
    token = Cancellation()
    assert not token.cancelled
    token.raise_if_cancelled()


def test_cancel_is_permanent_and_raises_the_fixed_error() -> None:
    token = Cancellation()
    token.cancel()
    token.cancel()
    assert token.cancelled
    with pytest.raises(ProcessingFailed) as caught:
        token.raise_if_cancelled()
    assert caught.value.code == "request_cancelled"
    assert caught.value.category is Category.PROCESSING_FAILED
    assert caught.value.message == CANCELLED_MESSAGE


def test_cancel_from_another_thread_is_seen() -> None:
    token = Cancellation()
    thread = threading.Thread(target=token.cancel)
    thread.start()
    thread.join()
    assert token.cancelled


def test_the_shared_default_is_never_cancelled() -> None:
    with pytest.raises(TypeError):
        NEVER_CANCELLED.cancel()
    assert not NEVER_CANCELLED.cancelled
    NEVER_CANCELLED.raise_if_cancelled()
