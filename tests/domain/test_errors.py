import copy
import multiprocessing
import pickle
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import cast

import pytest

from app.domain.errors import (
    KNOWN_CODES,
    Category,
    ConfigurationError,
    DependencyTimeout,
    DependencyUnavailable,
    GeometryError,
    IncompatibleEmbedding,
    InvalidInput,
    ProcessingFailed,
    ServiceError,
)

# Leaf errors share the (message, code) constructor, unlike the base class.
LeafError = Callable[[str, str], ServiceError]

# (class, category, a code valid for it)
CASES: list[tuple[LeafError, Category, str]] = [
    (InvalidInput, Category.INVALID_INPUT, "not_a_pdf"),
    (InvalidInput, Category.INVALID_INPUT, "pdf_unreadable"),
    (InvalidInput, Category.INVALID_INPUT, "unsupported_language"),
    (ProcessingFailed, Category.PROCESSING_FAILED, "ocr_failed"),
    (ProcessingFailed, Category.PROCESSING_FAILED, "ner_failed"),
    (DependencyUnavailable, Category.DEPENDENCY_UNAVAILABLE, "llm_unavailable"),
    (DependencyTimeout, Category.DEPENDENCY_TIMEOUT, "vector_store_timeout"),
    (IncompatibleEmbedding, Category.INCOMPATIBLE_EMBEDDING, "incompatible_embedding"),
    (GeometryError, Category.PROCESSING_FAILED, "geometry_error"),
    (ConfigurationError, Category.INVALID_INPUT, "invalid_configuration"),
]


@pytest.mark.parametrize(("error_type", "category", "code"), CASES)
def test_error_exposes_code_message_and_category(
    error_type: LeafError, category: Category, code: str
) -> None:
    error = error_type("something went wrong", code)

    assert error.code == code
    assert error.message == "something went wrong"
    assert error.category is category
    assert str(error) == f"{code}: something went wrong"


@pytest.mark.parametrize(("error_type", "category", "code"), CASES)
def test_every_error_is_caught_by_the_base_class(
    error_type: LeafError, category: Category, code: str
) -> None:
    with pytest.raises(ServiceError) as caught:
        raise error_type("boom", code)

    assert caught.value.category is category


def test_subclasses_with_a_natural_code_default_to_it() -> None:
    assert GeometryError("bad box").code == "geometry_error"
    assert ConfigurationError("bad value").code == "invalid_configuration"
    assert IncompatibleEmbedding("other model").code == "incompatible_embedding"


def test_subclass_without_a_natural_code_requires_one() -> None:
    with pytest.raises(ValueError, match="explicit error code"):
        InvalidInput("no code given")


def test_configuration_error_is_also_a_value_error() -> None:
    assert isinstance(ConfigurationError("bad"), ValueError)
    assert isinstance(GeometryError("bad"), ProcessingFailed)


def test_base_error_rejects_unknown_code() -> None:
    with pytest.raises(ValueError, match="unknown error code"):
        ServiceError("made_up", "message", Category.INVALID_INPUT)


@pytest.mark.parametrize("error_type", [InvalidInput, ProcessingFailed])
def test_subclasses_reject_unknown_codes(error_type: LeafError) -> None:
    with pytest.raises(ValueError, match="unknown error code"):
        error_type("message", "made_up")


def test_registry_is_closed_and_snake_case() -> None:
    assert set(KNOWN_CODES) == {
        "not_a_pdf",
        "pdf_unreadable",
        "empty_file",
        "too_large",
        "too_many_pages",
        "invalid_names",
        "unsupported_language",
        "ocr_failed",
        "ner_failed",
        "llm_unavailable",
        "llm_timeout",
        "vector_store_unavailable",
        "vector_store_timeout",
        "incompatible_embedding",
        "geometry_error",
        "invalid_configuration",
    }
    assert all(code.islower() and " " not in code for code in KNOWN_CODES)


def _raise_wrapping(vendor_text: str) -> None:
    try:
        raise RuntimeError(vendor_text)
    except RuntimeError as original:
        message = "LLM service is unavailable"
        raise DependencyUnavailable(message, "llm_unavailable") from original


def test_wrapped_vendor_text_never_reaches_the_rendered_error() -> None:
    """``__cause__`` is kept for logging but is not part of str, repr, code or message."""
    api_key = "sk-test-not-a-real-key"
    vendor_text = f"401 Unauthorized for key {api_key}"
    with pytest.raises(DependencyUnavailable) as caught:
        _raise_wrapping(vendor_text)

    error = caught.value
    assert error.code == "llm_unavailable"
    assert error.message == "LLM service is unavailable"
    for rendered in (str(error), error.code, error.message, repr(error)):
        assert api_key not in rendered
        assert vendor_text not in rendered
    assert isinstance(error.__cause__, RuntimeError)


def _pickle_roundtrip[T: ServiceError](error: T) -> T:
    return cast("T", pickle.loads(pickle.dumps(error)))  # noqa: S301 - round-trips our own objects


CLONES: list[Callable[[ServiceError], ServiceError]] = [
    copy.copy,
    copy.deepcopy,
    _pickle_roundtrip,
]


def _assert_same_error(restored: ServiceError, original: ServiceError) -> None:
    assert type(restored) is type(original)
    assert restored.code == original.code
    assert restored.message == original.message
    assert restored.category is original.category
    assert str(restored) == str(original)
    assert restored.args == original.args


@pytest.mark.parametrize(("error_type", "category", "code"), CASES)
@pytest.mark.parametrize("clone", CLONES)
def test_errors_survive_copy_and_pickle(
    error_type: LeafError,
    category: Category,
    code: str,
    clone: Callable[[ServiceError], ServiceError],
) -> None:
    original = error_type("something went wrong", code)

    _assert_same_error(clone(original), original)


@pytest.mark.parametrize("clone", CLONES)
def test_base_service_error_survives_copy_and_pickle(
    clone: Callable[[ServiceError], ServiceError],
) -> None:
    original = ServiceError("ocr_failed", "page unreadable", Category.PROCESSING_FAILED)

    _assert_same_error(clone(original), original)


def test_restored_errors_keep_their_class_hierarchy() -> None:
    restored = _pickle_roundtrip(ConfigurationError("bad value"))
    timeout = _pickle_roundtrip(DependencyTimeout("slow", "llm_timeout"))

    assert isinstance(restored, ValueError)
    assert isinstance(restored, InvalidInput)
    assert (timeout.code, timeout.message) == ("llm_timeout", "slow")


def _echo(error: ServiceError) -> ServiceError:
    return error


def test_error_crosses_a_process_boundary() -> None:
    original = DependencyTimeout("slow", "llm_timeout")

    # "spawn": forking a multi-threaded process (torch threads may be alive) can deadlock.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        restored = pool.submit(_echo, original).result(timeout=60)

    _assert_same_error(restored, original)
