"""Error hierarchy of the service core.

Every error carries a stable machine-readable ``code``, a human-readable ``message`` and a
``Category``. Adapters wrap vendor exceptions into these types with fixed messages, so an error
never repeats text (such as an API key) taken from the underlying exception. The wrapped exception
stays available as ``__cause__`` for logging but is never rendered into ``str`` or ``repr``.
Errors can be copied and pickled, so they cross process boundaries intact.
"""

from enum import StrEnum
from typing import ClassVar

KNOWN_CODES: frozenset[str] = frozenset(
    {
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
)


class Category(StrEnum):
    """Closed set of failure categories; the HTTP edge maps each one to a status code."""

    INVALID_INPUT = "invalid_input"
    PROCESSING_FAILED = "processing_failed"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    DEPENDENCY_TIMEOUT = "dependency_timeout"
    INCOMPATIBLE_EMBEDDING = "incompatible_embedding"


class ServiceError(Exception):
    """Base class of every error raised by the core."""

    def __init__(self, code: str, message: str, category: Category) -> None:
        if code not in KNOWN_CODES:
            raise ValueError(f"unknown error code: {code!r}")
        super().__init__(code, message)
        self.code = code
        self.message = message
        self.category = category

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def __reduce__(self) -> tuple[object, ...]:
        # The constructor signatures differ per subclass, so ``args`` cannot be replayed.
        return (_restore, (type(self), self.code, self.message, self.category))


def _restore[E: ServiceError](
    error_type: type[E], code: str, message: str, category: Category
) -> E:
    error = error_type.__new__(error_type)
    Exception.__init__(error, code, message)
    error.code = code
    error.message = message
    error.category = category
    return error


class _CategorizedError(ServiceError):
    """Error with a class-fixed category; callers give a message and, optionally, a code."""

    fixed_category: ClassVar[Category]
    default_code: ClassVar[str | None] = None

    def __init__(self, message: str, code: str | None = None) -> None:
        code = code if code is not None else self.default_code
        if code is None:
            raise ValueError(f"{type(self).__name__} requires an explicit error code")
        super().__init__(code, message, self.fixed_category)


class InvalidInput(_CategorizedError):
    """The request is not acceptable as given."""

    fixed_category = Category.INVALID_INPUT


class ProcessingFailed(_CategorizedError):
    """The input was accepted but could not be processed."""

    fixed_category = Category.PROCESSING_FAILED


class DependencyUnavailable(_CategorizedError):
    """An external dependency could not be reached or refused the call."""

    fixed_category = Category.DEPENDENCY_UNAVAILABLE


class DependencyTimeout(_CategorizedError):
    """An external dependency did not answer in time."""

    fixed_category = Category.DEPENDENCY_TIMEOUT


class IncompatibleEmbedding(_CategorizedError):
    """Stored vectors and the active embedding provider have different identities."""

    fixed_category = Category.INCOMPATIBLE_EMBEDDING
    default_code = "incompatible_embedding"


class GeometryError(ProcessingFailed):
    """A box or coordinate operation received impossible input."""

    default_code = "geometry_error"


class ConfigurationError(InvalidInput, ValueError):
    """A configuration value is invalid."""

    default_code = "invalid_configuration"
