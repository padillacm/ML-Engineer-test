"""A string wrapper that never shows its value in ``str``/``repr`` and refuses to be pickled."""

import hmac
from typing import NoReturn

from app.domain.errors import ConfigurationError

_MASK = "***"
_FIRST_USABLE = 0x21  # "!": the first printable ASCII character after the space
_LAST_USABLE = 0x7E  # "~"
_UNUSABLE = (
    "a secret must be a non-empty string of printable ASCII characters "
    "without whitespace or control characters"
)
_NOT_PICKLABLE = "Secret values cannot be pickled"


class Secret:
    """Holds a credential; only ``reveal()`` returns it.

    The value must be printable ASCII (``!`` to ``~``) with no whitespace or control characters,
    since it is sent verbatim in an HTTP header; anything else raises ``ConfigurationError`` with a
    fixed message that never contains the value. ``str``, ``repr`` and therefore any dataclass or
    f-string that embeds the wrapper show a mask. Equality compares in constant time; the value is
    not hashable so it never lands in a set or a dict key by accident.

    Pickling raises ``TypeError``: a pickle is a byte string that ends up in files, caches and
    queues, and the key would travel in clear inside it. ``copy.copy`` and ``copy.deepcopy`` return
    the same object; the wrapper is immutable, so sharing it is safe and keeps the key in one place.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str) or not _is_usable(value):
            raise ConfigurationError(_UNUSABLE)
        self._value = value

    def reveal(self) -> str:
        """Return the wrapped value; call it only where the value is sent to its destination."""
        return self._value

    def __repr__(self) -> str:
        return f"Secret('{_MASK}')"

    def __str__(self) -> str:
        return _MASK

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Secret):
            return NotImplemented
        return hmac.compare_digest(self._value.encode(), other._value.encode())

    __hash__ = None  # type: ignore[assignment]

    def __reduce__(self) -> NoReturn:
        raise TypeError(_NOT_PICKLABLE)

    def __reduce_ex__(self, protocol: object) -> NoReturn:
        raise TypeError(_NOT_PICKLABLE)

    def __getstate__(self) -> NoReturn:
        raise TypeError(_NOT_PICKLABLE)

    def __copy__(self) -> "Secret":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "Secret":
        return self


def _is_usable(value: str) -> bool:
    return bool(value) and all(_FIRST_USABLE <= ord(char) <= _LAST_USABLE for char in value)
