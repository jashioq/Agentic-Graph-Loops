"""Reading typed fields out of JSON that may be wrong.

Layer: runtime. A pure leaf: imports nothing else in AGL. What it narrows is
agent output or a document a person hand-edited, so every failure is a message
someone has to act on.

Every function raises `InvalidFieldError`, the way `paths` raises
`InvalidNameError`: one error, stated here, and callers ask for nothing. A caller
that wants its own type catches it once around the read it is doing and re-raises
— the message is already the one a person needs, so passing it through is enough.
"""

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "InvalidFieldError",
    "as_object",
    "as_optional_text",
    "as_text",
    "as_text_list",
    "as_whole_number",
    "reject_unknown_fields",
    "require_fields",
    "type_name",
]


class InvalidFieldError(Exception):
    """Raised when a field is absent, or is not the type it has to be."""


def as_object(value: Any, where: str) -> dict[str, Any]:
    """Narrow `value` to a JSON object or raise `InvalidFieldError`."""
    if not isinstance(value, dict):
        raise InvalidFieldError(f"{where} must be an object, got {type_name(value)}")
    return value


def require_fields(fields: Mapping[str, Any], required: Sequence[str], where: str) -> None:
    """Raise for the first name in `required` that `fields` does not have."""
    for name in required:
        if name not in fields:
            raise InvalidFieldError(f"{where}: missing required field {name!r}")


def reject_unknown_fields(fields: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    """Raise for a field outside `allowed`, rather than ignoring it.

    A misspelled key that is quietly dropped is a person's edit that appeared to
    take and did not, which is worse than being told.
    """
    for name in fields:
        if name not in allowed:
            raise InvalidFieldError(f"{where}: unknown field {name!r}")


def as_text(value: Any, name: str, where: str) -> str:
    """Narrow `value` to non-empty text or raise `InvalidFieldError`."""
    if not isinstance(value, str) or not value.strip():
        raise InvalidFieldError(f"{where}: {name} must be non-empty text, got {value!r}")
    return value


def as_optional_text(value: Any, name: str, where: str) -> str | None:
    """`as_text`, except that `None` is an answer rather than a failure."""
    if value is None:
        return None
    return as_text(value, name, where)


def as_text_list(
    value: Any, name: str, where: str, *, allow_empty: bool = True
) -> tuple[str, ...]:
    """Narrow `value` to a tuple of non-empty strings or raise `InvalidFieldError`.

    `allow_empty=False` also refuses an array with nothing in it, for a field
    whose whole point is to list something.
    """
    if not isinstance(value, list):
        raise InvalidFieldError(f"{where}: {name} must be an array, got {type_name(value)}")
    if not value and not allow_empty:
        raise InvalidFieldError(f"{where}: {name} is empty")
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise InvalidFieldError(f"{where}: every {name} entry must be non-empty text")
    return tuple(value)


def as_whole_number(value: Any, name: str, where: str) -> int:
    """Narrow `value` to a non-negative `int` or raise `InvalidFieldError`. `bool` is refused."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InvalidFieldError(f"{where}: {name} must be a whole number, got {value!r}")
    return value


def type_name(value: Any) -> str:
    """What something is, for an error message a person has to act on."""
    return type(value).__name__
