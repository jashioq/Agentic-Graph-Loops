"""Reading typed fields out of JSON that may be wrong.

Layer: runtime. A pure leaf: imports nothing else in AGL. What it narrows is
agent output or a document a person hand-edited, so every failure is a message
someone has to act on. Each function takes the callable to hand that message to,
so what a failure *means* stays with the caller: this module reads a field and
either returns it or reports it, and never decides what to raise.

`OnError` returns `NoReturn` because a narrowing that failed has no value to give
back. A caller is free to log, wrap or count on the way, but it ends by raising
its own error type.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, NoReturn

__all__ = [
    "OnError",
    "as_object",
    "as_optional_text",
    "as_text",
    "as_text_list",
    "as_whole_number",
    "reject_unknown_fields",
    "require_fields",
    "type_name",
]

type OnError = Callable[[str], NoReturn]
"""What a caller is handed a failure through: the message, and no way back."""


def as_object(value: Any, where: str, on_error: OnError) -> dict[str, Any]:
    """Narrow `value` to a JSON object, or hand `on_error` why it is not one."""
    if not isinstance(value, dict):
        on_error(f"{where} must be an object, got {type_name(value)}")
    return value


def require_fields(
    fields: Mapping[str, Any], required: Sequence[str], where: str, on_error: OnError
) -> None:
    """Report the first name in `required` that `fields` does not have."""
    for name in required:
        if name not in fields:
            on_error(f"{where}: missing required field {name!r}")


def reject_unknown_fields(
    fields: Mapping[str, Any], allowed: Sequence[str], where: str, on_error: OnError
) -> None:
    """Report a field outside `allowed`, rather than ignoring it.

    A misspelled key that is quietly dropped is a person's edit that appeared to
    take and did not, which is worse than being told.
    """
    for name in fields:
        if name not in allowed:
            on_error(f"{where}: unknown field {name!r}")


def as_text(value: Any, name: str, where: str, on_error: OnError) -> str:
    """Narrow `value` to non-empty text, or hand `on_error` why it is not."""
    if not isinstance(value, str) or not value.strip():
        on_error(f"{where}: {name} must be non-empty text, got {value!r}")
    return value


def as_optional_text(value: Any, name: str, where: str, on_error: OnError) -> str | None:
    """`as_text`, except that `None` is an answer rather than a failure."""
    if value is None:
        return None
    return as_text(value, name, where, on_error)


def as_text_list(
    value: Any, name: str, where: str, on_error: OnError, *, allow_empty: bool = True
) -> tuple[str, ...]:
    """Narrow `value` to a tuple of non-empty strings, or report why it is not.

    `allow_empty=False` also refuses an array with nothing in it, for a field
    whose whole point is to list something.
    """
    if not isinstance(value, list):
        on_error(f"{where}: {name} must be an array, got {type_name(value)}")
    if not value and not allow_empty:
        on_error(f"{where}: {name} is empty")
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            on_error(f"{where}: every {name} entry must be non-empty text")
    return tuple(value)


def as_whole_number(value: Any, name: str, where: str, on_error: OnError) -> int:
    """Narrow `value` to a non-negative `int`, or report why it is not. `bool` is refused."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        on_error(f"{where}: {name} must be a whole number, got {value!r}")
    return value


def type_name(value: Any) -> str:
    """What something is, for an error message a person has to act on."""
    return type(value).__name__
