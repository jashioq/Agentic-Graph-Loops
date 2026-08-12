"""Pure rendering of components to Rich renderables.

Layer: core (impl). No I/O, no console, no state — the only inputs are a
component tree and a `time.monotonic()` reading, so a frame is reproducible.
This is also the one place where a `Color` becomes a Rich style string.
"""

from typing import Final, assert_never

from rich.console import RenderableType
from rich.text import Text as RichText

from agl.core.terminal.api import Color, Component, Row, Rows, Spacer, Text, Timer

__all__ = ["line_count", "to_renderable"]

_SECONDS_PER_MINUTE: Final = 60
_SECONDS_PER_HOUR: Final = 3600

_STYLES: Final[dict[Color, str]] = {
    Color.WHITE: "white",
    Color.GREY: "grey58",
    Color.RED: "red",
    Color.GREEN: "green",
    Color.BLUE: "blue",
    Color.YELLOW: "yellow",
    Color.MAGENTA: "magenta",
    Color.CYAN: "cyan",
    Color.BOLD_YELLOW: "bold yellow",
    Color.BOLD_RED: "bold red",
    Color.DIM_GREEN: "dim green",
    Color.DIM_GREY: "dim grey58",
}


def to_renderable(component: Component, now: float) -> RenderableType:
    """Render `component` as of `now`. Pure: same inputs, same frame."""
    text = _to_text(component, now)
    if "\n" not in text.plain:
        text.rstrip()  # a single-line frame is itself the line
    return text


def line_count(component: Component, now: float) -> int:
    """How many terminal lines `component` occupies. Used to size regions."""
    return _to_text(component, now).plain.count("\n") + 1


def _to_text(component: Component, now: float) -> RichText:
    match component:
        case Text():
            return RichText(component.content, style=_STYLES[component.color])
        case Timer():
            return RichText(
                _format_elapsed(now - component.since), style=_STYLES[component.color]
            )
        case Spacer():
            return RichText(" " * max(0, component.width))
        case Row():
            return _row_to_text(component, now)
        case Rows():
            return RichText("\n").join(_line_to_text(row, now) for row in component.children)
        case _:  # pragma: no cover - exhaustive over Component
            assert_never(component)


def _row_to_text(row: Row, now: float) -> RichText:
    """Children left to right, no implicit separator. Gaps come from `Spacer`.

    Nothing is stripped here: a `Row` nested inside another `Row` sits mid-line,
    where a trailing `Spacer` is a real gap rather than trailing whitespace.
    """
    text = RichText()
    for child in row.children:
        text.append_text(_to_text(child, now))
    return text


def _line_to_text(component: Component, now: float) -> RichText:
    """Render a component that ends a line, so trailing whitespace is dropped."""
    text = _to_text(component, now)
    text.rstrip()
    return text


def _format_elapsed(seconds: float) -> str:
    """`M:SS` under an hour, `H:MM:SS` at or above it. Never negative."""
    total = max(0, int(seconds))
    minutes, secs = divmod(total, _SECONDS_PER_MINUTE)
    if total < _SECONDS_PER_HOUR:
        return f"{minutes}:{secs:02d}"
    hours, minutes = divmod(minutes, _SECONDS_PER_MINUTE)
    return f"{hours}:{minutes:02d}:{secs:02d}"
