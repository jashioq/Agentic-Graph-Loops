"""Terminal API: immutable screen descriptions and the abstract terminal.

Layer: core. A screen is data — a tree of components — and rendering is a pure
function of that tree plus the current time, so timers tick with nothing pushing
updates. No I/O here, and nothing about Rich.

A terminal must not be the reason a run fails: bad input is re-prompted, and
`TerminalError` is left for the one case with no answer to be had.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "Answer",
    "Color",
    "Component",
    "LiveSession",
    "Option",
    "Question",
    "Row",
    "Rows",
    "Screen",
    "Spacer",
    "Terminal",
    "TerminalError",
    "Text",
    "Timer",
]


class TerminalError(Exception):
    """Raised when the terminal cannot do what was asked: stdin exhausted or closed."""


class Color(Enum):
    """The palette a workflow may paint with. Mapped to styles in the impl."""

    WHITE = "white"
    GREY = "grey"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    MAGENTA = "magenta"
    CYAN = "cyan"
    BOLD_YELLOW = "bold_yellow"
    BOLD_RED = "bold_red"
    DIM_GREEN = "dim_green"
    DIM_GREY = "dim_grey"
    DIM_RED = "dim_red"


@dataclass(frozen=True)
class Text:
    """A run of literal text in one color."""

    content: str
    color: Color = Color.WHITE


@dataclass(frozen=True)
class Timer:
    """Elapsed time since a `time.monotonic()` reading, formatted on render."""

    since: float
    color: Color = Color.GREY


@dataclass(frozen=True)
class Spacer:
    """A horizontal gap. The only source of separation inside a `Row`."""

    width: int


@dataclass(frozen=True, init=False)
class Row:
    """Children laid out left to right with no implicit separator."""

    children: tuple["Component", ...]

    def __init__(self, *children: "Component") -> None:
        object.__setattr__(self, "children", children)


@dataclass(frozen=True, init=False)
class Rows:
    """Rows stacked top to bottom, one per line."""

    children: tuple[Row, ...]

    def __init__(self, *children: Row) -> None:
        object.__setattr__(self, "children", children)


type Component = Text | Timer | Spacer | Row | Rows


@dataclass(frozen=True)
class Screen:
    """Three sticky regions: header on top, footer below, content between.

    Content is first because it is the only one a screen must have.
    """

    content: Rows
    header: Row | Rows | None = None
    footer: Row | Rows | None = None


@dataclass(frozen=True)
class Option:
    """One offered answer to a `Question`."""

    label: str
    description: str


@dataclass(frozen=True)
class Question:
    """A question put to the user, taking over the screen until answered."""

    header: str
    title: str
    options: tuple[Option, ...]


@dataclass(frozen=True)
class Answer:
    """The chosen option's label, or whatever free text the user typed."""

    text: str
    was_free_text: bool


class Terminal(ABC):
    """Owns the screen. Workflows describe frames; the terminal paints them."""

    @abstractmethod
    def live(
        self, build: Callable[[], Screen] | None = None, fps: int = 4
    ) -> AbstractAsyncContextManager["LiveSession"]:
        """Repaints `build()` at `fps` until the context exits.

        param: build - `None` leaves the screen blank until the first `show`
        param: fps - frames per second; non-positive raises `ValueError`
        """


class LiveSession(ABC):
    """A running live display. Questions interrupt it and then hand it back."""

    @abstractmethod
    def show(self, build: Callable[[], Screen]) -> None:
        """Replaces the frame source and paints it once, so a stage change shows at once."""
# TODO ask should be a runtime module or workflow defined. Core module should be only for displaying stuff. If a different screen needs to be displayed (for question) then that should be requested by runtime or workflow
    @abstractmethod
    async def ask(self, question: Question) -> Answer:
        """Takes over the screen to ask. Concurrent callers queue FIFO.

        return: Answer - a chosen label, or free text for anything that is not a
            plain decimal; meaningless input is re-prompted, and stdin with no
            answer to give raises `TerminalError`
        """
