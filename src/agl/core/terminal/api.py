"""Terminal API: immutable screen descriptions and the abstract terminal.

Layer: core. Everything a workflow needs to drive the terminal lives here.
A screen is data — a tree of components — and rendering is a pure function of
that tree plus the current time, so timers tick without anything pushing
updates. This module holds no I/O and knows nothing about Rich.

`TerminalError` is the module's one failure. A terminal is never the interesting
part of a run, so the rule is that it must not be the reason one fails: bad
input is re-prompted, not raised on. What is left is the case where there is no
answer to be had at all — stdin exhausted or closed — and that is what the error
is for.
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
    """Raised when the terminal cannot do what was asked of it.

    In practice that is one thing: a question with nobody able to answer it,
    because stdin is exhausted or closed. Deliberately not an `EOFError` —
    a caller catching this is catching a terminal failure, not adopting a
    built-in whose meaning is wider.
    """


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
    """Three sticky regions: header on top, footer on the bottom, content between.

    Content comes first because it is the only one a screen must have; most
    frames are content alone, and saying so twice over at every call site added
    nothing.
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
        self, build: Callable[[], Screen], fps: int = 4
    ) -> AbstractAsyncContextManager["LiveSession"]:
        """Repaint `build()` at `fps` until the context exits.

        `fps` is frames per second and must be positive; a non-positive value
        raises `ValueError` here rather than becoming a division inside the
        repaint loop.
        """


class LiveSession(ABC):
    """A running live display. Questions interrupt it and then hand it back."""

    @abstractmethod
    async def ask(self, question: Question) -> Answer:
        """Take over the screen to ask. Concurrent callers queue FIFO.

        Input that means nothing — blank, or a number outside the offered range
        — is re-prompted rather than raised on or taken at face value. Anything
        that is not a plain decimal number is the user's own words and comes
        back as free text.

        Raises `TerminalError` when there is no answer to be had: stdin
        exhausted or closed, which is what a run on piped input meets at its
        first question.
        """
