"""Terminal API: immutable screen descriptions and the abstract terminal.

Layer: core. Everything a workflow needs to drive the terminal lives here.
A screen is data — a tree of components — and rendering is a pure function of
that tree plus the current time, so timers tick without anything pushing
updates. This module holds no I/O and knows nothing about Rich.
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
    "Text",
    "Timer",
]


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
    """Three sticky regions: header on top, footer on the bottom, content between."""

    header: Row | Rows | None
    content: Rows
    footer: Row | Rows | None


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
        """Repaint `build()` at `fps` until the context exits."""


class LiveSession(ABC):
    """A running live display. Questions interrupt it and then hand it back."""

    @abstractmethod
    async def ask(self, question: Question) -> Answer:
        """Take over the screen to ask. Concurrent callers queue FIFO."""
