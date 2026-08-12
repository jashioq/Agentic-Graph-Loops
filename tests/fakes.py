"""Fakes for core modules. Real classes inheriting the ABC, never mocks.

An incomplete fake fails at instantiation, which is the point: when an ABC grows
a method, every fake here stops working until it is implemented.
"""

import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager

from rich.console import Console

from agl.core.terminal import Answer, LiveSession, Question, Screen, Terminal
from agl.core.terminal.impl.render import to_renderable

__all__ = ["HeadlessTerminal"]


class HeadlessTerminal(Terminal):
    """A terminal that paints nothing and answers from a script.

    Every frame it would have rendered is recorded as plain text in `frames`,
    through the real renderer, so assertions see what a user would have seen.
    Pass `clock` to pin the time and make frames containing timers repeatable.
    """

    def __init__(
        self,
        answers: Iterable[Answer] = (),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.frames: list[str] = []
        self.questions: list[Question] = []
        self._answers = deque(answers)
        self._clock = clock

    @asynccontextmanager
    async def live(
        self, build: Callable[[], Screen], fps: int = 4
    ) -> AsyncIterator[LiveSession]:
        """Capture a frame on entry and on exit. No repaint task, no timing."""
        session = _HeadlessSession(self, build)
        session.frame()
        try:
            yield session
        finally:
            session.frame()

    def answer(self, question: Question) -> Answer:
        """Pop the next scripted answer, or fail loudly naming the question."""
        self.questions.append(question)
        assert self._answers, f"no scripted answer left for: {question.title}"
        return self._answers.popleft()

    def capture(self, screen: Screen) -> str:
        """Render `screen` to plain text and record it."""
        console = Console(width=80, record=True, no_color=True)
        now = self._clock()
        for region in (screen.header, screen.content, screen.footer):
            if region is not None:
                console.print(to_renderable(region, now))
        text = console.export_text()
        self.frames.append(text)
        return text


class _HeadlessSession(LiveSession):
    """Frames are captured on demand — a test decides when time moves."""

    def __init__(self, terminal: HeadlessTerminal, build: Callable[[], Screen]) -> None:
        self._terminal = terminal
        self._build = build

    def frame(self) -> str:
        """Rebuild the screen, record it, and hand the text back."""
        return self._terminal.capture(self._build())

    async def ask(self, question: Question) -> Answer:
        """Record the frame the question interrupted, then answer from the script."""
        self.frame()
        return self._terminal.answer(question)
