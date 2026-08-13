"""The Rich-backed terminal: a live display plus a question flow.

Layer: core (impl). This is where the I/O lives. Nothing pushes updates — a
repaint task calls `build()` at `1/fps` and renders whatever it returns, so a
`Timer` ticks because the frame is rebuilt, not because anything told it to.

A question takes the whole screen: the live display stops, the question is
printed, stdin is read on a worker thread so the event loop keeps running, and
the display is restored. A lock serializes concurrent askers into a FIFO queue.

When the console cannot animate — piped output, a dumb terminal — there is no
display to update, so frames are printed as they change instead.

A `build()` that raises becomes an error frame rather than a dead repaint task:
the display keeps moving, the failure is on screen, and the next tick retries.

Nothing about reading an answer is allowed to fail a run. Blank input and a
number outside the offered range are re-prompted; anything that is not a plain
decimal number is taken as the user's own words. The one thing that cannot be
recovered from is stdin having nothing left in it, and that raises
`TerminalError` — the module's only exception, and not the `EOFError` that used
to escape from here undocumented.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from rich.console import Console
from rich.live import Live
from rich.text import Text as RichText

from agl.core.terminal.api import (
    Answer,
    Color,
    LiveSession,
    Question,
    Row,
    Rows,
    Screen,
    Terminal,
    TerminalError,
    Text,
)
from agl.core.terminal.impl.render import to_renderable
from agl.core.terminal.impl.screen import to_layout

__all__ = ["RichTerminal"]

_OTHER_LABEL = "Other… (type your answer)"


def _error_screen(error: Exception) -> Screen:
    """The frame shown in place of one `build()` failed to produce."""
    message = Text(f"frame failed: {type(error).__name__}: {error}", Color.BOLD_RED)
    return Screen(Rows(Row(message)))


class RichTerminal(Terminal):
    """Paints screens with Rich and reads answers from stdin."""

    def __init__(self, console: Console | None = None, stdin: Iterator[str] | None = None) -> None:
        """`stdin` overrides `input()` with a scripted iterator, for tests."""
        self._console = console if console is not None else Console()
        self._stdin = stdin

    def live(
        self, build: Callable[[], Screen], fps: int = 4
    ) -> AbstractAsyncContextManager[LiveSession]:
        """Repaint `build()` at `fps` for as long as the context is open.

        Checked here rather than inside the context manager's body, so a caller
        that passes nonsense hears about it at the call and not on entry.
        """
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        return self._live(build, fps)

    @asynccontextmanager
    async def _live(
        self, build: Callable[[], Screen], fps: int
    ) -> AsyncIterator[LiveSession]:
        # Not transient: the last frame of a run is worth keeping on screen.
        display = Live(console=self._console, auto_refresh=False, transient=False)
        session = _RichLiveSession(display, self._console, self._read_line)
        display.start(refresh=False)
        session.paint(build)
        repaint = asyncio.create_task(self._repaint(session, build, fps))
        try:
            yield session
        finally:
            repaint.cancel()
            try:
                await repaint
            except asyncio.CancelledError:
                pass
            display.stop()

    async def _repaint(
        self, session: "_RichLiveSession", build: Callable[[], Screen], fps: int
    ) -> None:
        """Tick until cancelled.

        `paint` handles a failing frame, so this task ends only by cancellation
        and the `await repaint` on exit cannot mask an error from the body.
        """
        interval = 1 / fps
        while True:
            await asyncio.sleep(interval)
            session.paint(build)

    def _read_line(self) -> str:
        """Blocking read. Always called on a worker thread."""
        if self._stdin is None:
            return input()
        try:
            return next(self._stdin)
        except StopIteration:
            raise EOFError("no more scripted input") from None


class _RichLiveSession(LiveSession):
    """A running live display, and the questions that interrupt it."""

    def __init__(self, display: Live, console: Console, read_line: Callable[[], str]) -> None:
        self._display = display
        self._console = console
        self._read_line = read_line
        self._lock = asyncio.Lock()
        self._suspended = False
        self._last_frame = ""

    def paint(self, build: Callable[[], Screen]) -> None:
        """Render one frame, unless a question currently owns the screen.

        A `build()` that raises is caught here, not left to kill the repaint
        task: the failure is shown as a frame and the next tick tries again.
        """
        if self._suspended:
            return
        try:
            screen = build()
        except Exception as error:
            screen = _error_screen(error)
        now = time.monotonic()
        if self._can_animate():
            self._display.update(to_layout(screen, now), refresh=True)
        else:
            self._print_frame(screen, now)

    def _can_animate(self) -> bool:
        """Whether Rich will actually redraw in place. Matches `Live`'s own test."""
        return self._console.is_terminal and not self._console.is_dumb_terminal

    def _print_frame(self, screen: Screen, now: float) -> None:
        """Log the frame, skipping repeats so a static screen prints once."""
        with self._console.capture() as capture:
            for region in (screen.header, screen.content, screen.footer):
                if region is not None:
                    self._console.print(to_renderable(region, now))
        frame = capture.get()
        if frame == self._last_frame:
            return
        self._last_frame = frame
        # Written straight out: re-printing it would re-render text Rich has
        # already rendered.
        self._console.file.write(frame)
        self._console.file.flush()

    async def ask(self, question: Question) -> Answer:
        """Take over the screen. Concurrent callers queue behind the lock, FIFO."""
        async with self._lock:
            self._suspended = True
            self._display.stop()
            self._console.clear()
            try:
                self._print(question)
                return await self._read_answer(question)
            finally:
                self._console.clear()
                # The question took the screen with it: the next frame is new
                # again, however much it looks like the one before the question.
                self._last_frame = ""
                self._display.start(refresh=False)
                self._suspended = False

    def _print(self, question: Question) -> None:
        self._console.print(RichText(question.header, style="bold"))
        self._console.print(RichText(question.title))
        self._console.print()
        for number, option in enumerate(question.options, start=1):
            entry = RichText(f"  {number}. ")
            entry.append(option.label, style="bold")
            entry.append(f" — {option.description}", style="grey58")
            self._console.print(entry)
        self._console.print(RichText(f"  {self._other_number(question)}. {_OTHER_LABEL}"))
        self._console.print()

    async def _read_answer(self, question: Question) -> Answer:
        """Loop until the input means something.

        Blank input re-prompts, and so does a number outside the range on
        screen: with three options, `9` is a slip, and returning `Answer("9")`
        would hand an agent a digit as though the user had meant to say it.

        `isdecimal` rather than `isdigit`, because `isdigit` says yes to
        superscripts and other numerics that `int` then refuses — `"²"` used to
        get a `ValueError` out of here. Anything that is not a plain decimal
        number is what the user typed, and comes straight back.
        """
        while True:
            reply = await self._read()
            if not reply:
                continue
            if not reply.isdecimal():
                return Answer(reply, was_free_text=True)

            number = int(reply)
            if 1 <= number <= len(question.options):
                return Answer(question.options[number - 1].label, was_free_text=False)
            if number == self._other_number(question):
                return Answer(await self._read_free_text(), was_free_text=True)
            self._console.print(
                RichText(f"Please enter a number between 1 and {self._other_number(question)}.")
            )

    async def _read_free_text(self) -> str:
        self._console.print(RichText("Your answer:"))
        while True:
            reply = await self._read()
            if reply:
                return reply

    async def _read(self) -> str:
        """Read one line off the event loop, so the rest of the run keeps going.

        Raises `TerminalError` when there is nothing left to read. Both sources
        report that as `EOFError` — `input()` on a closed stdin, and the
        scripted iterator when it runs out — and it is the one thing here a
        re-prompt cannot fix.
        """
        loop = asyncio.get_running_loop()
        try:
            line = await loop.run_in_executor(None, self._read_line)
        except EOFError as error:
            raise TerminalError(
                "cannot read an answer: standard input is exhausted or closed"
            ) from error
        return line.strip()

    @staticmethod
    def _other_number(question: Question) -> int:
        return len(question.options) + 1
