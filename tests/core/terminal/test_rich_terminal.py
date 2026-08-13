"""The live loop and the question flow.

Stdin is a scripted iterator handed to the constructor — nothing is patched.
The console writes to a buffer, so a real `Live` runs without a real terminal.

Nothing here asserts on wall clock. Tests that need a repaint to have happened
wait on an `asyncio.Event` the `build` callback sets, and tests that need an
`ask` to be parked mid-read gate stdin on a `threading.Event` (`GatedInput`) —
`_read_line` runs on a worker thread, so blocking it is exactly what a user who
has not typed yet does. `_TIMEOUT` only bounds a hang; it is never a delay a
passing test waits out.
"""

import asyncio
import io
import threading
from collections.abc import Iterator

import pytest
from rich.console import Console

from agl.core.terminal import (
    Answer,
    Option,
    Question,
    Row,
    Rows,
    Screen,
    TerminalError,
    Text,
)
from agl.core.terminal.impl.rich_terminal import RichTerminal

_TIMEOUT = 5.0

QUESTION = Question(
    header="T-04  login screen",
    title="Which storage layer should the token cache use?",
    options=(
        Option("DataStore", "Survives process death, async API"),
        Option("In-memory only", "Simplest, lost on restart"),
        Option("EncryptedSharedPreferences", "Synchronous, encrypted at rest"),
    ),
)


class GatedInput(Iterator[str]):
    """Scripted stdin that parks the reader until the test lets it through.

    Reads happen on a worker thread, so blocking one holds an `ask` exactly
    where a real user would hold it: printed, and waiting for a line.
    """

    def __init__(self, *lines: str) -> None:
        self._lines = iter(lines)
        self._reading = threading.Event()
        self._released = threading.Event()

    def __next__(self) -> str:
        self._reading.set()
        assert self._released.wait(_TIMEOUT), "input was never released"
        return next(self._lines)

    async def wait_until_read(self) -> None:
        """Return once a reader is parked on this input, off the event loop."""
        assert await asyncio.to_thread(self._reading.wait, _TIMEOUT), "nothing ever read"

    def release(self) -> None:
        self._released.set()


def make_terminal(*lines: str) -> tuple[RichTerminal, io.StringIO]:
    return make_terminal_from(iter(lines))


def make_terminal_from(stdin: Iterator[str]) -> tuple[RichTerminal, io.StringIO]:
    buffer = io.StringIO()
    console = Console(file=buffer, width=80, no_color=True)
    return RichTerminal(console=console, stdin=stdin), buffer


def build_screen() -> Screen:
    return Screen(Rows(Row(Text("dashboard"))))


async def drain_ready_tasks() -> None:
    """Let every task that can make progress right now run to its next await."""
    for _ in range(10):
        await asyncio.sleep(0)


async def ask(*lines: str, question: Question = QUESTION) -> Answer:
    terminal, _ = make_terminal(*lines)
    async with terminal.live(build_screen) as session:
        return await session.ask(question)


async def test_a_listed_number_returns_that_option() -> None:
    assert await ask("1") == Answer("DataStore", was_free_text=False)


async def test_the_last_listed_number_returns_that_option() -> None:
    assert await ask("3") == Answer("EncryptedSharedPreferences", was_free_text=False)


async def test_surrounding_whitespace_is_ignored() -> None:
    assert await ask("  2  ") == Answer("In-memory only", was_free_text=False)


async def test_an_out_of_range_number_reprompts_rather_than_becoming_an_answer() -> None:
    # With three options, `9` is a mistyped selection, not a considered reply.
    # Handing "9" to an agent as the user's own words is worse than asking again.
    assert await ask("9", "2") == Answer("In-memory only", was_free_text=False)


async def test_zero_reprompts_too() -> None:
    assert await ask("0", "1") == Answer("DataStore", was_free_text=False)


async def test_the_reprompt_says_what_would_be_accepted() -> None:
    terminal, buffer = make_terminal("9", "1")
    async with terminal.live(build_screen) as session:
        await session.ask(QUESTION)

    assert "between 1 and 4" in buffer.getvalue()


@pytest.mark.parametrize("reply", ["²", "³", "½", "Ⅳ"])
async def test_a_digit_character_int_cannot_parse_is_free_text_not_a_crash(
    reply: str,
) -> None:
    # `str.isdigit` says yes to superscripts and other numerics that `int`
    # refuses, so the check has to be `isdecimal`. Anything outside it is just
    # something the user typed.
    assert await ask(reply) == Answer(reply, was_free_text=True)


async def test_the_other_entry_reprompts_for_text() -> None:
    assert await ask("4", "use Room instead") == Answer("use Room instead", was_free_text=True)


async def test_the_other_entry_reprompts_until_the_text_is_not_empty() -> None:
    assert await ask("4", "   ", "use Room") == Answer("use Room", was_free_text=True)


async def test_bare_free_text_is_returned_directly() -> None:
    assert await ask("none of these") == Answer("none of these", was_free_text=True)


async def test_empty_input_reprompts() -> None:
    assert await ask("", "   ", "2") == Answer("In-memory only", was_free_text=False)


async def test_the_question_screen_lists_options_and_an_other_entry() -> None:
    terminal, buffer = make_terminal("1")
    async with terminal.live(build_screen) as session:
        await session.ask(QUESTION)
    output = buffer.getvalue()
    assert "T-04  login screen" in output
    assert "Which storage layer should the token cache use?" in output
    assert "1. DataStore" in output
    assert "Survives process death, async API" in output
    assert "3. EncryptedSharedPreferences" in output
    assert "4. Other" in output


async def test_a_question_with_no_options_offers_only_the_other_entry() -> None:
    question = Question(header="h", title="t", options=())
    assert await ask("free text", question=question) == Answer("free text", was_free_text=True)


async def test_concurrent_asks_resolve_in_fifo_order() -> None:
    first = Question("first", "first?", (Option("A1", "d"), Option("A2", "d")))
    second = Question("second", "second?", (Option("B1", "d"), Option("B2", "d")))
    stdin = GatedInput("1", "2")
    terminal, buffer = make_terminal_from(stdin)

    async with terminal.live(build_screen) as session:
        earlier = asyncio.create_task(session.ask(first))
        # The gate, not a yield count, is what puts `earlier` first in the queue.
        await stdin.wait_until_read()
        later = asyncio.create_task(session.ask(second))
        stdin.release()
        answers = await asyncio.gather(earlier, later)

    assert answers == [Answer("A1", was_free_text=False), Answer("B2", was_free_text=False)]
    output = buffer.getvalue()
    assert output.index("first?") < output.index("second?")


async def test_only_one_question_is_on_screen_at_a_time() -> None:
    first = Question("first", "first?", (Option("A1", "d"),))
    second = Question("second", "second?", (Option("B1", "d"),))
    stdin = GatedInput("1", "1")
    terminal, buffer = make_terminal_from(stdin)

    async with terminal.live(build_screen) as session:
        earlier = asyncio.create_task(session.ask(first))
        await stdin.wait_until_read()
        later = asyncio.create_task(session.ask(second))
        # The first ask cannot finish while the gate is shut, so the queued one
        # gets every chance to run: it stays off screen because of the lock.
        await drain_ready_tasks()
        assert "first?" in buffer.getvalue()
        assert "second?" not in buffer.getvalue()
        stdin.release()
        await asyncio.gather(earlier, later)


async def test_frames_reach_a_console_that_cannot_animate() -> None:
    # A piped stdout, or TERM=dumb: Rich's Live draws nothing at all there, so
    # the frames have to be printed instead of silently dropped.
    terminal, buffer = make_terminal()
    async with terminal.live(build_screen):
        pass
    assert "dashboard" in buffer.getvalue()


async def test_an_unchanged_frame_is_not_reprinted() -> None:
    repainted = asyncio.Event()
    calls = 0

    def build() -> Screen:
        nonlocal calls
        calls += 1
        if calls > 1:
            repainted.set()
        return build_screen()

    terminal, buffer = make_terminal()
    async with terminal.live(build, fps=100):
        # Wait for repaints to have happened, rather than for a wall-clock
        # duration in which they may or may not have.
        await asyncio.wait_for(repainted.wait(), _TIMEOUT)
    assert buffer.getvalue().count("dashboard") == 1


async def test_a_question_makes_the_next_frame_print_again() -> None:
    # The question wiped the screen, so the frame that follows it is new again
    # even though it is identical to the one before: dedup must not span a
    # question, or a piped run shows nothing but the question afterwards.
    resumed = asyncio.Event()
    answered = False

    def build() -> Screen:
        if answered:
            resumed.set()
        return build_screen()

    terminal, buffer = make_terminal("1")
    async with terminal.live(build, fps=100) as session:
        await session.ask(QUESTION)
        answered = True
        await asyncio.wait_for(resumed.wait(), _TIMEOUT)

    assert buffer.getvalue().count("dashboard") == 2


async def test_a_changed_frame_is_printed_again() -> None:
    status = "pending"
    rebuilt = asyncio.Event()

    def build() -> Screen:
        if status == "merged":
            # Set before the frame is written, but a waiter cannot resume until
            # `paint` returns: nothing between here and the write awaits.
            rebuilt.set()
        return Screen(Rows(Row(Text(status))))

    terminal, buffer = make_terminal()
    async with terminal.live(build, fps=100):
        status = "merged"
        await asyncio.wait_for(rebuilt.wait(), _TIMEOUT)

    output = buffer.getvalue()
    assert "pending" in output
    assert "merged" in output


async def test_a_failing_build_is_rendered_as_an_error_and_repainting_continues() -> None:
    survived = asyncio.Event()
    calls = 0

    def build() -> Screen:
        nonlocal calls
        calls += 1
        if calls == 1:
            return build_screen()
        if calls >= 3:
            # Reached only if the repaint task outlived the first failure.
            survived.set()
        raise RuntimeError("build blew up")

    terminal, buffer = make_terminal()
    async with terminal.live(build, fps=100):
        await asyncio.wait_for(survived.wait(), _TIMEOUT)

    output = buffer.getvalue()
    assert "dashboard" in output
    assert "build blew up" in output


async def test_a_failing_build_does_not_mask_an_error_from_the_body() -> None:
    failed = asyncio.Event()

    def build() -> Screen:
        failed.set()
        raise RuntimeError("build blew up")

    terminal, _ = make_terminal()
    with pytest.raises(ValueError, match="from the body"):
        async with terminal.live(build, fps=100):
            await asyncio.wait_for(failed.wait(), _TIMEOUT)
            raise ValueError("from the body")


async def test_live_paints_a_frame_on_entry() -> None:
    calls = 0

    def build() -> Screen:
        nonlocal calls
        calls += 1
        return build_screen()

    terminal, _ = make_terminal()
    async with terminal.live(build):
        assert calls == 1


async def test_live_keeps_repainting_after_the_frame_it_paints_on_entry() -> None:
    repainted = asyncio.Event()
    calls = 0

    def build() -> Screen:
        nonlocal calls
        calls += 1
        if calls > 1:
            repainted.set()
        return build_screen()

    terminal, _ = make_terminal()
    async with terminal.live(build, fps=100):
        await asyncio.wait_for(repainted.wait(), _TIMEOUT)
    assert calls > 1


async def test_repainting_stops_when_the_context_exits() -> None:
    repainted = asyncio.Event()
    calls = 0

    def build() -> Screen:
        nonlocal calls
        calls += 1
        if calls > 1:
            repainted.set()
        return build_screen()

    terminal, _ = make_terminal()
    async with terminal.live(build, fps=100):
        await asyncio.wait_for(repainted.wait(), _TIMEOUT)
    settled = calls
    # A live repaint task at fps=100 would tick ~10 times in this window. The
    # sleep can only hide a failure, never invent one: nothing but a surviving
    # repaint task can call `build` again.
    await asyncio.sleep(0.1)
    assert calls == settled


async def test_the_dashboard_resumes_after_a_question() -> None:
    resumed = asyncio.Event()
    answered = False
    calls = 0

    def build() -> Screen:
        nonlocal calls
        calls += 1
        if answered:
            resumed.set()
        return build_screen()

    terminal, _ = make_terminal("1")
    async with terminal.live(build, fps=100) as session:
        await session.ask(QUESTION)
        answered = True
        before = calls
        await asyncio.wait_for(resumed.wait(), _TIMEOUT)
        assert calls > before


async def test_exhausted_stdin_raises_the_modules_own_error() -> None:
    # A run on piped stdin dies at its first question. That it dies is the
    # contract; what it must not do is die with an exception the ABC never
    # mentioned, from a module that had no typed error at all.
    with pytest.raises(TerminalError):
        await ask()


async def test_exhausted_stdin_during_a_free_text_reprompt_raises_too() -> None:
    with pytest.raises(TerminalError):
        await ask("4")


async def test_the_error_is_not_an_eof_error_escaping_by_another_name() -> None:
    assert not issubclass(TerminalError, EOFError)


@pytest.mark.parametrize("fps", [0, -1])
async def test_a_non_positive_fps_is_refused_rather_than_dividing_by_zero(
    fps: int,
) -> None:
    terminal, _ = make_terminal()
    with pytest.raises(ValueError, match="fps"):
        terminal.live(build_screen, fps=fps)


async def test_a_screen_with_only_content_paints() -> None:
    terminal, buffer = make_terminal()

    async with terminal.live(lambda: Screen(Rows(Row(Text("bare"))))):
        pass

    assert "bare" in buffer.getvalue()
