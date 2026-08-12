"""The live loop and the question flow.

Stdin is a scripted iterator handed to the constructor — nothing is patched.
The console writes to a buffer, so a real `Live` runs without a real terminal.
"""

import asyncio
import io

import pytest
from rich.console import Console

from agl.core.terminal import Answer, Option, Question, Row, Rows, Screen, Text
from agl.core.terminal.impl.rich_terminal import RichTerminal

QUESTION = Question(
    header="T-04  login screen",
    title="Which storage layer should the token cache use?",
    options=(
        Option("DataStore", "Survives process death, async API"),
        Option("In-memory only", "Simplest, lost on restart"),
        Option("EncryptedSharedPreferences", "Synchronous, encrypted at rest"),
    ),
)


def make_terminal(*lines: str) -> tuple[RichTerminal, io.StringIO]:
    buffer = io.StringIO()
    console = Console(file=buffer, width=80, no_color=True)
    return RichTerminal(console=console, stdin=iter(lines)), buffer


def build_screen() -> Screen:
    return Screen(header=None, content=Rows(Row(Text("dashboard"))), footer=None)


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


async def test_an_out_of_range_number_is_taken_as_free_text() -> None:
    assert await ask("9") == Answer("9", was_free_text=True)


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
    terminal, buffer = make_terminal("1", "2")

    async with terminal.live(build_screen) as session:
        earlier = asyncio.create_task(session.ask(first))
        await asyncio.sleep(0)
        later = asyncio.create_task(session.ask(second))
        answers = await asyncio.gather(earlier, later)

    assert answers == [Answer("A1", was_free_text=False), Answer("B2", was_free_text=False)]
    output = buffer.getvalue()
    assert output.index("first?") < output.index("second?")


async def test_only_one_question_is_on_screen_at_a_time() -> None:
    first = Question("first", "first?", (Option("A1", "d"),))
    second = Question("second", "second?", (Option("B1", "d"),))
    terminal, buffer = make_terminal("1", "1")

    async with terminal.live(build_screen) as session:
        earlier = asyncio.create_task(session.ask(first))
        await asyncio.sleep(0)
        later = asyncio.create_task(session.ask(second))
        await asyncio.sleep(0)
        # The first question owns the screen; the queued one has not printed.
        assert "first?" in buffer.getvalue()
        assert "second?" not in buffer.getvalue()
        await asyncio.gather(earlier, later)


async def test_frames_reach_a_console_that_cannot_animate() -> None:
    # A piped stdout, or TERM=dumb: Rich's Live draws nothing at all there, so
    # the frames have to be printed instead of silently dropped.
    terminal, buffer = make_terminal()
    async with terminal.live(build_screen):
        pass
    assert "dashboard" in buffer.getvalue()


async def test_an_unchanged_frame_is_not_reprinted() -> None:
    terminal, buffer = make_terminal()
    async with terminal.live(build_screen, fps=100):
        await asyncio.sleep(0.1)
    assert buffer.getvalue().count("dashboard") == 1


async def test_a_changed_frame_is_printed_again() -> None:
    status = "pending"

    def build() -> Screen:
        return Screen(header=None, content=Rows(Row(Text(status))), footer=None)

    terminal, buffer = make_terminal()
    async with terminal.live(build, fps=100):
        status = "merged"
        await asyncio.sleep(0.1)

    output = buffer.getvalue()
    assert "pending" in output
    assert "merged" in output


async def test_live_paints_a_frame_on_entry() -> None:
    calls = 0

    def build() -> Screen:
        nonlocal calls
        calls += 1
        return build_screen()

    terminal, _ = make_terminal()
    async with terminal.live(build):
        assert calls == 1


async def test_live_repaints_at_the_requested_rate() -> None:
    calls = 0

    def build() -> Screen:
        nonlocal calls
        calls += 1
        return build_screen()

    terminal, _ = make_terminal()
    async with terminal.live(build, fps=100):
        await asyncio.sleep(0.1)
        assert calls > 1


async def test_repainting_stops_when_the_context_exits() -> None:
    calls = 0

    def build() -> Screen:
        nonlocal calls
        calls += 1
        return build_screen()

    terminal, _ = make_terminal()
    async with terminal.live(build, fps=100):
        await asyncio.sleep(0.05)
    settled = calls
    await asyncio.sleep(0.05)
    assert calls == settled


async def test_the_dashboard_resumes_after_a_question() -> None:
    calls = 0

    def build() -> Screen:
        nonlocal calls
        calls += 1
        return build_screen()

    terminal, _ = make_terminal("1")
    async with terminal.live(build, fps=100) as session:
        await session.ask(QUESTION)
        resumed = calls
        await asyncio.sleep(0.1)
        assert calls > resumed


async def test_exhausted_scripted_input_raises() -> None:
    with pytest.raises(EOFError):
        await ask()
