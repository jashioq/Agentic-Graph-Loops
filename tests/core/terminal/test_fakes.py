"""The headless terminal is what workflow tests will drive, so it has its own."""

import pytest

from agl.core.terminal import (
    Answer,
    Option,
    Question,
    Row,
    Rows,
    Screen,
    Spacer,
    Terminal,
    Text,
    Timer,
)
from tests.fakes import HeadlessTerminal

QUESTION = Question("T-04", "Which storage layer?", (Option("DataStore", "async"),))


def build_screen() -> Screen:
    return Screen(
        header=Row(Text("add-auth-ticket-18732")),
        content=Rows(Row(Text("T-01"), Spacer(2), Text("merged"))),
        footer=Row(Text("elapsed"), Spacer(1), Timer(since=0.0)),
    )


def test_headless_terminal_is_a_terminal() -> None:
    assert isinstance(HeadlessTerminal(), Terminal)


async def test_it_captures_a_frame_on_entry() -> None:
    terminal = HeadlessTerminal()
    async with terminal.live(build_screen):
        pass
    assert terminal.frames
    assert "add-auth-ticket-18732" in terminal.frames[0]
    assert "T-01  merged" in terminal.frames[0]


async def test_frames_render_timers_against_the_injected_clock() -> None:
    terminal = HeadlessTerminal(clock=lambda: 61.0)
    async with terminal.live(build_screen):
        pass
    assert "elapsed 1:01" in terminal.frames[0]


async def test_frames_are_repeatable_with_a_fixed_clock() -> None:
    terminal = HeadlessTerminal(clock=lambda: 5.0)
    async with terminal.live(build_screen) as session:
        session.frame()
    assert terminal.frames[0] == terminal.frames[-1]


async def test_frame_returns_what_it_captured() -> None:
    terminal = HeadlessTerminal(clock=lambda: 0.0)
    async with terminal.live(build_screen) as session:
        assert "T-01  merged" in session.frame()


async def test_it_answers_from_the_scripted_queue_in_order() -> None:
    terminal = HeadlessTerminal(
        answers=[Answer("DataStore", was_free_text=False), Answer("Room", was_free_text=True)]
    )
    async with terminal.live(build_screen) as session:
        assert await session.ask(QUESTION) == Answer("DataStore", was_free_text=False)
        assert await session.ask(QUESTION) == Answer("Room", was_free_text=True)


async def test_it_records_the_questions_it_was_asked() -> None:
    terminal = HeadlessTerminal(answers=[Answer("DataStore", was_free_text=False)])
    async with terminal.live(build_screen) as session:
        await session.ask(QUESTION)
    assert terminal.questions == [QUESTION]


async def test_it_captures_the_frame_a_question_interrupted() -> None:
    terminal = HeadlessTerminal(answers=[Answer("DataStore", was_free_text=False)])
    async with terminal.live(build_screen) as session:
        captured = len(terminal.frames)
        await session.ask(QUESTION)
    assert len(terminal.frames) > captured


async def test_it_raises_when_the_scripted_answers_run_out() -> None:
    terminal = HeadlessTerminal()
    async with terminal.live(build_screen) as session:
        with pytest.raises(AssertionError, match="Which storage layer"):
            await session.ask(QUESTION)


async def test_it_records_every_frame_plus_one_on_entry_and_exit() -> None:
    terminal = HeadlessTerminal(clock=lambda: 0.0)
    async with terminal.live(build_screen) as session:
        session.frame()
        session.frame()
    assert len(terminal.frames) == 4


async def test_show_swaps_the_screen_and_records_the_new_frame() -> None:
    terminal = HeadlessTerminal(clock=lambda: 0.0)
    async with terminal.live(build_screen) as session:
        session.show(lambda: Screen(Rows(Row(Text("approval")))))

    assert "T-01  merged" in terminal.frames[0]
    assert terminal.frames[1].strip() == "approval"


async def test_frames_after_show_come_from_the_screen_it_installed() -> None:
    terminal = HeadlessTerminal(clock=lambda: 0.0)
    async with terminal.live(build_screen) as session:
        session.show(lambda: Screen(Rows(Row(Text("approval")))))
        assert session.frame().strip() == "approval"

    assert terminal.frames[-1].strip() == "approval"


async def test_a_session_can_start_blank_and_be_given_a_screen_later() -> None:
    terminal = HeadlessTerminal(clock=lambda: 0.0)
    async with terminal.live() as session:
        assert terminal.frames[0].strip() == ""
        session.show(build_screen)

    assert "T-01  merged" in terminal.frames[1]


async def test_frames_track_changing_state() -> None:
    status = "pending"

    def build() -> Screen:
        return Screen(Rows(Row(Text(status))))

    terminal = HeadlessTerminal(clock=lambda: 0.0)
    async with terminal.live(build) as session:
        status = "in progress"
        session.frame()

    assert terminal.frames[0].strip() == "pending"
    assert terminal.frames[1].strip() == "in progress"
