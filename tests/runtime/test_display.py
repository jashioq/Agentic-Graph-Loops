"""The board of display-only state, and the one path to the terminal.

`HeadlessTerminal` throughout: it records every frame it would have painted
through the real renderer, so an assertion here is an assertion about what a
user would have seen.

`Board` is tested with `now` passed explicitly wherever a clock is involved —
the same reason `render` takes `now` as a parameter, so a stamp is a value a
test can name rather than whatever the machine thought the time was.
"""

from agl.core.agent import AgentOption, AgentQuestion
from agl.core.terminal import Answer, Option, Question, Row, Rows, Screen, Text
from agl.runtime.display import Board, Display, live
from tests.fakes import HeadlessTerminal


def screen(text: str) -> Screen:
    return Screen(Rows(Row(Text(text))))


# -- Board ------------------------------------------------------------------


def test_a_fresh_board_holds_only_its_start() -> None:
    board = Board(started_at=12.0)

    assert board.started_at == 12.0
    assert board.marks == {}
    assert board.status_since == {}
    assert board.activity == {}


def test_a_mark_is_readable_by_name() -> None:
    board = Board(started_at=0.0)

    board.mark("approved", now=7.5)

    assert board.since("approved") == 7.5


def test_an_unset_mark_reads_as_none() -> None:
    assert Board(started_at=0.0).since("approved") is None


def test_marks_are_independent_of_each_other() -> None:
    board = Board(started_at=0.0)

    board.mark("approved", now=1.0)
    board.mark("merged", now=2.0)

    assert board.since("approved") == 1.0
    assert board.since("merged") == 2.0


def test_a_mark_defaults_to_the_clock_started_at_is_taken_from() -> None:
    board = Board(started_at=0.0)

    board.mark("approved")

    stamped = board.since("approved")
    assert stamped is not None and stamped > 0.0


def test_a_stamp_records_when_a_key_arrived_where_it_is() -> None:
    board = Board(started_at=0.0)

    board.stamp("T-01", now=3.0)
    board.stamp("T-01", now=9.0)

    assert board.status_since["T-01"] == 9.0


def test_dropping_a_key_takes_everything_the_board_holds_about_it() -> None:
    board = Board(started_at=0.0)
    board.stamp("T-01", now=1.0)
    board.activity["T-01"] = "editing api.py"
    board.stamp("T-02", now=2.0)

    board.drop("T-01")

    assert "T-01" not in board.status_since
    assert "T-01" not in board.activity
    assert board.status_since["T-02"] == 2.0


def test_dropping_a_key_the_board_never_saw_is_not_an_error() -> None:
    board = Board(started_at=0.0)

    board.drop("T-99")

    assert board.status_since == {}


# -- live() and show() -------------------------------------------------------


async def test_live_opens_one_session_for_the_whole_run() -> None:
    terminal = HeadlessTerminal()
    board = Board(started_at=0.0)

    async with live(terminal, board) as display:
        assert isinstance(display, Display)

    assert len(terminal.frames) == 2  # one on entry, one on exit


async def test_the_screen_is_blank_until_the_first_show() -> None:
    terminal = HeadlessTerminal()

    async with live(terminal, Board(started_at=0.0)):
        pass

    assert terminal.frames[0].strip() == ""


async def test_show_swaps_the_screen_within_one_session() -> None:
    terminal = HeadlessTerminal()

    async with live(terminal, Board(started_at=0.0)) as display:
        display.show(lambda: screen("interviewing"))
        display.show(lambda: screen("dashboard"))

    assert "interviewing" in terminal.frames[1]
    assert "dashboard" in terminal.frames[2]
    assert "dashboard" in terminal.frames[-1]


# -- questions ----------------------------------------------------------------


async def test_ask_puts_a_terminal_question_and_hands_back_the_answer() -> None:
    terminal = HeadlessTerminal([Answer("approve", was_free_text=False)])
    question = Question(header="add-auth", title="Approve?", options=(Option("approve", "go"),))

    async with live(terminal, Board(started_at=0.0)) as display:
        answer = await display.ask(question)

    assert answer == Answer("approve", was_free_text=False)
    assert terminal.questions == [question]


async def test_ask_agent_translates_the_agent_s_question_and_returns_the_text() -> None:
    terminal = HeadlessTerminal([Answer("postgres", was_free_text=True)])
    question = AgentQuestion(
        title="Which store?",
        options=(AgentOption("sqlite", "a file"), AgentOption("postgres", "a server")),
    )

    async with live(terminal, Board(started_at=0.0)) as display:
        text = await display.ask_agent("T-01", question)

    assert text == "postgres"
    assert terminal.questions == [
        Question(
            header="T-01",
            title="Which store?",
            options=(Option("sqlite", "a file"), Option("postgres", "a server")),
        )
    ]


async def test_confirm_asks_under_the_given_header_and_returns_nothing() -> None:
    terminal = HeadlessTerminal([Answer("continue", was_free_text=False)])

    async with live(terminal, Board(started_at=0.0)) as display:
        assert await display.confirm("add-auth", "press enter to continue") is None

    asked = terminal.questions[0]
    assert asked.header == "add-auth"
    assert asked.title == "press enter to continue"
    assert len(asked.options) == 1


async def test_confirm_takes_free_text_as_a_press_of_enter() -> None:
    """Whatever the user typed, they were asked only to come back."""
    terminal = HeadlessTerminal([Answer("fixed it", was_free_text=True)])

    async with live(terminal, Board(started_at=0.0)) as display:
        assert await display.confirm("add-auth", "press enter to continue") is None


# -- activity ------------------------------------------------------------------


async def test_activity_writes_to_the_board_under_its_own_key() -> None:
    board = Board(started_at=0.0)

    async with live(HeadlessTerminal(), board) as display:
        display.activity("T-01")("editing api.py")
        display.activity("T-02")("running tests")

    assert board.activity == {"T-01": "editing api.py", "T-02": "running tests"}


async def test_the_last_writer_to_one_key_wins() -> None:
    board = Board(started_at=0.0)

    async with live(HeadlessTerminal(), board) as display:
        write = display.activity("T-01")
        write("reading")
        write("writing")

    assert board.activity["T-01"] == "writing"
