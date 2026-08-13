"""A question raised inside an agent run, answered on the terminal, and back again.

Three modules and one round trip. `agent` raises an `AgentQuestion` because it
must not import `terminal`; the wiring here is the layer that knows why the
question was asked, so it is the layer that turns one into a `terminal.Question`
and the `Answer` back into the string the run is waiting for.

The terminal is headless: it renders through the real renderer into strings and
answers from a script, so a question that never reached a screen shows up as a
missing frame rather than as a test that quietly passed.
"""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from agl.core.agent import NO_PARAMS, AgentOption, AgentQuestion, AgentSpec, Tool
from agl.core.terminal import (
    Answer,
    LiveSession,
    Option,
    Question,
    Row,
    Rows,
    Screen,
    Text,
)
from tests.fakes import FakeAgentRunner, HeadlessTerminal, ScriptedRun

type OnQuestion = Callable[[AgentQuestion], Awaitable[str]]

LABEL = "add-auth"
TIMEOUT = 5.0

STORAGE = AgentQuestion(
    title="Which storage layer?",
    options=(
        AgentOption(label="sqlite", description="One file on disk"),
        AgentOption(label="postgres", description="A server to run"),
    ),
)
NAMING = AgentQuestion(
    title="What should the token field be called?",
    options=(AgentOption(label="token", description="The obvious one"),),
)


# -- the wiring under test -------------------------------------------------


def ask_on(session: LiveSession, header: str) -> OnQuestion:
    """The translation a workflow owns: agent's question in, its answer out."""

    async def on_question(question: AgentQuestion) -> str:
        answer = await session.ask(
            Question(
                header=header,
                title=question.title,
                options=tuple(
                    Option(label=option.label, description=option.description)
                    for option in question.options
                ),
            )
        )
        return answer.text

    return on_question


def screen(states: dict[str, str]) -> Callable[[], Screen]:
    """A live display over some state the run is changing as it goes."""

    def build() -> Screen:
        return Screen(
            header=Row(Text(LABEL)),
            content=Rows(*(Row(Text(f"{node} {state}")) for node, state in states.items())),
            footer=None,
        )

    return build


def hold_tool(here: asyncio.Event, release: asyncio.Event) -> Tool:
    """Parks a run mid-call so another run can be brought alongside it."""

    async def handler(arguments: dict[str, Any]) -> str:
        here.set()
        await release.wait()
        return "held"

    return Tool(name="hold", description="Wait.", schema=NO_PARAMS, handler=handler)


def spec(role: str, cwd: Path, *tools: Tool) -> AgentSpec:
    return AgentSpec(prompt="Do the work.", cwd=cwd, role=role, tools=tools)


# -- one question ----------------------------------------------------------


async def test_the_question_reaches_the_terminal_and_the_answer_comes_back(
    tmp_path: Path,
) -> None:
    terminal = HeadlessTerminal(answers=[Answer("sqlite", was_free_text=False)], clock=lambda: 0.0)
    runner = FakeAgentRunner({"implement": ScriptedRun("done", question=STORAGE)})

    async with terminal.live(screen({"T-01": "running"})) as session:
        result = await runner.run(spec("implement", tmp_path), on_question=ask_on(session, "T-01"))

    assert terminal.questions == [
        Question(
            header="T-01",
            title="Which storage layer?",
            options=(
                Option("sqlite", "One file on disk"),
                Option("postgres", "A server to run"),
            ),
        )
    ]
    assert runner.answers == ["sqlite"]
    assert result.text == "done"


async def test_the_question_interrupted_a_live_display(tmp_path: Path) -> None:
    terminal = HeadlessTerminal(answers=[Answer("sqlite", was_free_text=False)], clock=lambda: 0.0)
    runner = FakeAgentRunner({"implement": ScriptedRun("done", question=STORAGE)})
    states = {"T-01": "asking"}

    async with terminal.live(screen(states)) as session:
        await runner.run(spec("implement", tmp_path), on_question=ask_on(session, "T-01"))
        states["T-01"] = "merged"

    # The frame the question took the screen over from, and the one after it.
    assert "T-01 asking" in terminal.frames[1]
    assert "T-01 merged" in terminal.frames[-1]


async def test_free_text_comes_back_as_the_answer(tmp_path: Path) -> None:
    terminal = HeadlessTerminal(answers=[Answer("neither, use a flat file", was_free_text=True)])
    runner = FakeAgentRunner({"implement": ScriptedRun("done", question=STORAGE)})

    async with terminal.live(screen({})) as session:
        await runner.run(spec("implement", tmp_path), on_question=ask_on(session, "T-01"))

    assert runner.answers == ["neither, use a flat file"]
    assert runner.answers[0] not in [option.label for option in STORAGE.options]


# -- two at once -----------------------------------------------------------


async def test_two_concurrent_runs_each_get_their_own_answer(tmp_path: Path) -> None:
    terminal = HeadlessTerminal(
        answers=[Answer("sqlite", was_free_text=False), Answer("token", was_free_text=False)],
        clock=lambda: 0.0,
    )
    runner = FakeAgentRunner(
        {
            "storage": ScriptedRun("a", calls=(("hold", {}),), question=STORAGE),
            "naming": ScriptedRun("b", calls=(("hold", {}),), question=NAMING),
        }
    )
    here = {"T-01": asyncio.Event(), "T-02": asyncio.Event()}
    release = asyncio.Event()
    answered: dict[str, str] = {}
    asking = 0
    most_asking = 0

    def watched(inner: OnQuestion, node: str) -> OnQuestion:
        """Records the answer this run got, and how many asks overlapped."""

        async def on_question(question: AgentQuestion) -> str:
            nonlocal asking, most_asking
            asking += 1
            most_asking = max(most_asking, asking)
            try:
                answer = await inner(question)
            finally:
                asking -= 1
            answered[node] = answer
            return answer

        return on_question

    async def run(node: str, role: str) -> None:
        await runner.run(
            spec(role, tmp_path, hold_tool(here[node], release)),
            on_question=watched(ask_on(session, node), node),
        )

    async with terminal.live(screen({})) as session:
        async with asyncio.timeout(TIMEOUT):
            both = asyncio.gather(run("T-01", "storage"), run("T-02", "naming"))
            await here["T-01"].wait()
            await here["T-02"].wait()
            # Both runs are now inside `run`, before either has asked.
            release.set()
            await both

    # Which run asked first is the scheduler's business; that each got the
    # answer to its own question is not.
    order = [question.header for question in terminal.questions]
    assert sorted(order) == ["T-01", "T-02"]
    assert answered == dict(zip(order, ["sqlite", "token"], strict=True))
    assert most_asking == 1


async def test_the_terminal_showed_one_question_at_a_time(tmp_path: Path) -> None:
    # The headless terminal cannot interleave two asks — `ask` has no await in
    # it — so what this pins is that the wiring goes through the session for
    # every question rather than answering any of them itself.
    terminal = HeadlessTerminal(
        answers=[Answer("sqlite", was_free_text=False), Answer("token", was_free_text=False)],
        clock=lambda: 0.0,
    )
    runner = FakeAgentRunner(
        {
            "storage": ScriptedRun("a", question=STORAGE),
            "naming": ScriptedRun("b", question=NAMING),
        }
    )

    async with terminal.live(screen({})) as session:
        async with asyncio.timeout(TIMEOUT):
            await asyncio.gather(
                runner.run(spec("storage", tmp_path), on_question=ask_on(session, "T-01")),
                runner.run(spec("naming", tmp_path), on_question=ask_on(session, "T-02")),
            )

    assert [question.title for question in terminal.questions] == [STORAGE.title, NAMING.title]
    assert len(terminal.questions) == 2


# -- nobody to ask ---------------------------------------------------------


async def test_a_run_with_no_question_handler_completes(tmp_path: Path) -> None:
    terminal = HeadlessTerminal(clock=lambda: 0.0)
    runner = FakeAgentRunner({"implement": ScriptedRun("decided for myself")})

    async with terminal.live(screen({"T-01": "running"})):
        async with asyncio.timeout(TIMEOUT):
            result = await runner.run(spec("implement", tmp_path))

    assert result.text == "decided for myself"
    assert terminal.questions == []


async def test_a_question_with_nobody_wired_up_fails_instead_of_hanging(tmp_path: Path) -> None:
    # `ClaudeRunner` refuses the question and tells the model to use its own
    # judgement; the fake asserts instead, because a workflow that scripted a
    # question and passed no handler has a wiring bug. Either way it is not a
    # run that waits forever for an answer nobody will give.
    runner = FakeAgentRunner({"implement": ScriptedRun("done", question=STORAGE)})

    async with asyncio.timeout(TIMEOUT):
        with pytest.raises(AssertionError, match="on_question"):
            await runner.run(spec("implement", tmp_path))
