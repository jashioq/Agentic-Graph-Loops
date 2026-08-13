"""Fakes for core modules. Real classes inheriting the ABC, never mocks.

An incomplete fake fails at instantiation, which is the point: when an ABC grows
a method, every fake here stops working until it is implemented.
"""

import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from rich.console import Console

from agl.core.agent import (
    AgentQuestion,
    AgentResult,
    AgentRunner,
    AgentSpec,
    Tool,
)
from agl.core.store import MissingKeyError, Store
from agl.core.terminal import Answer, LiveSession, Question, Screen, Terminal
from agl.core.terminal.impl.render import to_renderable

__all__ = [
    "FakeAgentRunner",
    "HeadlessTerminal",
    "MemoryStore",
    "ScriptedRun",
    "ToolResult",
]


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


class MemoryStore(Store):
    """Documents in a dict, so the base class's JSON layer can be exercised alone.

    Keys are taken as given — validation is a `FileStore` concern, and there is
    no root here for a key to escape from.
    """

    def __init__(self) -> None:
        self.documents: dict[str, str] = {}

    def read(self, key: str) -> str:
        if key not in self.documents:
            raise MissingKeyError(key)
        return self.documents[key]

    def write(self, key: str, content: str) -> None:
        self.documents[key] = content

    def delete(self, key: str) -> None:
        if key not in self.documents:
            raise MissingKeyError(key)
        del self.documents[key]

    def exists(self, key: str) -> bool:
        return key in self.documents

    def list(self, prefix: str = "") -> tuple[str, ...]:
        return tuple(sorted(key for key in self.documents if key.startswith(prefix)))


@dataclass(frozen=True)
class ToolResult:
    """What one tool call handed back, and whether it came back as a failure.

    The two are recorded together because a failed call is an outcome the run
    kept going after, not an exception it died on — so a test asserting on
    `tool_results` has to be able to tell the two apart without reading the
    text. It is the same pair MCP hands the model: the text, and `is_error`.
    """

    text: str
    is_error: bool = False


@dataclass(frozen=True)
class ScriptedRun:
    """What one call to `FakeAgentRunner` should do, and what it hands back.

    `text` is the shorthand every script starts as; pass a whole `AgentResult`
    as `result` when a test cares about the cost, the session, or the structured
    output. `calls` names tools from the spec the run should invoke on its way
    through, which is how a workflow test proves a role was given the right ones.
    """

    text: str = ""
    result: AgentResult | None = None
    activity: tuple[str, ...] = ()
    question: AgentQuestion | None = None
    calls: tuple[tuple[str, dict[str, Any]], ...] = ()

    def outcome(self) -> AgentResult:
        """The `AgentResult` this run produces."""
        if self.result is not None:
            return self.result
        return AgentResult(
            text=self.text,
            structured=None,
            session_id="fake-session",
            cost_usd=0.0,
            num_turns=1,
            duration_ms=0,
            terminal_reason="completed",
        )


type Script = Mapping[str, ScriptedRun | AgentResult | str] | Iterable[
    ScriptedRun | AgentResult | str
]


class FakeAgentRunner(AgentRunner):
    """An agent that never calls a model and does exactly what it was told to.

    Script it either by role — `{"implement": "the patch"}`, reusable because a
    workflow runs one role over many items — or as a list consumed in order.
    Entries may be a bare string, a whole `AgentResult`, or a `ScriptedRun` when
    the call should also invoke tools, ask a question, or report activity.

    Everything it was asked to do is recorded: `specs`, `answers`, and every
    call's `ToolResult` in `tool_results`, failures included.
    """

    def __init__(self, script: Script = ()) -> None:
        self.specs: list[AgentSpec] = []
        self.answers: list[str] = []
        self.tool_results: list[ToolResult] = []
        self._by_role: dict[str, ScriptedRun] | None = None
        self._in_order: deque[ScriptedRun] | None = None

        if isinstance(script, Mapping):
            self._by_role = {role: _scripted(run) for role, run in script.items()}
        else:
            self._in_order = deque(_scripted(run) for run in script)

    async def run(
        self,
        spec: AgentSpec,
        on_activity: Callable[[str], None] | None = None,
        on_question: Callable[[AgentQuestion], Awaitable[str]] | None = None,
    ) -> AgentResult:
        self.specs.append(spec)
        scripted = self._next(spec.role)

        for activity in scripted.activity:
            if on_activity is not None:
                on_activity(activity)

        for name, arguments in scripted.calls:
            self.tool_results.append(await self._invoke(spec, name, arguments))

        if scripted.question is not None:
            assert on_question is not None, (
                f"role {spec.role!r} was scripted to ask "
                f"{scripted.question.title!r} but was called without on_question"
            )
            self.answers.append(await on_question(scripted.question))

        return scripted.outcome()

    def _next(self, role: str) -> ScriptedRun:
        """The scripted run for this call, failing loudly when there is none."""
        if self._by_role is not None:
            assert role in self._by_role, (
                f"no script for role {role!r}; "
                f"scripted roles are {sorted(self._by_role)}"
            )
            return self._by_role[role]

        assert self._in_order, f"the script ran out before role {role!r} was called"
        return self._in_order.popleft()

    async def _invoke(self, spec: AgentSpec, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Call one of the spec's own tools, wrapped the way a real run wraps it.

        The wrapping is `agent.impl.tools`', not an invention of the fake's: a
        handler that *raises* is a fact about the world the model should react
        to, so it becomes an error result and the run carries on; a handler
        returning a *non-string* violates `Tool`'s own type, which the model
        cannot fix, so it raises rather than being coerced into an answer.
        """
        for tool in spec.tools:
            if tool.name == name:
                return await _wrapped(tool, arguments)
        raise AssertionError(
            f"role {spec.role!r} has no tool {name!r}; "
            f"it was given {[tool.name for tool in spec.tools]}"
        )


async def _wrapped(tool: Tool, arguments: dict[str, Any]) -> ToolResult:
    """One handler call, classified the way `agent.impl.tools._register` does."""
    try:
        result = await tool.handler(arguments)
    except Exception as error:  # noqa: BLE001 - the model decides what to do
        return ToolResult(f"{type(error).__name__}: {error}", is_error=True)
    if not isinstance(result, str):
        raise TypeError(f"tool {tool.name!r} returned {type(result).__name__}, expected str")
    return ToolResult(result)


def _scripted(run: ScriptedRun | AgentResult | str) -> ScriptedRun:
    """Every shorthand a script may use, as one `ScriptedRun`."""
    if isinstance(run, ScriptedRun):
        return run
    if isinstance(run, AgentResult):
        return ScriptedRun(result=run)
    return ScriptedRun(text=run)
