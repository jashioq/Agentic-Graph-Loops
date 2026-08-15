"""The one path to the terminal: display-only state, and the session over it.

Layer: runtime. The single module that imports `agl.core.terminal` *and*
`agl.core.agent` — forbidden inside core, and exactly the cross-module wiring
this layer exists for. An `AgentQuestion` and a `Question` are the same question
asked by two modules that must not know about each other, and `ask_agent` is
where the translation happens, once, rather than in every workflow that runs an
agent a person can talk to.

`Board` is the only mutable state in runtime, and it is display-only by
construction: nothing here is ever read to decide what a run does. Delete the
board and the run still finishes correctly, you just cannot watch it. It is
created and owned by the workflow, which is also what keeps that true — runtime
holds no shared run state of its own.

Two kinds of clock live on it, and the difference is the reason `marks` is a
dict rather than a field per stage. `started_at` covers the whole session, from
before the first question is asked, and is what a header's timer reads.
Everything else a run wants to time from — approval, a first merge — is a named
mark set when it happens, so a stage's timer means the stage rather than the
session. `status_since` is per key and answers "how long has this been where it
is", which is a different question again.

`Display` holds the session, so a workflow never keeps one itself and never has
to assert it is still open: the session exists for as long as the `live` block
does, and every path to the terminal goes through the object that block yields.
"""

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from agl.core.agent import AgentQuestion
from agl.core.terminal import (
    Answer,
    LiveSession,
    Option,
    Question,
    Screen,
    Terminal,
)

__all__ = ["Board", "Display", "live"]


@dataclass
class Board:
    """Ephemeral, in-memory, read only by rendering.

    Everything here is about watching a run rather than running it: when the
    session began, the stages that have been reached, when each key last
    changed, and the one-line activity whatever is working on it last reported.
    """

    started_at: float
    marks: dict[str, float] = field(default_factory=dict)
    status_since: dict[str, float] = field(default_factory=dict)
    activity: dict[str, str] = field(default_factory=dict)

    def mark(self, name: str, now: float | None = None) -> None:
        """Record that a named stage was reached, for a timer to count from."""
        self.marks[name] = time.monotonic() if now is None else now

    def since(self, name: str) -> float | None:
        """When `name` was marked, or `None` if it has not been reached yet."""
        return self.marks.get(name)

    def stamp(self, key: str, now: float | None = None) -> None:
        """Record when `key` arrived where it is.

        `now` is explicit so a test can assert on the value; the default reads
        the same monotonic clock `started_at` is taken from, so a caller does
        not have to thread one through.
        """
        self.status_since[key] = time.monotonic() if now is None else now

    def drop(self, key: str) -> None:
        """Forget everything the board holds about `key`.

        For a key that has left the run — an addition a later step refused, say.
        A stamp or an activity line for something no longer there is state
        rendering would have nothing to draw.
        """
        self.status_since.pop(key, None)
        self.activity.pop(key, None)


class Display:
    """One live session, and everything a workflow does to the screen through it."""

    def __init__(self, session: LiveSession, board: Board) -> None:
        self._session = session
        self._board = board

    def show(self, build: Callable[[], Screen]) -> None:
        """Draw this stage's screen from now on, starting immediately."""
        self._session.show(build)

    async def ask(self, question: Question) -> Answer:
        """Take over the screen to ask, and hand back what the user said."""
        return await self._session.ask(question)

    async def ask_agent(self, header: str, question: AgentQuestion) -> str:
        """An agent's own question, put to the user under `header`.

        `header` is who is asking — a run label, a work item id — because a
        person answering three concurrent agents has no other way to tell which
        one is waiting on them.
        """
        answer = await self._session.ask(
            Question(
                header=header,
                title=question.title,
                options=tuple(Option(o.label, o.description) for o in question.options),
            )
        )
        return answer.text

    async def confirm(self, header: str, title: str) -> None:
        """Hold until the user comes back. Whatever they typed is not an answer.

        The gate for "something out there needs a person, and the run cannot
        tell when they are done." There is nothing to return: the caller looks
        at the world again rather than at what was said here.
        """
        await self._session.ask(
            Question(header=header, title=title, options=(Option("continue", "carry on"),))
        )

    def activity(self, key: str) -> Callable[[str], None]:
        """A writer for one key's activity line, to hand to a long-running call.

        Built per key and closed over it, so concurrent work reports into its
        own row rather than into a shared "current" line that only ever shows
        whichever one wrote last.
        """

        def on_activity(text: str) -> None:
            self._board.activity[key] = text

        return on_activity


@asynccontextmanager
async def live(terminal: Terminal, board: Board) -> AsyncIterator[Display]:
    """Open the terminal once, for the whole run, and yield the way in.

    The session starts with no screen at all — blank until the first `show` —
    because a run opens the display before it knows what its first stage looks
    like. Stages swap the screen on the session they were given rather than
    opening one each, so the terminal is entered and left exactly once.
    """
    async with terminal.live() as session:
        yield Display(session, board)
