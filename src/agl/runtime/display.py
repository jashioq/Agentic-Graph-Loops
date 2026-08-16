"""The one path to the terminal: display-only state, and the session over it.

Layer: runtime. The single module importing both `agl.core.terminal` and
`agl.core.agent` — the cross-module wiring this layer exists for. `ask_agent` is
where an `AgentQuestion` becomes a `Question`, once.

`Board` is the only mutable state in runtime and is display-only: nothing here
is read to decide what a run does. The workflow owns it. `Display` holds the
session, which exists for exactly as long as the `live` block does.
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

# TODO I feel like all of this should be in the workflow. This does not seem very reusable between workflows, each workflow will have to build something like that again if it doesn't use tickets. Workflows should just use the simple core module, define their own screens and then request them to be disaplyed. This is unnecessary complexity.
@dataclass
class Board:
    """Ephemeral, in-memory, read only by rendering.

    `started_at` covers the whole session; `marks` are named points a stage
    times from; `status_since` is per key; `activity` is one line per key.
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
        """Record when `key` arrived where it is. `now` defaults to the monotonic clock."""
        self.status_since[key] = time.monotonic() if now is None else now

    def drop(self, key: str) -> None:
        """Forget everything the board holds about `key`, which has left the run."""
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
        """An agent's own question, put to the user.

        param: header - who is asking: a run label or work item id
        return: str - what the user said
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
        """Holds until the user comes back. Nothing is returned: the caller re-reads
        the world rather than what was said here."""
        await self._session.ask(
            Question(header=header, title=title, options=(Option("continue", "carry on"),))
        )

    def activity(self, key: str) -> Callable[[str], None]:
        """A writer for one key's activity line, closed over the key it writes to."""

        def on_activity(text: str) -> None:
            self._board.activity[key] = text

        return on_activity


@asynccontextmanager
async def live(terminal: Terminal, board: Board) -> AsyncIterator[Display]:
    """Opens the terminal once for the whole run and yields the way in.

    Blank until the first `show`; stages swap the screen on this one session.
    """
    async with terminal.live() as session:
        yield Display(session, board)
