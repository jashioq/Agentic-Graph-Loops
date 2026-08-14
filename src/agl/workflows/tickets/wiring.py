"""Per-ticket callbacks: the agent context, activity, and questions.

Layer: workflows. Imports `agl.core.agent`, `agl.core.store`, this workflow's
`models` and `state`, and `agents` for the types it builds callbacks around.

`Wiring.ctx` and `Wiring.ask` are called once per ticket, inside the
scheduler's loop, and each `ask` closes over the ticket id it was built for.
A shared mutable "current ticket" in their place would make concurrent
tickets race, and answers would land on the wrong agent in a log that looks
perfectly well-formed — so every caller must build a fresh `ask` per ticket
rather than reusing one.

`ask`'s suspend/resume around the wait must stay symmetric: `AWAITING_INPUT`
returns the ticket to the status it was suspended from, and `transition` is
what records both moves, so a ticket's status history reads as a straight
line with one detour rather than a fork.
"""

from collections.abc import Awaitable, Callable
from pathlib import Path

from agl.core.agent import AgentQuestion, AgentRunner
from agl.core.store import Store
from agl.core.terminal import LiveSession, Option, Question
from agl.workflows.tickets import state
from agl.workflows.tickets.agents import AgentContext, Limits
from agl.workflows.tickets.models import Status
from agl.workflows.tickets.state import Live, RunState

__all__ = ["PROMPTS_DIR", "Wiring"]

PROMPTS_DIR = Path(__file__).parent / "prompts"


class Wiring:
    """Builds the `AgentContext`, activity writer, and `ask` callback for one ticket."""

    def __init__(
        self,
        agent: AgentRunner,
        store: Store,
        repo: Path,
        state_: RunState,
        label: str,
        live: Callable[[], Live | None],
    ) -> None:
        self._agent = agent
        self._store = store
        self._repo = repo
        self._state = state_
        self._label = label
        self._live = live

    def ctx(self, ask: Callable[[AgentQuestion], Awaitable[str]]) -> AgentContext:
        return AgentContext(
            runner=self._agent,
            store=self._store,
            repo=self._repo,
            prompts=PROMPTS_DIR,
            limits=Limits(model="sonnet"),
            ask=ask,
        )

    def live(self) -> Live | None:
        """The `Live` this run is watched through, if anyone is watching."""
        return self._live()

    def activity(self, ticket_id: str) -> Callable[[str], None]:
        def on_activity(text: str) -> None:
            live = self._live()
            assert live is not None
            live.activity[ticket_id] = text

        return on_activity

    def ticket_ask(
        self, session: LiveSession, ticket_id: str
    ) -> Callable[[AgentQuestion], Awaitable[str]]:
        """`ask` for a specific ticket — never `None`, so suspend/resume always fires."""
        return self.ask(session, ticket_id)

    def ask(
        self, session: LiveSession, ticket_id: str | None
    ) -> Callable[[AgentQuestion], Awaitable[str]]:
        async def ask(question: AgentQuestion) -> str:
            frm = self._suspend(ticket_id)
            answer = await session.ask(_to_question(ticket_id or self._label, question))
            self._resume(ticket_id, frm)
            return answer.text

        return ask

    def _suspend(self, ticket_id: str | None) -> Status | None:
        if ticket_id is None:
            return None
        frm = self._state.tickets[ticket_id].status
        state.set_status(self._state, self._live(), ticket_id, Status.AWAITING_INPUT)
        return frm

    def _resume(self, ticket_id: str | None, frm: Status | None) -> None:
        if ticket_id is not None and frm is not None:
            state.set_status(self._state, self._live(), ticket_id, frm)


def _to_question(header: str, question: AgentQuestion) -> Question:
    return Question(
        header=header,
        title=question.title,
        options=tuple(Option(o.label, o.description) for o in question.options),
    )
