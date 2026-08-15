"""The decompose approval flow: propose tickets, ask approve/revise/abort, loop.

Layer: workflows. Imports `agl.runtime.dag` and `agl.core.terminal`, and this
workflow's `agents`, `models`, `render`, `tools`, and `wiring`.

Runs before `Run.implement_all` builds the dashboard — the decompose screen
shows tickets once there are any, and falls back to the bare session header
when there are none yet. `Run.live` exists for the whole session by this
point, so both states have a timer and an activity string to show.
"""

from agl.core.store import Store
from agl.core.terminal import LiveSession, Option, Question, Row, Rows, Screen, Terminal, Text
from agl.runtime.dag import Dag
from agl.workflows.tickets import agents
from agl.workflows.tickets import tools as ticket_tools
from agl.workflows.tickets.models import Ticket, tickets_from_json
from agl.workflows.tickets.render import session_header
from agl.workflows.tickets.state import Live
from agl.workflows.tickets.wiring import Wiring

__all__ = ["Approval", "DecomposeAbortedError", "session_screen"]


class DecomposeAbortedError(Exception):
    """Raised when the user aborted decomposition before approving any tickets."""


class Approval:
    """Propose tickets, ask for approval, and loop on a revision until settled."""

    def __init__(self, terminal: Terminal, store: Store, label: str, wiring: Wiring) -> None:
        self._terminal = terminal
        self._store = store
        self._label = label
        self._wiring = wiring

    async def run(self) -> tuple[Ticket, ...]:
        tickets: tuple[Ticket, ...] = ()

        def screen() -> Screen:
            return _decompose_screen(self._label, self._live(), tickets)

        async with self._terminal.live(screen) as session:
            revision = ""
            while True:
                tickets = await self._propose(session, revision)
                answer = await self._ask_approval(session, tickets)
                if answer is None:
                    return tickets
                revision = answer

    def _live(self) -> Live:
        live = self._wiring.live()
        assert live is not None
        return live

    async def _propose(self, session: LiveSession, revision: str) -> tuple[Ticket, ...]:
        if revision:
            self._append_spec(revision)
        ctx = self._wiring.ctx(self._wiring.ask(session, None))
        await agents.decompose(ctx, self._wiring.activity(self._label))
        payload = self._store.read_json(ticket_tools.TICKETS_KEY)
        return tickets_from_json(payload)

    async def _ask_approval(
        self, session: LiveSession, tickets: tuple[Ticket, ...]
    ) -> str | None:
        question = Question(
            header=self._label,
            title=f"Approve these {len(tickets)} tickets?",
            options=(
                Option("approve", "Start the run with these tickets"),
                Option("abort", "Cancel without creating any tickets"),
            ),
        )
        answer = await session.ask(question)
        if answer.was_free_text:
            return answer.text
        if answer.text == "approve":
            return None
        raise DecomposeAbortedError("the user aborted decomposition")

    def _append_spec(self, revision: str) -> None:
        spec = self._store.read(ticket_tools.SPEC_KEY)
        self._store.write(
            ticket_tools.SPEC_KEY, f"{spec}\n\n## Decomposition feedback\n\n{revision}\n"
        )


def session_screen(label: str, live: Live) -> Screen:
    return Screen(header=session_header(label, live), content=Rows())


def _decompose_screen(label: str, live: Live, tickets: tuple[Ticket, ...]) -> Screen:
    if not tickets:
        return session_screen(label, live)
    dag = Dag()
    for ticket in tickets:
        dag.add_node(ticket.id)
    for ticket in tickets:
        for blocker in ticket.blocked_by:
            dag.add_edge(ticket.id, blocker)
    by_id = {t.id: t for t in tickets}
    rows = []
    for level in dag.levels():
        for ticket_id in level:
            ticket = by_id[ticket_id]
            blocked = ", ".join(ticket.blocked_by) if ticket.blocked_by else "—"
            rows.append(Row(Text(f"{ticket.id}: {ticket.title} (blocked by: {blocked})")))
    return Screen(header=session_header(label, live), content=Rows(*rows))
