"""The decompose approval flow: propose tickets, ask approve/revise/abort, loop.

Layer: workflows. Imports `agl.core.dag` and `agl.core.terminal`, and this
workflow's `agents`, `models`, `tools`, and `wiring`.

Runs exactly once per run, before `Run.live` exists — approval happens
against a proposed plan, not yet a running graph, which is why the screen
falls back to the plain label when there is nothing to show yet.
"""

from agl.core.dag import Dag
from agl.core.store import Store
from agl.core.terminal import LiveSession, Option, Question, Row, Rows, Screen, Terminal, Text
from agl.workflows.tickets import agents
from agl.workflows.tickets import tools as ticket_tools
from agl.workflows.tickets.models import Ticket, tickets_from_json
from agl.workflows.tickets.wiring import Wiring

__all__ = ["Approval", "DecomposeAbortedError", "plain_screen"]


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
            return _decompose_screen(self._label, tickets)

        async with self._terminal.live(screen) as session:
            revision = ""
            while True:
                tickets = await self._propose(session, revision)
                answer = await self._ask_approval(session, tickets)
                if answer is None:
                    return tickets
                revision = answer

    async def _propose(self, session: LiveSession, revision: str) -> tuple[Ticket, ...]:
        if revision:
            self._append_spec(revision)
        ctx = self._wiring.ctx(self._wiring.ask(session, None))
        await agents.decompose(ctx)
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


def plain_screen(label: str) -> Screen:
    return Screen(content=Rows(Row(Text(label))))


def _decompose_screen(label: str, tickets: tuple[Ticket, ...]) -> Screen:
    if not tickets:
        return plain_screen(label)
    dag = Dag()
    for ticket in tickets:
        dag.add_node(ticket.id)
    for ticket in tickets:
        for blocker in ticket.blocked_by:
            dag.add_edge(ticket.id, blocker)
    by_id = {t.id: t for t in tickets}
    rows = [Row(Text(label))]
    for level in dag.levels():
        for ticket_id in level:
            ticket = by_id[ticket_id]
            blocked = ", ".join(ticket.blocked_by) if ticket.blocked_by else "—"
            rows.append(Row(Text(f"{ticket.id}: {ticket.title} (blocked by: {blocked})")))
    return Screen(content=Rows(*rows))
