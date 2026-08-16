"""One run's state as a value, and the pure transitions that move it.

Layer: workflows. Pure — no I/O, no async, and nothing from below but `dag`.
Everything derivable is derived: the graph and the display order are rebuilt on
every question, so neither can contradict the tickets it came from.
`documents/state_document.py` stores a `Run`; nothing here knows how.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from agl.runtime.dag import CycleError, Dag, NodeId, NodeState
from agl.workflows.tickets.errors import (
    DuplicateTicketError,
    Halt,
    InvalidStateError,
    UnknownTicketError,
)
from agl.workflows.tickets.models import Status, Ticket, transition

__all__ = [
    "Run",
    "bugs_first",
    "check",
    "dag_of",
    "display_order",
    "with_base_sha",
    "with_bugs",
    "with_halt",
    "with_status",
    "with_tickets",
]


@dataclass(frozen=True)
class Run:
    """One run's whole state, as a value. Every change produces a new one.

    Tickets are in insertion order, which is itself state: it is what
    `display_order` and the graph's node ordering are built from.
    """

    tickets: tuple[Ticket, ...] = ()
    halt: Halt | None = None

    def ticket(self, ticket_id: str) -> Ticket:
        """The ticket, or `UnknownTicketError`."""
        found = self.get(ticket_id)
        if found is None:
            raise UnknownTicketError(ticket_id)
        return found

    def get(self, ticket_id: str) -> Ticket | None:
        """The ticket, or `None` — for a caller that is asking rather than assuming."""
        for ticket in self.tickets:
            if ticket.id == ticket_id:
                return ticket
        return None


# -- transitions ----------------------------------------------------------


def with_tickets(run: Run, tickets: Sequence[Ticket]) -> Run:
    """`run` plus `tickets`: the approved set, or a round of bugs.

    Every id has to be new and every blocker has to resolve, to a ticket in the
    run or in this batch. `check` catches a cycle closed within the batch.
    """
    incoming = tuple(tickets)
    seen: set[str] = set()
    known = {ticket.id for ticket in run.tickets}
    for ticket in incoming:
        if ticket.id in known or ticket.id in seen:
            raise DuplicateTicketError(ticket.id)
        if ticket.status is not Status.PENDING:
            raise InvalidStateError(
                f"ticket {ticket.id!r} joins the run as {ticket.status.value}, "
                "but a new ticket is pending"
            )
        seen.add(ticket.id)
    for ticket in incoming:
        for blocker in ticket.blocked_by:
            if blocker not in known | seen:
                raise UnknownTicketError(blocker)

    widened = replace(run, tickets=(*run.tickets, *incoming))
    check(widened)
    return widened


def with_status(run: Run, ticket_id: str, status: Status) -> Run:
    """`run` with one ticket moved to `status`, through the life cycle's own rules.

    Raises `UnknownTicketError` for an unheld id and `IllegalTransitionError`
    for a forbidden move; nothing is built in either case.
    """
    return _replaced(run, transition(run.ticket(ticket_id), status))


def with_bugs(run: Run, parent_id: str, bugs: Sequence[Ticket]) -> Run:
    """Fold review findings into the run as work the parent now waits on.

    One write: the bugs, the parent's new blockers, its released status and its
    advanced `review_round` all land together. Raises `ValueError` for a bug
    that does not name `parent_id`, or that waits on the ticket it fixes.
    """
    parent = run.ticket(parent_id)
    incoming = tuple(bugs)
    if not incoming:
        raise ValueError(f"no bugs to file against {parent_id!r}")
    for bug in incoming:
        if bug.parent != parent_id:
            raise ValueError(f"bug {bug.id!r} names parent {bug.parent!r}, not {parent_id!r}")
        if parent_id in bug.blocked_by:
            raise ValueError(
                f"bug {bug.id!r} cannot be blocked by {parent_id!r}, the ticket it fixes"
            )

    waiting = transition(
        replace(
            parent,
            blocked_by=(*parent.blocked_by, *(bug.id for bug in incoming)),
            review_round=parent.review_round + 1,
        ),
        Status.PENDING,
    )
    filed = _replaced(with_tickets(run, incoming), waiting)
    check(filed)
    return filed


def with_halt(run: Run, halt: Halt | None) -> Run:
    """`run` stopped for the reason `halt` gives, or carrying on when it is `None`."""
    return replace(run, halt=halt)


def with_base_sha(run: Run, ticket_id: str, sha: str) -> Run:
    """`run` with one ticket remembering where its branch stood before its work."""
    # `base_sha` is taken once, in the same synchronous step that opens the worktree.
    return _replaced(run, replace(run.ticket(ticket_id), base_sha=sha))


def _replaced(run: Run, ticket: Ticket) -> Run:
    """`run` with `ticket` in the place the ticket of that id already had."""
    return replace(
        run,
        tickets=tuple(ticket if held.id == ticket.id else held for held in run.tickets),
    )


# -- derivations ----------------------------------------------------------

# What graph state a status implies.
_NODE_STATE: dict[Status, NodeState] = {
    Status.PENDING: NodeState.PENDING,
    Status.IN_PROGRESS: NodeState.CLAIMED,
    Status.IN_REVIEW: NodeState.CLAIMED,
    Status.MERGING: NodeState.CLAIMED,
    Status.AWAITING_INPUT: NodeState.CLAIMED,
    Status.MERGED: NodeState.DONE,
}


def dag_of(run: Run) -> Dag:
    """The run's dependency graph, built fresh and thrown away by the caller.

    A node per ticket at the state its status implies, then an edge per
    `blocked_by` entry. Nothing is kept between calls.
    """
    dag = Dag(priority=bugs_first(run))
    for ticket in run.tickets:
        dag.add_node(ticket.id, _NODE_STATE[ticket.status])
    for ticket in run.tickets:
        for blocker in ticket.blocked_by:
            dag.add_edge(ticket.id, blocker)
    return dag


def bugs_first(run: Run) -> Callable[[NodeId], bool]:
    """A `Dag` priority key that puts every ready bug ahead of every ready feature.

    `Dag.ready()` sorts stably, so ties keep insertion order for free.
    """
    bugs = {ticket.id: ticket.is_bug for ticket in run.tickets}

    def priority(node_id: NodeId) -> bool:
        return not bugs[node_id]

    return priority


def display_order(run: Run) -> tuple[str, ...]:
    """Every ticket in insertion order, each bug directly under its parent.

    A bug whose parent is not in the run — which `check` refuses — keeps its
    own place rather than disappearing.
    """
    children: dict[str, list[str]] = {}
    held = {ticket.id for ticket in run.tickets}
    for ticket in run.tickets:
        if ticket.parent is not None and ticket.parent in held:
            children.setdefault(ticket.parent, []).append(ticket.id)

    order: list[str] = []

    def emit(ticket_id: str) -> None:
        order.append(ticket_id)
        for child in children.get(ticket_id, ()):
            emit(child)

    for ticket in run.tickets:
        if ticket.parent is None or ticket.parent not in held:
            emit(ticket.id)
    return tuple(order)


# -- validation -----------------------------------------------------------


def check(run: Run) -> None:
    """Raise `InvalidStateError` unless `run` is a state this workflow could reach.

    A hand-edited document is a supported input, so the message names the
    ticket and the field a person has to go and fix.
    """
    seen: set[str] = set()
    for ticket in run.tickets:
        if ticket.id in seen:
            raise InvalidStateError(f"duplicate ticket id {ticket.id!r}")
        seen.add(ticket.id)
    for ticket in run.tickets:
        for blocker in ticket.blocked_by:
            if blocker not in seen:
                raise InvalidStateError(
                    f"ticket {ticket.id!r} is blocked by {blocker!r}, which is not in the run"
                )
        if ticket.parent is not None and ticket.parent not in seen:
            raise InvalidStateError(
                f"ticket {ticket.id!r} names parent {ticket.parent!r}, which is not in the run"
            )
        if ticket.status is Status.AWAITING_INPUT and ticket.resume_to is None:
            raise InvalidStateError(
                f"ticket {ticket.id!r} is waiting on the user with no status to return to"
            )
    try:
        dag_of(run)
    except CycleError as cycle:
        raise InvalidStateError(f"the tickets block each other in a cycle: {cycle}") from cycle
