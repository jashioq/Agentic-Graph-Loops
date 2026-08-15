"""One run's state, and the operations that move it without letting it drift.

Layer: workflows. Pure — no I/O, no async, and nothing from `agl.core` but
`dag`. Everything here is a plain function over the two dataclasses, so a test
can build any situation by hand and assert on it.

The two dataclasses are split by **lifetime**, not by logic against display.
`RunState` is execution truth: delete it and the run cannot proceed. `Live` is
ephemeral and in-memory, read only by rendering: delete it and the run still
finishes correctly, you just cannot watch it. Nothing in `Live` may ever be read
to make a decision, which is what keeps that difference real rather than
aspirational.

Two structures answer questions about the same tickets. `Dag` holds scheduling
state and `Ticket` holds workflow state; both are true at once, and the
invariant is that they never contradict each other — a `PENDING` node is a
`PENDING` ticket, a `CLAIMED` node is one an agent or the merge queue has
(`IN_PROGRESS`, `IN_REVIEW`, `MERGING`, `AWAITING_INPUT`), and a `DONE` node is
a `MERGED` ticket. `_NODE_STATE` below is that table, and `check_consistent`
enforces it.

The invariant is the one most likely to drift once a scheduler is driving all
this, and here is the cheapest place to catch the drift. `set_status` is the
single writer for every part of it — ticket status, graph state, and the `Live`
stamp — so the three move as one and no caller can update two and forget the
third.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from agl.runtime.dag import Dag, NodeState
from agl.workflows.tickets.models import Status, Ticket, can_transition, transition

__all__ = [
    "DuplicateTicketError",
    "Halt",
    "InconsistentStateError",
    "Live",
    "RunState",
    "UnknownTicketError",
    "add_tickets",
    "check_consistent",
    "display_order",
    "file_bugs",
    "set_status",
]


@dataclass(frozen=True)
class Halt:
    """Why a run stopped early, in the words the user is going to read.

    `resumable` is whether pressing enter can plausibly help. A merge conflict
    or a failing build are true of the repository right now, and a person
    fixing the repository changes the answer. An exception that escaped the
    workflow closed over broken state before the run started — a build
    callable pointed at a command that does not exist, say — and no amount of
    editing the repository changes what that callable does; the process has to
    restart. Defaults to `True` because most halts are the resumable kind.
    """

    reason: str
    detail: str = ""
    resumable: bool = True


@dataclass
class RunState:
    """Execution truth. What the workflow reads to decide what to do."""

    label: str
    base_branch: str
    dag: Dag
    tickets: dict[str, Ticket]
    halt: Halt | None = None


@dataclass
class Live:
    """Ephemeral. In-memory only, read only by rendering.

    Everything here is about watching a run rather than running it: when each
    ticket last changed status, and the one-line activity an agent last
    reported. A run with no `Live` at all produces the same `RunState`.

    Two clocks, two meanings. `started_at` covers the whole session — set the
    moment the run begins, before the interview has asked its first question —
    and is what a session header's timer reads: a liveness signal for the
    agent currently working. `approved_at` is set later, at ticket approval,
    and is what the dashboard footer reads: how long the implementation loop
    has been going. Ten minutes spent answering interview questions must not
    read as ten minutes of run time, so the two are independent stamps rather
    than one derived from the other. `None` until approval happens.
    """

    started_at: float
    approved_at: float | None = None
    status_since: dict[str, float] = field(default_factory=dict)
    activity: dict[str, str] = field(default_factory=dict)


class InconsistentStateError(Exception):
    """Raised when the graph and the tickets disagree about the same ticket."""


class UnknownTicketError(Exception):
    """Raised when an operation names a ticket the run does not hold."""

    def __init__(self, ticket_id: str) -> None:
        super().__init__(ticket_id)
        self.ticket_id = ticket_id


class DuplicateTicketError(Exception):
    """Raised when a ticket would take an id the run already uses."""

    def __init__(self, ticket_id: str) -> None:
        super().__init__(f"ticket {ticket_id!r} is already in the run")
        self.ticket_id = ticket_id


# The invariant, as data. Read one way it says what graph state a status
# implies; read the other it is the table `check_consistent` enforces.
_NODE_STATE: dict[Status, NodeState] = {
    Status.PENDING: NodeState.PENDING,
    Status.IN_PROGRESS: NodeState.CLAIMED,
    Status.IN_REVIEW: NodeState.CLAIMED,
    Status.MERGING: NodeState.CLAIMED,
    Status.AWAITING_INPUT: NodeState.CLAIMED,
    Status.MERGED: NodeState.DONE,
}


# -- the invariant --------------------------------------------------------


def check_consistent(state: RunState) -> None:
    """Raise `InconsistentStateError` unless the graph and the tickets agree.

    Checks both that the two hold the same ids and that every id's two states
    are the pair the table allows. Cheap enough to call after every mutation,
    and tests do.
    """
    ticket_ids = set(state.tickets)
    node_ids = set(state.dag.nodes())
    for missing in sorted(ticket_ids - node_ids):
        raise InconsistentStateError(f"ticket {missing!r} has no node in the graph")
    for missing in sorted(node_ids - ticket_ids):
        raise InconsistentStateError(f"node {missing!r} has no ticket")
    for ticket_id, ticket in state.tickets.items():
        expected = _NODE_STATE[ticket.status]
        actual = state.dag.state(ticket_id)
        if actual is not expected:
            raise InconsistentStateError(
                f"ticket {ticket_id!r} is {ticket.status.value} but its node is "
                f"{actual.value}, expected {expected.value}"
            )


# -- operations -----------------------------------------------------------


def add_tickets(
    state: RunState,
    live: Live | None,
    tickets: Sequence[Ticket],
    *,
    now: float | None = None,
) -> None:
    """Put `tickets` into the run and build the graph edges from `blocked_by`.

    Every id has to be new and every blocker has to resolve — to a ticket
    already in the run or to another one in this batch, so a batch may name its
    blockers in any order. Validates before it touches anything, and unwinds if
    the graph refuses an edge, so a raising call leaves the run as it was.
    """
    incoming = tuple(tickets)
    _check_new(state, incoming)

    added: list[str] = []
    try:
        for ticket in incoming:
            state.tickets[ticket.id] = ticket
            state.dag.add_node(ticket.id)
            added.append(ticket.id)
        for ticket in incoming:
            for blocker in ticket.blocked_by:
                state.dag.add_edge(ticket.id, blocker)
    except Exception:
        # An edge can still be refused for a reason no up-front check can see —
        # a cycle within the batch. The graph unwinds its own failed mutation;
        # this unwinds the ones that already succeeded.
        _drop(state, live, added)
        raise

    for ticket in incoming:
        _stamp(live, ticket.id, now)


def set_status(
    state: RunState,
    live: Live | None,
    ticket_id: str,
    status: Status,
    *,
    now: float | None = None,
) -> None:
    """Move a ticket to `status`, moving the graph and the stamp with it.

    The single writer for all three, so they cannot drift apart. The graph move
    is whatever the table demands and nothing when the status stays on the same
    row — a question and its answer both sit on `CLAIMED`, so asking one does
    not give the ticket back to the scheduler.

    Raises `UnknownTicketError` for an id the run does not hold,
    `IllegalTransitionError` for a move the life cycle forbids, and `ValueError`
    from the graph when a ticket is started while something still blocks it.
    Nothing is written in any of those cases.
    """
    ticket = _ticket(state, ticket_id)
    if not can_transition(ticket.status, status):
        # `transition` owns the message. It cannot mutate on a move
        # `can_transition` has already refused, so this only ever raises.
        transition(ticket, status)
    _move_node(state, ticket_id, status)
    # The one move left that can be refused here is a return from
    # `AWAITING_INPUT` into a status it was not suspended from, and that pairing
    # never moves the node, so there is nothing to undo.
    transition(ticket, status)
    _stamp(live, ticket_id, now)


def file_bugs(
    state: RunState,
    live: Live | None,
    parent_id: str,
    bugs: Sequence[Ticket],
    *,
    now: float | None = None,
) -> None:
    """Fold review findings into a running graph as work the parent waits on.

    The runtime mutation, and the ordering is load-bearing: **add the nodes, add
    the edges, then release the parent.** Releasing first leaves the parent
    ready for a beat, and a scheduler that looks in that window claims it out
    from under the very work meant to block it. `Dag`'s module docstring
    documents this; the last step here is the release.

    Every bug has to name `parent_id` as its parent, and none may be blocked by
    it — a bug waiting on the ticket that is waiting on the bug is the one way
    this graph can cycle. A bug can still reach the parent the long way round,
    through some other ticket that already waits on it, and the graph refuses
    that edge; the run unwinds to where it was and the finding is not lost.
    """
    _ticket(state, parent_id)  # an unknown parent fails before anything is built
    incoming = tuple(bugs)
    if not incoming:
        raise ValueError(f"no bugs to file against {parent_id!r}")
    for bug in incoming:
        if bug.parent != parent_id:
            raise ValueError(
                f"bug {bug.id!r} names parent {bug.parent!r}, not {parent_id!r}"
            )
        if parent_id in bug.blocked_by:
            raise ValueError(
                f"bug {bug.id!r} cannot be blocked by {parent_id!r}, the ticket it fixes"
            )

    add_tickets(state, live, incoming, now=now)
    try:
        for bug in incoming:
            state.dag.add_edge(parent_id, bug.id)
    except Exception:
        _drop(state, live, [bug.id for bug in incoming])
        raise

    # Last, and only now that the bugs are in place to hold it back.
    set_status(state, live, parent_id, Status.PENDING, now=now)


def display_order(state: RunState) -> tuple[str, ...]:
    """Every ticket in insertion order, each bug directly under its parent.

    Pure, and recomputed rather than stored: a stored order is one more thing to
    keep in sync with `file_bugs`, and this is deterministic from what the run
    already holds. A bug whose parent is not in the run — which nothing here can
    produce — keeps its own place rather than disappearing.
    """
    children: dict[str, list[str]] = {}
    for ticket_id, ticket in state.tickets.items():
        if ticket.parent is not None and ticket.parent in state.tickets:
            children.setdefault(ticket.parent, []).append(ticket_id)

    order: list[str] = []

    def emit(ticket_id: str) -> None:
        order.append(ticket_id)
        for child in children.get(ticket_id, ()):
            emit(child)

    for ticket_id, ticket in state.tickets.items():
        if ticket.parent is None or ticket.parent not in state.tickets:
            emit(ticket_id)
    return tuple(order)


# -- internals ------------------------------------------------------------


def _ticket(state: RunState, ticket_id: str) -> Ticket:
    """The ticket, or `UnknownTicketError`."""
    ticket = state.tickets.get(ticket_id)
    if ticket is None:
        raise UnknownTicketError(ticket_id)
    return ticket


def _check_new(state: RunState, incoming: tuple[Ticket, ...]) -> None:
    """Everything about a batch that can be decided before touching anything."""
    ids: set[str] = set()
    for ticket in incoming:
        if ticket.id in state.tickets or ticket.id in ids:
            raise DuplicateTicketError(ticket.id)
        if ticket.status is not Status.PENDING:
            raise InconsistentStateError(
                f"ticket {ticket.id!r} joins the run as {ticket.status.value}, "
                "but a new ticket is pending"
            )
        ids.add(ticket.id)
    known = set(state.tickets) | ids
    for ticket in incoming:
        for blocker in ticket.blocked_by:
            if blocker not in known:
                raise UnknownTicketError(blocker)


def _drop(state: RunState, live: Live | None, ticket_ids: Sequence[str]) -> None:
    """Take tickets back out of the run, undoing an addition a later step refused.

    The stamps go too. `Live` is display-only, but a stamp for a ticket that is
    no longer in the run is a row rendering would have nothing to draw.
    """
    for ticket_id in ticket_ids:
        state.dag.remove_node(ticket_id)
        del state.tickets[ticket_id]
        if live is not None:
            live.status_since.pop(ticket_id, None)


def _move_node(state: RunState, ticket_id: str, status: Status) -> None:
    """Put the graph node where `status` says it belongs, or leave it alone."""
    target = _NODE_STATE[status]
    current = state.dag.state(ticket_id)
    if current is target:
        return
    if target is NodeState.CLAIMED:
        state.dag.claim(ticket_id)
    elif target is NodeState.PENDING:
        state.dag.release(ticket_id)
    else:
        state.dag.complete(ticket_id)


def _stamp(live: Live | None, ticket_id: str, now: float | None) -> None:
    """Record when a ticket arrived where it is, if anyone is watching.

    `now` is explicit so a test can assert on the value; the default reads the
    same monotonic clock `Live.started_at` is taken from, so a workflow does not
    have to thread one through.
    """
    if live is None:
        return
    live.status_since[ticket_id] = time.monotonic() if now is None else now
