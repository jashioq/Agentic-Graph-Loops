"""One run's state, and the operations that move it without letting it drift.

Layer: workflows. Pure — no I/O, no async, and nothing from below but `dag`,
the display's `Board`, and the merge queue's vocabulary of outcomes. Every
method is a plain operation over the two structures a run holds, so a test can
build any situation by hand and assert on it.

`RunState` is execution truth: delete it and the run cannot proceed. The `Board`
it carries is ephemeral and read only by rendering: nothing here ever reads it
back to decide anything, so a run whose board is thrown away still finishes
correctly, you just cannot watch it. It is held here rather than threaded
through every call because the single writer below has to stamp it, and a
caller that has to remember to pass the board is a caller that will forget.

Two structures answer questions about the same tickets. `Dag` holds scheduling
state and `Ticket` holds workflow state; both are true at once, and the
invariant is that they never contradict each other — a `PENDING` node is a
`PENDING` ticket, a `CLAIMED` node is one an agent or the merge queue has
(`IN_PROGRESS`, `IN_REVIEW`, `MERGING`, `AWAITING_INPUT`), and a `DONE` node is
a `MERGED` ticket. `_NODE_STATE` below is that table, and `check_consistent`
enforces it.

The invariant is the one most likely to drift once a scheduler is driving all
this, and here is the cheapest place to catch the drift. `set_status` is the
single writer for every part of it — ticket status, graph state, and the board
stamp — so the three move as one and no caller can update two and forget the
third.

`halt_for` is the other thing here: pure, and message-writing. The merge queue
reports facts about git and the build, and this is where they become the halt a
person reads. It sits with the run that shows it rather than in the runtime,
because "resumable" is a statement about what this run can do about an outcome,
which no queue can know.
"""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from agl.core.command import ExecResult
from agl.runtime.dag import Dag, NodeId, NodeState
from agl.runtime.display import Board
from agl.runtime.merge import MergeOutcome, MergeStatus
from agl.workflows.tickets.models import Status, Ticket, can_transition, transition

__all__ = [
    "TAIL_LINES",
    "DuplicateTicketError",
    "Halt",
    "InconsistentStateError",
    "RunState",
    "UnknownTicketError",
    "bugs_first",
    "halt_for",
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


class RunState:
    """One run's tickets and graph, moved only through the methods below."""

    def __init__(self, label: str, base_branch: str, board: Board) -> None:
        self.label = label
        self.base_branch = base_branch
        self.board = board
        self.tickets: dict[str, Ticket] = {}
        self.dag = Dag(priority=bugs_first(self))
        self.halt: Halt | None = None

    # -- the invariant ----------------------------------------------------

    def check_consistent(self) -> None:
        """Raise `InconsistentStateError` unless the graph and the tickets agree.

        Checks both that the two hold the same ids and that every id's two
        states are the pair the table allows. Cheap enough to call after every
        mutation, and tests do.
        """
        ticket_ids = set(self.tickets)
        node_ids = set(self.dag.nodes())
        for missing in sorted(ticket_ids - node_ids):
            raise InconsistentStateError(f"ticket {missing!r} has no node in the graph")
        for missing in sorted(node_ids - ticket_ids):
            raise InconsistentStateError(f"node {missing!r} has no ticket")
        for ticket_id, ticket in self.tickets.items():
            expected = _NODE_STATE[ticket.status]
            actual = self.dag.state(ticket_id)
            if actual is not expected:
                raise InconsistentStateError(
                    f"ticket {ticket_id!r} is {ticket.status.value} but its node is "
                    f"{actual.value}, expected {expected.value}"
                )

    def is_halted(self) -> bool:
        """The predicate the scheduler asks before admitting more work."""
        return self.halt is not None

    # -- operations -------------------------------------------------------

    def add(self, tickets: Sequence[Ticket], *, now: float | None = None) -> None:
        """Put `tickets` into the run and build the graph edges from `blocked_by`.

        Every id has to be new and every blocker has to resolve — to a ticket
        already in the run or to another one in this batch, so a batch may name
        its blockers in any order. Validates before it touches anything, and
        unwinds if the graph refuses an edge, so a raising call leaves the run
        as it was.
        """
        incoming = tuple(tickets)
        self._check_new(incoming)

        added: list[str] = []
        try:
            for ticket in incoming:
                self.tickets[ticket.id] = ticket
                self.dag.add_node(ticket.id)
                added.append(ticket.id)
            for ticket in incoming:
                for blocker in ticket.blocked_by:
                    self.dag.add_edge(ticket.id, blocker)
        except Exception:
            # An edge can still be refused for a reason no up-front check can
            # see — a cycle within the batch. The graph unwinds its own failed
            # mutation; this unwinds the ones that already succeeded.
            self._drop(added)
            raise

        for ticket in incoming:
            self.board.stamp(ticket.id, now)

    def set_status(self, ticket_id: str, status: Status, *, now: float | None = None) -> None:
        """Move a ticket to `status`, moving the graph and the stamp with it.

        The single writer for all three, so they cannot drift apart. The graph
        move is whatever the table demands and nothing when the status stays on
        the same row — a question and its answer both sit on `CLAIMED`, so
        asking one does not give the ticket back to the scheduler.

        Raises `UnknownTicketError` for an id the run does not hold,
        `IllegalTransitionError` for a move the life cycle forbids, and
        `ValueError` from the graph when a ticket is started while something
        still blocks it. Nothing is written in any of those cases.
        """
        ticket = self._ticket(ticket_id)
        if not can_transition(ticket.status, status):
            # `transition` owns the message. It cannot mutate on a move
            # `can_transition` has already refused, so this only ever raises.
            transition(ticket, status)
        self._move_node(ticket_id, status)
        # The one move left that can be refused here is a return from
        # `AWAITING_INPUT` into a status it was not suspended from, and that
        # pairing never moves the node, so there is nothing to undo.
        transition(ticket, status)
        self.board.stamp(ticket_id, now)

    def file_bugs(
        self, parent_id: str, bugs: Sequence[Ticket], *, now: float | None = None
    ) -> None:
        """Fold review findings into a running graph as work the parent waits on.

        The runtime mutation, and the ordering is load-bearing: **add the nodes,
        add the edges, then release the parent.** Releasing first leaves the
        parent ready for a beat, and a scheduler that looks in that window
        claims it out from under the very work meant to block it. `Dag`'s module
        docstring documents this; the last step here is the release.

        Every bug has to name `parent_id` as its parent, and none may be blocked
        by it — a bug waiting on the ticket that is waiting on the bug is the one
        way this graph can cycle. A bug can still reach the parent the long way
        round, through some other ticket that already waits on it, and the graph
        refuses that edge; the run unwinds to where it was and the finding is
        not lost.
        """
        self._ticket(parent_id)  # an unknown parent fails before anything is built
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

        self.add(incoming, now=now)
        try:
            for bug in incoming:
                self.dag.add_edge(parent_id, bug.id)
        except Exception:
            self._drop([bug.id for bug in incoming])
            raise

        # Last, and only now that the bugs are in place to hold it back.
        self.set_status(parent_id, Status.PENDING, now=now)

    @contextmanager
    def awaiting(self, ticket_id: str) -> Iterator[None]:
        """Suspend a ticket into `AWAITING_INPUT` for the length of the block.

        Symmetric by construction: whatever the ticket was doing is what it
        goes back to, so its status history reads as a straight line with one
        detour rather than a fork. The resume is in a `finally` because a
        question that failed still leaves a ticket that is no longer waiting on
        anybody.
        """
        was = self._ticket(ticket_id).status
        self.set_status(ticket_id, Status.AWAITING_INPUT)
        try:
            yield
        finally:
            self.set_status(ticket_id, was)

    def display_order(self) -> tuple[str, ...]:
        """Every ticket in insertion order, each bug directly under its parent.

        Pure, and recomputed rather than stored: a stored order is one more
        thing to keep in sync with `file_bugs`, and this is deterministic from
        what the run already holds. A bug whose parent is not in the run — which
        nothing here can produce — keeps its own place rather than disappearing.
        """
        children: dict[str, list[str]] = {}
        for ticket_id, ticket in self.tickets.items():
            if ticket.parent is not None and ticket.parent in self.tickets:
                children.setdefault(ticket.parent, []).append(ticket_id)

        order: list[str] = []

        def emit(ticket_id: str) -> None:
            order.append(ticket_id)
            for child in children.get(ticket_id, ()):
                emit(child)

        for ticket_id, ticket in self.tickets.items():
            if ticket.parent is None or ticket.parent not in self.tickets:
                emit(ticket_id)
        return tuple(order)

    # -- internals --------------------------------------------------------

    def _ticket(self, ticket_id: str) -> Ticket:
        """The ticket, or `UnknownTicketError`."""
        ticket = self.tickets.get(ticket_id)
        if ticket is None:
            raise UnknownTicketError(ticket_id)
        return ticket

    def _check_new(self, incoming: tuple[Ticket, ...]) -> None:
        """Everything about a batch that can be decided before touching anything."""
        ids: set[str] = set()
        for ticket in incoming:
            if ticket.id in self.tickets or ticket.id in ids:
                raise DuplicateTicketError(ticket.id)
            if ticket.status is not Status.PENDING:
                raise InconsistentStateError(
                    f"ticket {ticket.id!r} joins the run as {ticket.status.value}, "
                    "but a new ticket is pending"
                )
            ids.add(ticket.id)
        known = set(self.tickets) | ids
        for ticket in incoming:
            for blocker in ticket.blocked_by:
                if blocker not in known:
                    raise UnknownTicketError(blocker)

    def _drop(self, ticket_ids: Sequence[str]) -> None:
        """Take tickets back out of the run, undoing an addition a later step refused.

        What the board holds goes too: a stamp for a ticket that is no longer in
        the run is a row rendering would have nothing to draw.
        """
        for ticket_id in ticket_ids:
            self.dag.remove_node(ticket_id)
            del self.tickets[ticket_id]
            self.board.drop(ticket_id)

    def _move_node(self, ticket_id: str, status: Status) -> None:
        """Put the graph node where `status` says it belongs, or leave it alone."""
        target = _NODE_STATE[status]
        current = self.dag.state(ticket_id)
        if current is target:
            return
        if target is NodeState.CLAIMED:
            self.dag.claim(ticket_id)
        elif target is NodeState.PENDING:
            self.dag.release(ticket_id)
        else:
            self.dag.complete(ticket_id)


def bugs_first(state: RunState) -> Callable[[NodeId], bool]:
    """A `Dag` priority key that puts every ready bug ahead of every ready feature.

    Ticket knowledge, so it lives here rather than in the scheduler: what the
    graph is asked for is a key over node ids, and only this run knows which of
    its nodes are bugs.

    `Dag.ready()` sorts with this key using a stable sort, so ties — bug vs
    bug, feature vs feature — keep insertion order for free; the key only has
    to say which of the two groups a node belongs to.

    A run that keeps generating bugs can leave feature tickets waiting a long
    time even though they became ready first. That is intended: finishing
    what is already open takes priority over opening more.
    """

    def priority(node_id: NodeId) -> bool:
        return not state.tickets[node_id].is_bug

    return priority


# -- what a merge outcome means to this run --------------------------------


TAIL_LINES = 20
"""How many lines of a failed build's output a halt carries.

A display choice, not a fact about builds. Which slice of the output matters is
language-specific — a Kotlin error sits early under a stack trace, a Rust one is
structured and late, a bundler dumps module paths — which is why the outcome
carries the result whole and the truncation lives here, next to the banner it
is being truncated for.
"""


def halt_for(outcome: MergeOutcome) -> Halt:
    """The halt this run shows for a merge that did not land.

    Resumable when a person editing the repository changes the answer — a
    conflict to resolve, a build to fix. Not resumable when nothing they can do
    to the repository would: git refusing a branch outright, or an exception
    that escaped a callable which closed over its broken state before the run
    started. Those say restart the process, because that is what would help.
    """
    if outcome.status is MergeStatus.CONFLICT:
        return Halt(
            reason=f"{outcome.key} conflicts with the base branch",
            detail=f"resolve in the repository root: {', '.join(outcome.conflicted)}",
        )
    if outcome.status is MergeStatus.BUILD_FAILED and outcome.build is not None:
        build = outcome.build
        what = "timed out" if build.timed_out else f"failed with exit {build.code}"
        return Halt(
            reason=f"{outcome.key} merged but the build {what}",
            detail=_tail(_output(build), TAIL_LINES),
        )
    if outcome.status is MergeStatus.VCS_ERROR:
        return Halt(f"{outcome.key} cannot be merged", outcome.error, resumable=False)
    return Halt(
        reason=f"{outcome.key} could not be processed: {outcome.error}",
        detail=outcome.error,
        resumable=False,
    )


def _output(result: ExecResult) -> str:
    """Both streams as one text, since which of them carries the diagnosis is
    the build tool's choice and not something this run can know."""
    return "\n".join(stream.strip("\n") for stream in (result.stdout, result.stderr) if stream)


def _tail(text: str, lines: int) -> str:
    """The last `lines` lines of `text`, and nothing about which ones matter."""
    kept = text.strip("\n").split("\n")[-lines:]
    return "\n".join(kept)
