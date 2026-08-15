"""One run's state as a value, and the pure transitions that move it.

Layer: workflows. Pure — no I/O, no async, and nothing from below but `dag` and
the merge queue's vocabulary of outcomes. Every function here is a plain
operation over a `Run`, so a test can build any situation by hand and assert on
it, and so a caller can decide what to do with a new state before committing to
it.

`Run` is a frozen snapshot: the tickets, and the halt. Nothing moves in place.
Every transition takes a `Run` and returns a new one, which is what makes the
document in `snapshot.py` the only writer and read-modify-write the only shape a
mutation can take. The label and the base branch are deliberately *not* here —
they are what a run was asked for, they live in `run.json`, and they reach every
use site through `RunContext`.

**Everything derivable is derived.** The graph is not stored; `dag_of` builds one
from `blocked_by` plus the status table, and nothing keeps it, so a graph cannot
contradict the tickets it came from. `blocked_by` *is* the graph: `with_bugs`
records its parent→bug edges by appending the bug ids to the parent's blockers,
and there is no second structure to keep in step. Display order is recomputed
too, for the same reason.

`check` is the other half of that trade. With the state in a document a person
can open and edit, the shapes that used to be impossible by construction are now
merely unwritten, so they are checked instead — once, at the point a document
becomes a `Run`, with a message naming what is wrong rather than halfway through
a run that has already started worktrees.

`halt_for` is the last thing here: pure, and message-writing. The merge queue
reports facts about git and the build, and this is where they become the halt a
person reads. It sits with the run that shows it rather than in the runtime,
because "resumable" is a statement about what this run can do about an outcome,
which no queue can know.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from agl.core.command import ExecResult
from agl.runtime.dag import CycleError, Dag, NodeId, NodeState
from agl.runtime.merge import MergeOutcome, MergeStatus
from agl.workflows.tickets.models import Status, Ticket, transition

__all__ = [
    "TAIL_LINES",
    "DuplicateTicketError",
    "Halt",
    "InvalidStateError",
    "Run",
    "UnknownTicketError",
    "bugs_first",
    "check",
    "dag_of",
    "display_order",
    "halt_for",
    "with_base_sha",
    "with_bugs",
    "with_halt",
    "with_status",
    "with_tickets",
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


class InvalidStateError(Exception):
    """Raised when a `Run` does not describe a state this workflow could reach."""


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


@dataclass(frozen=True)
class Run:
    """One run's whole state, as a value. Every change produces a new one.

    Tickets are a tuple in insertion order rather than a mapping, because the
    order is itself state — it is what `display_order` and the graph's node
    ordering are built from, and a document round trip has to preserve it.
    Lookup by id is a scan over a list that is tens of entries long at most.
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

    Every id has to be new and every blocker has to resolve — to a ticket
    already in the run or to another one in this batch, so a batch may name its
    blockers in any order. A new ticket joins `PENDING`, because nothing has
    happened to it yet, and `check` at the end is what catches the one thing no
    up-front pass can see: a cycle closed within the batch.
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

    Raises `UnknownTicketError` for an id the run does not hold and
    `IllegalTransitionError` for a move the life cycle forbids. Nothing is built
    in either case, which is the freedom a frozen ticket buys.
    """
    return _replaced(run, transition(run.ticket(ticket_id), status))


def with_bugs(run: Run, parent_id: str, bugs: Sequence[Ticket]) -> Run:
    """Fold review findings into the run as work the parent now waits on.

    One function because it is one write. The old graph mutation had a
    load-bearing ordering — add the nodes, add the edges, *then* release the
    parent, or a scheduler looking in the gap claims it out from under the very
    work meant to block it — and that ordering is invisible here: a reader only
    ever sees the whole result, because the whole result is what gets written.

    The parent's `review_round` advances here too, because the review that
    produced these findings is the review that just finished. Bumping it
    anywhere else would leave a window in which the parent is back in the graph
    carrying a round it has completed, and the next round's findings would be
    written over the last round's documents.

    Every bug has to name `parent_id` as its parent, and none may be blocked by
    it — a bug waiting on the ticket that is waiting on the bug is the one way
    this graph can cycle. A bug can still reach the parent the long way round,
    through some other ticket that already waits on it, and `check` refuses
    that; the caller is left holding the run it had.
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
    """`run` with one ticket remembering where its branch stood before its work.

    Written in the same synchronous step that opens the ticket's worktree, and
    never again: re-taking it after a commit has landed would erase the very
    difference it exists to measure.
    """
    return _replaced(run, replace(run.ticket(ticket_id), base_sha=sha))


def _replaced(run: Run, ticket: Ticket) -> Run:
    """`run` with `ticket` in the place the ticket of that id already had."""
    return replace(
        run,
        tickets=tuple(ticket if held.id == ticket.id else held for held in run.tickets),
    )


# -- derivations ----------------------------------------------------------

# What graph state a status implies. Read the other way it is why no invariant
# is enforced any more: with the graph derived through this table on every
# question, it cannot disagree with the tickets it was built from.
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

    A node per ticket, stated at the state its status implies, then an edge per
    `blocked_by` entry. Nothing is kept between calls, so there is no graph to
    fall out of step with the tickets — the cost is rebuilding one per question,
    which is a pass over a few dozen strings.
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

    Ticket knowledge, so it lives here rather than in the scheduler: what the
    graph is asked for is a key over node ids, and only this run knows which of
    its nodes are bugs.

    `Dag.ready()` sorts with this key using a stable sort, so ties — bug vs bug,
    feature vs feature — keep insertion order for free; the key only has to say
    which of the two groups a node belongs to.

    A run that keeps generating bugs can leave feature tickets waiting a long
    time even though they became ready first. That is intended: finishing what
    is already open takes priority over opening more.
    """
    bugs = {ticket.id: ticket.is_bug for ticket in run.tickets}

    def priority(node_id: NodeId) -> bool:
        return not bugs[node_id]

    return priority


def display_order(run: Run) -> tuple[str, ...]:
    """Every ticket in insertion order, each bug directly under its parent.

    Pure, and recomputed rather than stored: a stored order is one more thing to
    keep in sync with `with_bugs`, and this is deterministic from what the run
    already holds. A bug whose parent is not in the run — which `check` refuses
    — keeps its own place rather than disappearing.
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

    A hand-edited document is a supported input, so every shape that used to be
    impossible by construction is checked here instead, and the message names
    the ticket and the field a person has to go and fix. Called on every
    transition and at the end of every parse, which is cheap: it is one pass
    plus a graph build.
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
