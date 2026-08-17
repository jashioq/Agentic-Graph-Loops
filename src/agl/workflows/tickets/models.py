"""The ticket: what one unit of work is, and the states it may pass through.

Layer: workflows. Pure vocabulary — no I/O, no async, and nothing from `agl`
but this workflow's `errors`. What an agent produces is not here: the decompose
schema and its parser live in `documents/tickets_document.py`.
"""

from dataclasses import dataclass, replace
from enum import Enum

from agl.workflows.tickets.errors import IllegalTransitionError

__all__ = ["Status", "Ticket", "can_transition", "transition"]


class Status(Enum):
    """Where a ticket is in the workflow."""

    PENDING = "pending"  # not started, or waiting on blockers
    IN_PROGRESS = "in_progress"  # an implementation or bug-fix agent is running
    IN_REVIEW = "in_review"  # reviewers are running
    MERGING = "merging"  # in the merge queue, or being merged
    MERGED = "merged"  # merged and the build passed — terminal
    AWAITING_INPUT = "awaiting_input"  # an agent asked, and is blocked on the user


@dataclass(frozen=True)
class Ticket:
    """One unit of work: what to build, what it waits for, and where it is.

    On a bug ticket — one with a `parent` — `deliverables` are the findings it
    has to fix.
    """

    id: str
    title: str
    status: Status
    deliverables: tuple[str, ...]
    blocked_by: tuple[str, ...] = ()
    parent: str | None = None  # non-None ⇒ this is a bug ticket
    review_round: int = 0
    # Only ever non-None while `status is AWAITING_INPUT`; see `transition`.
    resume_to: "Status | None" = None
    # `base_sha` is taken once, in the same synchronous step that opens the worktree.
    base_sha: str | None = None

    @property
    def is_bug(self) -> bool:
        """Whether this ticket fixes findings against another one."""
        return self.parent is not None


# -- the life cycle -------------------------------------------------------

# Every status an agent can be running in, and so every status a question can
# interrupt.
_RUNNING = (Status.IN_PROGRESS, Status.IN_REVIEW, Status.MERGING)

# The legal moves, as data. Re-entry is legal for every resting status; `MERGED`
# is terminal; `AWAITING_INPUT` suspends a status rather than being one, so it
# has no re-entry. `PENDING` is every claimed status' way in and out.
_MOVES: dict[Status, frozenset[Status]] = {
    Status.PENDING: frozenset(
        {
            Status.PENDING,
            Status.IN_PROGRESS,
            Status.IN_REVIEW,
            Status.MERGING,
            Status.MERGED,
        }
    ),
    Status.IN_PROGRESS: frozenset(
        {
            Status.IN_PROGRESS,
            Status.IN_REVIEW,
            Status.MERGING,
            Status.PENDING,
            Status.AWAITING_INPUT,
        }
    ),
    Status.IN_REVIEW: frozenset(
        {Status.IN_REVIEW, Status.PENDING, Status.MERGING, Status.AWAITING_INPUT}
    ),
    Status.MERGING: frozenset(
        {Status.MERGING, Status.MERGED, Status.PENDING, Status.AWAITING_INPUT}
    ),
    Status.MERGED: frozenset(),
    Status.AWAITING_INPUT: frozenset(_RUNNING),
}


def can_transition(frm: Status, to: Status) -> bool:
    """Whether `frm` -> `to` is a legal move for some ticket.

    Answers about statuses alone, so `AWAITING_INPUT` -> anything an agent runs
    in is legal here; `transition` narrows that to the one status the ticket
    actually came from.
    """
    return to in _MOVES[frm]


def transition(ticket: Ticket, to: Status) -> Ticket:
    """`ticket` moved to `to`, raising `IllegalTransitionError` on an illegal move.

    Returns a new ticket and leaves the one passed in alone. Entering
    `AWAITING_INPUT` records the status being suspended; leaving it requires
    returning to that status, and clears the record.
    """
    frm = ticket.status
    if not can_transition(frm, to):
        raise IllegalTransitionError(f"{ticket.id}: cannot move from {frm.value} to {to.value}")
    resume_to: Status | None = None
    if frm is Status.AWAITING_INPUT:
        if ticket.resume_to is None:
            raise IllegalTransitionError(
                f"{ticket.id}: waiting on the user with no recorded status to return to"
            )
        if to is not ticket.resume_to:
            raise IllegalTransitionError(
                f"{ticket.id}: waiting on the user must return to "
                f"{ticket.resume_to.value}, not {to.value}"
            )
    elif to is Status.AWAITING_INPUT:
        resume_to = frm
    return replace(ticket, status=to, resume_to=resume_to)
