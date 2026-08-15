"""The ticket: what one unit of work is, and the states it may pass through.

Layer: workflows. Pure — no I/O, no async, and nothing imported from `agl`. It
is the vocabulary the rest of the ticket workflow is written in, so it has to be
readable on its own.

A `Ticket` holds only what cannot be recomputed. `is_bug` and `first_pass` are
properties rather than fields because a stored copy of something derivable is a
second source of truth waiting to disagree with the first. The branch a ticket's
work lands on is not here at all: it is `paths.branch(label, id)`, computed
where it is needed from two things already known.

The status machine has no `fixing` state. A ticket whose review filed findings
goes back to `PENDING` with `blocked_by` edges to the bug tickets that carry
them; the graph already says "waiting on something", and a second mechanism for
the same fact is a second thing to keep in sync.

`AWAITING_INPUT` is a *suspension* rather than a place in the life cycle: a
ticket enters it from wherever an agent happened to be running and has to return
to exactly there. That memory lives in `resume_to`, written and cleared by
`transition` itself, so no caller can enter the state and forget to record where
it came from.
"""

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

__all__ = [
    "TICKETS_SCHEMA",
    "IllegalTransitionError",
    "InvalidTicketsError",
    "Status",
    "Ticket",
    "can_transition",
    "tickets_from_json",
    "transition",
]


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

    Frozen, because there is one authority for a ticket and it is not this
    object: a ticket is a value read out of the run's state, and moving one
    means writing a new value back where it came from. A mutable ticket invites
    the opposite — a caller holding a snapshot edits it, the run's own copy
    changes underneath everything that indexes by id, and nothing goes through
    the operation that keeps the graph and the board in step. Frozen, a stale
    snapshot is merely stale, which is a bug you can see.

    On a bug ticket — one with a `parent` — `deliverables` are the findings it
    has to fix. Same field, read differently by context, because a second field
    that is empty on every feature ticket would carry no more information.
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

    @property
    def is_bug(self) -> bool:
        """Whether this ticket fixes findings against another one."""
        return self.parent is not None

    @property
    def first_pass(self) -> bool:
        """Whether nobody has reviewed this yet.

        Reads better at the call site than the comparison, and it is what
        decides whether an implementation agent runs at all.
        """
        return self.review_round == 0


class IllegalTransitionError(Exception):
    """Raised when a ticket is asked to move somewhere it cannot go."""


class InvalidTicketsError(Exception):
    """Raised when agent output does not describe a usable set of tickets."""


# -- the life cycle -------------------------------------------------------

# Every status an agent can be running in, and so every status a question can
# interrupt.
_RUNNING = (Status.IN_PROGRESS, Status.IN_REVIEW, Status.MERGING)

# The legal moves, as data. Re-entry (a status to itself) is legal for every
# resting status, which is what lets a caller re-stamp how long a ticket has
# been where it is. `MERGED` is terminal and has no moves at all, and
# `AWAITING_INPUT` has no re-entry because it is a suspension of another status
# rather than one a ticket rests in.
_MOVES: dict[Status, frozenset[Status]] = {
    Status.PENDING: frozenset({Status.PENDING, Status.IN_PROGRESS}),
    Status.IN_PROGRESS: frozenset(
        {Status.IN_PROGRESS, Status.IN_REVIEW, Status.MERGING, Status.AWAITING_INPUT}
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

    Answers about statuses alone, so it cannot know which state a waiting ticket
    was suspended from. `AWAITING_INPUT` -> anything an agent runs in is legal
    here; `transition` narrows that to the one status the ticket actually came
    from.
    """
    return to in _MOVES[frm]


def transition(ticket: Ticket, to: Status) -> Ticket:
    """`ticket` moved to `to`, raising `IllegalTransitionError` on an illegal move.

    Returns a new ticket and leaves the one passed in alone, so the caller is
    what decides where a moved ticket goes — and validates every rule before it
    builds anything, so a raising call has produced nothing to put anywhere.

    Entering `AWAITING_INPUT` records the status being suspended; leaving it
    requires returning to that status and clears the record. A ticket found in
    `AWAITING_INPUT` with nothing recorded cannot move at all, which is the
    right answer: something built it by hand into a state the workflow has no
    way back out of.
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


# -- what the decompose agent produces ------------------------------------

# The same shape `paths.validate_node_id` enforces, spelled out rather than
# imported: this file stays free of `agl.core`, and an id that reaches a branch
# name has to satisfy that function regardless of which one of us checks first.
_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9-]*$"
_ID_RE = re.compile(_ID_PATTERN)

TICKETS_KEY = "tickets"
_REQUIRED = ["id", "title", "deliverables"]
_OPTIONAL = ["blocked_by"]

TICKETS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        TICKETS_KEY: {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "pattern": _ID_PATTERN},
                    "title": {"type": "string", "minLength": 1},
                    "deliverables": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "blocked_by": {
                        "type": "array",
                        "items": {"type": "string", "pattern": _ID_PATTERN},
                    },
                },
                "required": _REQUIRED,
                "additionalProperties": False,
            },
        }
    },
    "required": [TICKETS_KEY],
    "additionalProperties": False,
}
"""The `output_schema` handed to the decompose agent.

It describes what an agent supplies, which is not what a `Ticket` holds:
`status`, `review_round` and `parent` are the orchestrator's, so asking for them
would invite an agent to make them up. `additionalProperties: false` is what
turns a hallucinated field into a validation failure instead of silent data.

Read-only by convention, like `agent.NO_PARAMS` — hand it to an `AgentSpec`
rather than mutating it."""


def tickets_from_json(data: Any) -> tuple[Ticket, ...]:
    """Parse decompose-agent output into tickets, raising `InvalidTicketsError`.

    Re-checks everything `TICKETS_SCHEMA` states, because a schema handed to a
    model is a request rather than a guarantee, plus the two rules JSON schema
    cannot express: ids are unique, and every `blocked_by` names a ticket in the
    same set. The fields an agent does not supply take their defaults —
    `PENDING`, round zero, no parent.
    """
    payload = _object(data, "output")
    _known_fields(payload, [TICKETS_KEY], (), "output")
    raw = payload[TICKETS_KEY]
    if not isinstance(raw, list):
        raise InvalidTicketsError(f"{TICKETS_KEY!r} must be an array, got {_kind(raw)}")
    if not raw:
        raise InvalidTicketsError(f"{TICKETS_KEY!r} is empty: a run needs at least one ticket")

    tickets = tuple(_one_ticket(item, index) for index, item in enumerate(raw))
    _check_ids(tickets)
    return tickets


def _one_ticket(item: Any, index: int) -> Ticket:
    """Build one ticket from one array entry, checking every field it names."""
    where = f"ticket {index}"
    fields = _object(item, where)
    _known_fields(fields, _REQUIRED, _OPTIONAL, where)

    ticket_id = fields["id"]
    if not isinstance(ticket_id, str) or not _ID_RE.fullmatch(ticket_id):
        raise InvalidTicketsError(
            f"{where}: id {ticket_id!r} must be letters, digits and hyphens, "
            "starting with a letter or digit"
        )
    where = f"ticket {ticket_id!r}"

    title = fields["title"]
    if not isinstance(title, str) or not title.strip():
        raise InvalidTicketsError(f"{where}: title must be non-empty text, got {title!r}")

    return Ticket(
        id=ticket_id,
        title=title,
        status=Status.PENDING,
        deliverables=_text_list(fields["deliverables"], "deliverables", where, allow_empty=False),
        blocked_by=_text_list(fields.get("blocked_by", []), "blocked_by", where, allow_empty=True),
    )


def _check_ids(tickets: tuple[Ticket, ...]) -> None:
    """The two rules a JSON schema cannot state: unique ids, resolvable blockers."""
    seen: set[str] = set()
    for ticket in tickets:
        if ticket.id in seen:
            raise InvalidTicketsError(f"duplicate ticket id {ticket.id!r}")
        seen.add(ticket.id)
    for ticket in tickets:
        for blocker in ticket.blocked_by:
            if blocker == ticket.id:
                raise InvalidTicketsError(f"ticket {ticket.id!r} is blocked by itself")
            if blocker not in seen:
                raise InvalidTicketsError(
                    f"ticket {ticket.id!r} is blocked by unknown ticket {blocker!r}"
                )


# -- validation helpers ---------------------------------------------------


def _object(value: Any, where: str) -> dict[str, Any]:
    """Narrow `value` to a JSON object or raise."""
    if not isinstance(value, dict):
        raise InvalidTicketsError(f"{where} must be an object, got {_kind(value)}")
    return value


def _known_fields(
    fields: dict[str, Any], required: list[str], optional: tuple[str, ...] | list[str], where: str
) -> None:
    """Require every `required` key and refuse anything outside the two lists."""
    for name in required:
        if name not in fields:
            raise InvalidTicketsError(f"{where}: missing required field {name!r}")
    allowed = {*required, *optional}
    for name in fields:
        if name not in allowed:
            raise InvalidTicketsError(f"{where}: unknown field {name!r}")


def _text_list(value: Any, name: str, where: str, *, allow_empty: bool) -> tuple[str, ...]:
    """Narrow `value` to a tuple of non-empty strings or raise."""
    if not isinstance(value, list):
        raise InvalidTicketsError(f"{where}: {name} must be an array, got {_kind(value)}")
    if not value and not allow_empty:
        raise InvalidTicketsError(f"{where}: {name} is empty")
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise InvalidTicketsError(f"{where}: every {name} entry must be non-empty text")
    return tuple(value)


def _kind(value: Any) -> str:
    """What something is, for an error message a person has to act on."""
    return type(value).__name__
