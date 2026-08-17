"""Tickets and their life cycle: what a ticket is, and where it may move to."""

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from agl.workflows.tickets.errors import IllegalTransitionError
from agl.workflows.tickets.models import Status, Ticket, can_transition, transition

# -- fixtures as data -----------------------------------------------------


def ticket(**overrides: Any) -> Ticket:
    """A feature ticket in `PENDING`, with any field overridable."""
    fields: dict[str, Any] = {
        "id": "T-01",
        "title": "A thing",
        "status": Status.PENDING,
        "deliverables": ("a file",),
    }
    fields.update(overrides)
    return Ticket(**fields)


# The life cycle, written out rather than derived from the table under test:
# a table that agreed with itself would prove nothing.
LEGAL = [
    (Status.PENDING, Status.IN_PROGRESS),
    (Status.IN_PROGRESS, Status.IN_REVIEW),
    (Status.IN_PROGRESS, Status.MERGING),
    (Status.IN_REVIEW, Status.PENDING),
    (Status.IN_REVIEW, Status.MERGING),
    (Status.MERGING, Status.MERGED),
    (Status.MERGING, Status.PENDING),
    (Status.IN_PROGRESS, Status.AWAITING_INPUT),
    (Status.IN_REVIEW, Status.AWAITING_INPUT),
    (Status.MERGING, Status.AWAITING_INPUT),
    (Status.AWAITING_INPUT, Status.IN_PROGRESS),
    (Status.AWAITING_INPUT, Status.IN_REVIEW),
    (Status.AWAITING_INPUT, Status.MERGING),
    # A pending ticket is one nobody is working, not one nothing has been done
    # to: a run picked back up reads git and claims it straight into whatever
    # status the repository already implies.
    (Status.PENDING, Status.IN_REVIEW),
    (Status.PENDING, Status.MERGING),
    (Status.PENDING, Status.MERGED),
    # And the other direction: a claim can always be given back, which is what
    # the scheduler does with a ticket whose pass raised.
    (Status.IN_PROGRESS, Status.PENDING),
    # Re-entry: a status a ticket is already in can be set again, which is what
    # lets the caller re-stamp how long it has been there.
    (Status.PENDING, Status.PENDING),
    (Status.IN_PROGRESS, Status.IN_PROGRESS),
    (Status.IN_REVIEW, Status.IN_REVIEW),
    (Status.MERGING, Status.MERGING),
]

ILLEGAL = [
    # An agent has to be running for a question to exist.
    (Status.PENDING, Status.AWAITING_INPUT),
    (Status.AWAITING_INPUT, Status.AWAITING_INPUT),
    # Only the merge queue can declare something merged.
    (Status.IN_PROGRESS, Status.MERGED),
    (Status.IN_REVIEW, Status.MERGED),
    (Status.AWAITING_INPUT, Status.MERGED),
    # Work does not run backwards within a claim: a ticket that is under review
    # is not sent back to an implementation agent, it is given back to the queue.
    (Status.IN_REVIEW, Status.IN_PROGRESS),
    (Status.MERGING, Status.IN_PROGRESS),
    (Status.MERGING, Status.IN_REVIEW),
    (Status.AWAITING_INPUT, Status.PENDING),
]


# -- Ticket ---------------------------------------------------------------


def test_is_bug_is_exactly_having_a_parent() -> None:
    assert ticket().is_bug is False
    assert ticket(id="T-01-bug-1", parent="T-01").is_bug is True


# -- transitions ----------------------------------------------------------


@pytest.mark.parametrize(("frm", "to"), LEGAL)
def test_every_legal_transition_is_allowed(frm: Status, to: Status) -> None:
    assert can_transition(frm, to) is True


@pytest.mark.parametrize(("frm", "to"), ILLEGAL)
def test_illegal_transitions_are_refused(frm: Status, to: Status) -> None:
    assert can_transition(frm, to) is False


@pytest.mark.parametrize("to", list(Status))
def test_merged_is_terminal(to: Status) -> None:
    assert can_transition(Status.MERGED, to) is False


def test_transition_returns_a_moved_ticket_and_leaves_the_original_alone() -> None:
    subject = ticket()

    moved = transition(subject, Status.IN_PROGRESS)

    assert moved.status is Status.IN_PROGRESS
    assert subject.status is Status.PENDING
    assert moved is not subject


def test_transition_carries_every_other_field_across_unchanged() -> None:
    subject = ticket(id="T-01-bug-1", parent="T-01", review_round=2, blocked_by=("T-02",))

    moved = transition(subject, Status.IN_PROGRESS)

    assert (moved.id, moved.title, moved.deliverables) == (
        subject.id,
        subject.title,
        subject.deliverables,
    )
    assert (moved.blocked_by, moved.parent, moved.review_round) == (("T-02",), "T-01", 2)


def test_a_ticket_cannot_be_written_to() -> None:
    subject = ticket()

    with pytest.raises(FrozenInstanceError):
        subject.status = Status.IN_PROGRESS  # type: ignore[misc]


def test_an_illegal_transition_raises_and_builds_nothing() -> None:
    subject = ticket()

    with pytest.raises(IllegalTransitionError):
        transition(subject, Status.AWAITING_INPUT)

    assert subject.status is Status.PENDING


def test_a_merged_ticket_cannot_be_moved() -> None:
    subject = ticket(status=Status.MERGED)

    with pytest.raises(IllegalTransitionError):
        transition(subject, Status.PENDING)

    assert subject.status is Status.MERGED


# -- awaiting input -------------------------------------------------------


@pytest.mark.parametrize("frm", [Status.IN_PROGRESS, Status.IN_REVIEW, Status.MERGING])
def test_awaiting_input_remembers_where_it_came_from_and_returns_there(frm: Status) -> None:
    waiting = transition(ticket(status=frm), Status.AWAITING_INPUT)
    assert waiting.status is Status.AWAITING_INPUT
    assert waiting.resume_to is frm

    resumed = transition(waiting, frm)
    assert resumed.status is frm
    assert resumed.resume_to is None


def test_a_waiting_ticket_may_not_resume_into_a_state_it_did_not_come_from() -> None:
    waiting = transition(ticket(status=Status.IN_REVIEW), Status.AWAITING_INPUT)

    with pytest.raises(IllegalTransitionError):
        transition(waiting, Status.MERGING)

    assert waiting.status is Status.AWAITING_INPUT
    assert waiting.resume_to is Status.IN_REVIEW


def test_a_waiting_ticket_with_nothing_remembered_cannot_move() -> None:
    subject = ticket(status=Status.AWAITING_INPUT)

    with pytest.raises(IllegalTransitionError):
        transition(subject, Status.IN_PROGRESS)
