"""Tickets, their life cycle, and the shape the decompose agent has to produce."""

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from agl.workflows.tickets.models import (
    TICKETS_SCHEMA,
    IllegalTransitionError,
    InvalidTicketsError,
    Status,
    Ticket,
    can_transition,
    tickets_from_json,
    transition,
)

# -- fixtures as data -----------------------------------------------------

DECOMPOSED: dict[str, Any] = {
    "tickets": [
        {
            "id": "T-01",
            "title": "Add a token store",
            "deliverables": ["src/auth/store.py", "tests for it"],
        },
        {
            "id": "T-02",
            "title": "Wire the store into login",
            "deliverables": ["login uses the store"],
            "blocked_by": ["T-01"],
        },
    ]
}


def one(**overrides: Any) -> dict[str, Any]:
    """A single-ticket payload, with the one ticket's fields overridable."""
    ticket: dict[str, Any] = {"id": "T-01", "title": "A thing", "deliverables": ["a file"]}
    ticket.update(overrides)
    return {"tickets": [ticket]}


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


def test_a_ticket_round_trips_through_tickets_from_json() -> None:
    first, second = tickets_from_json(DECOMPOSED)

    assert first.id == "T-01"
    assert first.title == "Add a token store"
    assert first.deliverables == ("src/auth/store.py", "tests for it")
    assert first.blocked_by == ()
    assert second.id == "T-02"
    assert second.blocked_by == ("T-01",)


def test_fields_the_agent_does_not_supply_come_back_with_their_defaults() -> None:
    (parsed,) = tickets_from_json(one())

    assert parsed.status is Status.PENDING
    assert parsed.review_round == 0
    assert parsed.parent is None
    assert parsed.blocked_by == ()
    assert parsed.resume_to is None
    assert parsed.base_sha is None


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


# -- the schema -----------------------------------------------------------


def test_the_schema_forbids_anything_it_did_not_ask_for() -> None:
    assert TICKETS_SCHEMA["additionalProperties"] is False
    assert TICKETS_SCHEMA["properties"]["tickets"]["items"]["additionalProperties"] is False


def test_the_schema_asks_for_a_non_empty_array_of_tickets_with_patterned_ids() -> None:
    tickets = TICKETS_SCHEMA["properties"]["tickets"]
    assert TICKETS_SCHEMA["required"] == ["tickets"]
    assert tickets["type"] == "array"
    assert tickets["minItems"] == 1
    assert tickets["items"]["properties"]["id"]["pattern"]


def test_the_schema_does_not_ask_for_what_the_orchestrator_owns() -> None:
    asked = set(TICKETS_SCHEMA["properties"]["tickets"]["items"]["properties"])

    assert asked == {"id", "title", "deliverables", "blocked_by"}


def test_the_parser_enforces_every_field_the_schema_requires() -> None:
    required = TICKETS_SCHEMA["properties"]["tickets"]["items"]["required"]
    assert required  # a parser agreeing with an empty list proves nothing

    for name in required:
        payload = one()
        del payload["tickets"][0][name]
        with pytest.raises(InvalidTicketsError):
            tickets_from_json(payload)


# -- parsing failures -----------------------------------------------------


def test_tickets_from_json_rejects_a_duplicate_id() -> None:
    payload = {"tickets": [dict(one()["tickets"][0]), dict(one()["tickets"][0])]}

    with pytest.raises(InvalidTicketsError, match="duplicate"):
        tickets_from_json(payload)


def test_tickets_from_json_rejects_a_blocker_that_is_not_in_the_set() -> None:
    with pytest.raises(InvalidTicketsError, match="unknown"):
        tickets_from_json(one(blocked_by=["T-99"]))


def test_tickets_from_json_rejects_a_ticket_blocking_itself() -> None:
    with pytest.raises(InvalidTicketsError, match="itself"):
        tickets_from_json(one(blocked_by=["T-01"]))


def test_tickets_from_json_rejects_an_unknown_extra_field() -> None:
    with pytest.raises(InvalidTicketsError, match="unknown field"):
        tickets_from_json(one(status="pending"))


@pytest.mark.parametrize("bad_id", ["", "-leading", "T 01", "T/01", "T_01", "épée"])
def test_tickets_from_json_rejects_an_id_violating_the_pattern(bad_id: str) -> None:
    with pytest.raises(InvalidTicketsError, match="id"):
        tickets_from_json(one(id=bad_id))


@pytest.mark.parametrize("data", [None, [], "tickets", 7, {"tickets": None}])
def test_tickets_from_json_rejects_anything_that_is_not_the_shape(data: Any) -> None:
    with pytest.raises(InvalidTicketsError):
        tickets_from_json(data)


def test_tickets_from_json_rejects_an_empty_array() -> None:
    with pytest.raises(InvalidTicketsError, match="empty"):
        tickets_from_json({"tickets": []})


def test_tickets_from_json_rejects_a_ticket_that_is_not_an_object() -> None:
    with pytest.raises(InvalidTicketsError):
        tickets_from_json({"tickets": ["T-01"]})


def test_tickets_from_json_rejects_a_key_outside_the_named_one() -> None:
    payload = one()
    payload["notes"] = "extra"

    with pytest.raises(InvalidTicketsError, match="unknown field"):
        tickets_from_json(payload)


@pytest.mark.parametrize("deliverables", [[], "a file", [""], [1]])
def test_tickets_from_json_rejects_deliverables_that_are_not_a_list_of_text(
    deliverables: Any,
) -> None:
    with pytest.raises(InvalidTicketsError, match="deliverables"):
        tickets_from_json(one(deliverables=deliverables))


@pytest.mark.parametrize("title", ["", 7, None])
def test_tickets_from_json_rejects_a_title_that_is_not_text(title: Any) -> None:
    with pytest.raises(InvalidTicketsError, match="title"):
        tickets_from_json(one(title=title))


def test_tickets_from_json_rejects_blocked_by_that_is_not_a_list() -> None:
    with pytest.raises(InvalidTicketsError, match="blocked_by"):
        tickets_from_json(one(blocked_by="T-01"))
