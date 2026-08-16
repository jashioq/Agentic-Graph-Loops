"""The shape the decompose agent has to produce, and what the parser refuses.

The schema is what the agent is asked for and the parser is what the run
trusts, so the two are checked against each other rather than separately.
"""

from typing import Any

import pytest

from agl.workflows.tickets.documents.tickets_document import TICKETS_SCHEMA, tickets_from_json
from agl.workflows.tickets.errors import InvalidTicketsError
from agl.workflows.tickets.models import Status

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


# -- round trip -----------------------------------------------------------


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
