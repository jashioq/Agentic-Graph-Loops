"""Per-role tools: what a closure hands an agent, and what it cannot hand it.

A real `FileStore` in `tmp_path` throughout — the tools are three lines of
closure over a store, and faking the store would leave nothing under test.
`FakeAgentRunner` appears where the point is what a *run* can reach, since that
is the boundary being asserted: a spec carries tools, and a tool call goes
through them or fails.

The scoping guarantee is a property of the schema, not of a check inside a
handler, so it is asserted on the schema directly. A tool with no parameters
gives the model nothing to fill in and so nothing to widen.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from agl.core.agent import AgentSpec, Tool
from agl.core.store.impl.file_store import FileStore
from agl.workflows.tickets.models import TICKETS_SCHEMA, Ticket, tickets_from_json
from agl.workflows.tickets.tools import (
    SPEC_KEY,
    STANDARDS_KEY,
    TICKETS_KEY,
    decompose_tools,
    get_ticket,
    implement_tools,
    interview_tools,
    read_spec,
    read_standards,
    review_quality_tools,
    review_spec_tools,
    save_spec,
    save_tickets,
)
from tests.fakes import FakeAgentRunner, ScriptedRun

SPEC = "# Add auth\n\nUsers sign in with a password.\n"
STANDARDS = "# Standards\n\nNo mutable globals.\n"

PAYLOAD: dict[str, Any] = {
    "tickets": [
        {"id": "T-01", "title": "Add the token store", "deliverables": ["TokenStore.kt"]},
        {
            "id": "T-03",
            "title": "Add the login screen",
            "deliverables": ["LoginScreen.kt", "a test"],
            "blocked_by": ["T-01"],
        },
        {"id": "T-05", "title": "Add refresh", "deliverables": ["Refresh.kt"]},
    ]
}


@pytest.fixture
def store(tmp_path: Path) -> FileStore:
    """A run's documents: a spec, the standards, and the tickets."""
    store = FileStore(tmp_path / "run")
    store.write(SPEC_KEY, SPEC)
    store.write(STANDARDS_KEY, STANDARDS)
    store.write_json(TICKETS_KEY, PAYLOAD)
    return store


@pytest.fixture
def empty(tmp_path: Path) -> FileStore:
    """A store holding nothing at all, for the tools that have to cope with it."""
    return FileStore(tmp_path / "fresh")


async def call(tool: Tool, **arguments: Any) -> str:
    """Invoke a tool's handler the way a run would, with whatever arguments."""
    return await tool.handler(dict(arguments))


def entry(ticket_id: str) -> dict[str, Any]:
    """The stored record for one ticket."""
    return next(item for item in PAYLOAD["tickets"] if item["id"] == ticket_id)


def spec(role: str, cwd: Path, tools: tuple[Tool, ...]) -> AgentSpec:
    return AgentSpec(prompt="Do the work.", cwd=cwd, role=role, tools=tools)


# -- reads -----------------------------------------------------------------


async def test_read_spec_returns_what_the_store_holds(store: FileStore) -> None:
    assert await call(read_spec(store)) == SPEC


async def test_read_standards_returns_what_the_store_holds(store: FileStore) -> None:
    assert await call(read_standards(store)) == STANDARDS


@pytest.mark.parametrize("factory", [read_spec, read_standards])
def test_a_read_tool_takes_no_arguments_at_all(factory: Any, store: FileStore) -> None:
    schema = factory(store).schema
    assert schema["properties"] == {}
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize("factory", [save_spec, save_tickets])
def test_a_write_tool_takes_content_and_never_a_key(factory: Any, store: FileStore) -> None:
    # Where a document lands was decided when the tool was built, so there is
    # nothing in the schema for the model to point somewhere else.
    schema = factory(store).schema
    assert "key" not in schema["properties"]
    assert schema["additionalProperties"] is False


def test_save_tickets_asks_for_exactly_the_shape_the_workflow_parses(
    store: FileStore,
) -> None:
    # The schema the model fills in and the parser that guards the store are the
    # same statement of what a ticket is, so they cannot drift apart.
    assert save_tickets(store).schema is TICKETS_SCHEMA


async def test_a_missing_document_is_reported_rather_than_raised(empty: FileStore) -> None:
    # A project with no standards file is a fact for the agent to work around,
    # not a reason to kill a run that had already started.
    answer = await call(read_standards(empty))
    assert STANDARDS_KEY in answer  # it names what is missing
    assert answer != STANDARDS and not answer.startswith("#")  # and is not mistakable for one


async def test_a_missing_spec_is_reported_rather_than_raised(empty: FileStore) -> None:
    assert SPEC_KEY in await call(read_spec(empty))


# -- one ticket, and only one ----------------------------------------------


async def test_get_ticket_returns_the_ticket_it_was_built_for(store: FileStore) -> None:
    assert json.loads(await call(get_ticket(store, "T-03"))) == entry("T-03")


def test_get_ticket_has_no_parameter_to_pass(store: FileStore) -> None:
    # The scoping guarantee itself: a reviewer holding this tool for T-03 has no
    # argument it could pass to reach T-05. Nothing in the handler enforces
    # that; the empty schema is what makes it true.
    schema = get_ticket(store, "T-03").schema
    assert schema["properties"] == {}
    assert schema["additionalProperties"] is False


async def test_no_argument_reaches_another_ticket(store: FileStore) -> None:
    answer = await call(get_ticket(store, "T-03"), id="T-05", ticket_id="T-05")
    assert json.loads(answer) == entry("T-03")


async def test_two_tools_bound_to_different_ids_each_return_their_own(
    store: FileStore,
) -> None:
    assert json.loads(await call(get_ticket(store, "T-01"))) == entry("T-01")
    assert json.loads(await call(get_ticket(store, "T-05"))) == entry("T-05")


async def test_get_ticket_reads_the_store_at_call_time(store: FileStore) -> None:
    # Not a snapshot taken when the tool was built: a ticket edited mid-run —
    # by the orchestrator, or by a decompose that ran again — reads as it is now.
    tool = get_ticket(store, "T-03")
    changed = json.loads(json.dumps(PAYLOAD))
    changed["tickets"][1]["title"] = "Add the login screen, with biometrics"
    store.write_json(TICKETS_KEY, changed)

    assert json.loads(await call(tool))["title"] == "Add the login screen, with biometrics"


async def test_an_unknown_ticket_is_reported_rather_than_raised(store: FileStore) -> None:
    answer = await call(get_ticket(store, "T-99"))
    assert "T-99" in answer
    assert not answer.startswith("{")


async def test_get_ticket_with_no_tickets_stored_is_reported(empty: FileStore) -> None:
    assert TICKETS_KEY in await call(get_ticket(empty, "T-03"))


# -- writes land where the closure said ------------------------------------


async def test_save_spec_writes_the_content_at_the_closure_s_key(store: FileStore) -> None:
    await call(save_spec(store), content="# Rewritten\n")
    assert store.read(SPEC_KEY) == "# Rewritten\n"


async def test_save_spec_ignores_anything_the_agent_says_about_where(
    store: FileStore, tmp_path: Path
) -> None:
    # Driven through a run rather than argued: the call names a key of its own
    # and the document still lands at the one the closure chose.
    runner = FakeAgentRunner(
        {
            "interview": ScriptedRun(
                calls=(("save_spec", {"content": "# Mine\n", "key": "elsewhere.md"}),)
            )
        }
    )

    await runner.run(spec("interview", tmp_path, interview_tools(store)))

    assert store.read(SPEC_KEY) == "# Mine\n"
    assert store.exists("elsewhere.md") is False


async def test_save_spec_says_nothing_useful_when_given_no_content(store: FileStore) -> None:
    answer = await call(save_spec(store), text="# Wrong field\n")
    assert "content" in answer
    assert store.read(SPEC_KEY) == SPEC


# -- saving tickets --------------------------------------------------------

VALID: dict[str, Any] = {
    "tickets": [
        {"id": "N-01", "title": "Cut the branch", "deliverables": ["a branch"]},
        {
            "id": "N-02",
            "title": "Land the work",
            "deliverables": ["the work"],
            "blocked_by": ["N-01"],
        },
    ]
}


async def test_save_tickets_writes_what_the_workflow_reads_back(empty: FileStore) -> None:
    await call(save_tickets(empty), **VALID)

    tickets = tickets_from_json(empty.read_json(TICKETS_KEY))
    assert [ticket.id for ticket in tickets] == ["N-01", "N-02"]
    assert tickets[1].blocked_by == ("N-01",)
    assert all(isinstance(ticket, Ticket) for ticket in tickets)


async def test_save_tickets_replaces_what_was_there(store: FileStore) -> None:
    await call(save_tickets(store), **VALID)
    assert [t.id for t in tickets_from_json(store.read_json(TICKETS_KEY))] == ["N-01", "N-02"]


INVALID: dict[str, dict[str, Any]] = {
    "no tickets field": {},
    "empty list": {"tickets": []},
    "not a list": {"tickets": {"id": "N-01"}},
    "missing deliverables": {"tickets": [{"id": "N-01", "title": "Cut the branch"}]},
    "empty deliverables": {"tickets": [{"id": "N-01", "title": "t", "deliverables": []}]},
    "blank title": {"tickets": [{"id": "N-01", "title": " ", "deliverables": ["a"]}]},
    "unusable id": {"tickets": [{"id": "N 01/x", "title": "t", "deliverables": ["a"]}]},
    "invented field": {
        "tickets": [{"id": "N-01", "title": "t", "deliverables": ["a"], "status": "done"}]
    },
    "duplicate ids": {
        "tickets": [
            {"id": "N-01", "title": "t", "deliverables": ["a"]},
            {"id": "N-01", "title": "u", "deliverables": ["b"]},
        ]
    },
    "blocked by a stranger": {
        "tickets": [
            {"id": "N-01", "title": "t", "deliverables": ["a"], "blocked_by": ["N-09"]}
        ]
    },
}


@pytest.mark.parametrize("payload", INVALID.values(), ids=list(INVALID))
async def test_invalid_tickets_come_back_as_something_to_fix(
    payload: dict[str, Any], empty: FileStore
) -> None:
    # The whole reason these are tools rather than an `output_schema`: the model
    # is told what was wrong and gets to call again in the same session.
    answer = await call(save_tickets(empty), **payload)
    assert answer.strip()
    assert not answer.startswith("Saved")


@pytest.mark.parametrize("payload", INVALID.values(), ids=list(INVALID))
async def test_nothing_invalid_reaches_the_store(
    payload: dict[str, Any], store: FileStore
) -> None:
    await call(save_tickets(store), **payload)
    assert store.read_json(TICKETS_KEY) == PAYLOAD


@pytest.mark.parametrize("payload", INVALID.values(), ids=list(INVALID))
async def test_a_refused_save_writes_no_document_at_all(
    payload: dict[str, Any], empty: FileStore
) -> None:
    await call(save_tickets(empty), **payload)
    assert empty.list() == ()


async def test_a_corrected_second_call_lands(empty: FileStore) -> None:
    tool = save_tickets(empty)
    await call(tool, tickets=[])
    await call(tool, **VALID)
    assert [t.id for t in tickets_from_json(empty.read_json(TICKETS_KEY))] == ["N-01", "N-02"]


# -- what each role is given -----------------------------------------------


def names(tools: tuple[Tool, ...]) -> set[str]:
    return {tool.name for tool in tools}


def test_interview_can_only_save_the_spec(store: FileStore) -> None:
    assert names(interview_tools(store)) == {"save_spec"}


def test_decompose_reads_the_spec_and_saves_tickets(store: FileStore) -> None:
    assert names(decompose_tools(store)) == {"read_spec", "save_tickets"}


def test_implement_reads_everything_and_writes_nothing(store: FileStore) -> None:
    assert names(implement_tools(store, "T-03")) == {"get_ticket", "read_spec", "read_standards"}


def test_review_quality_is_given_the_standards_and_not_the_spec(store: FileStore) -> None:
    assert names(review_quality_tools(store, "T-03")) == {"get_ticket", "read_standards"}


def test_review_spec_is_given_the_spec_and_not_the_standards(store: FileStore) -> None:
    assert names(review_spec_tools(store, "T-03")) == {"get_ticket", "read_spec"}


def bundles(store: FileStore) -> list[tuple[Tool, ...]]:
    """Every role bundle there is, built over one store."""
    return [
        interview_tools(store),
        decompose_tools(store),
        implement_tools(store, "T-03"),
        review_quality_tools(store, "T-03"),
        review_spec_tools(store, "T-03"),
    ]


def test_no_bundle_carries_a_name_twice(store: FileStore) -> None:
    # Two tools of one name leave the model reaching one of them and never the
    # other; `build_tool_server` refuses the spec outright.
    for built in bundles(store):
        assert len(names(built)) == len(built), names(built)


def test_every_tool_describes_itself_for_the_model(store: FileStore) -> None:
    # A description is what the model decides on; an empty or one-word one is a
    # tool it will call at the wrong moment.
    for built in bundles(store):
        for tool in built:
            assert len(tool.description.split()) >= 8, tool.name


# -- the reviewer that must not see the spec -------------------------------


async def test_review_quality_has_no_tool_that_reaches_the_spec(
    store: FileStore, tmp_path: Path
) -> None:
    # By name above, and here by driving every tool the role holds: none of them
    # answers with the spec. Re-litigating design decisions is not its job.
    tools = review_quality_tools(store, "T-03")
    runner = FakeAgentRunner(
        {"review-quality": ScriptedRun(calls=tuple((tool.name, {}) for tool in tools))}
    )

    await runner.run(spec("review-quality", tmp_path, tools))

    assert len(runner.tool_results) == len(tools)
    for result in runner.tool_results:
        assert SPEC.strip() not in result.text
        assert result.is_error is False


async def test_review_quality_cannot_call_a_tool_it_was_not_given(
    store: FileStore, tmp_path: Path
) -> None:
    runner = FakeAgentRunner({"review-quality": ScriptedRun(calls=(("read_spec", {}),))})

    with pytest.raises(AssertionError, match="read_spec"):
        await runner.run(
            spec("review-quality", tmp_path, review_quality_tools(store, "T-03"))
        )


async def test_review_spec_does_get_the_spec(store: FileStore, tmp_path: Path) -> None:
    # The other half of the same fact: the difference between the two reviewers
    # is the bundle they were handed and nothing else.
    tools = review_spec_tools(store, "T-03")
    runner = FakeAgentRunner(
        {"review-spec": ScriptedRun(calls=tuple((tool.name, {}) for tool in tools))}
    )

    await runner.run(spec("review-spec", tmp_path, tools))

    assert SPEC in [result.text for result in runner.tool_results]
