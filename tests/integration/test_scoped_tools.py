"""Capability by closure: a run reaches what its tools closed over, and nothing else.

This is the mechanism the whole design rests on. `agent` never sees a ticket id
or a store key — it sees a name, a schema, and something to await — so what a
run may touch is decided entirely by what the caller bound into the handler
before handing it over. The schema is where that has to hold: a tool with no
parameters offers the model nothing to widen.

Both halves are asserted here. The positive one is that two runs given the same
tool *name* bound to different records each get their own. The negative one is
that there is no argument that changes the answer — the run scripted below
passes another record's id and still gets the one its tool was built for.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agl.core.agent import AgentSpec, Tool
from agl.core.store import Store
from agl.core.store.impl.file_store import FileStore
from tests.fakes import FakeAgentRunner, ScriptedRun

NO_PARAMS: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}

TEXT_ONLY: dict[str, Any] = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}

TICKETS = {
    "T-01": {"title": "Add auth", "owner": "ana"},
    "T-02": {"title": "Add logging", "owner": "bo"},
    "T-03": {"title": "Add caching", "owner": "cy"},
}


@pytest.fixture
def store(tmp_path: Path) -> FileStore:
    """A run's documents: three records in one JSON file."""
    store = FileStore(tmp_path / "run")
    store.write_json("tickets.json", TICKETS)
    return store


# -- the factories a workflow would have --------------------------------------


def read_ticket_tool(store: Store, ticket_id: str) -> Tool:
    """One record, bound at build time. The model is given no way to say which."""

    async def handler(arguments: dict[str, Any]) -> str:
        return json.dumps(store.read_json("tickets.json")[ticket_id], sort_keys=True)

    return Tool(
        name="read_ticket",
        description="The ticket you are working on.",
        schema=NO_PARAMS,
        handler=handler,
    )


def save_review_tool(store: Store, ticket_id: str) -> Tool:
    """Writes the review the model wrote, under the key the closure chose."""

    async def handler(arguments: dict[str, Any]) -> str:
        key = f"reviews/{ticket_id}.md"
        store.write(key, arguments["text"])
        return f"saved {key}"

    return Tool(
        name="save_review",
        description="Save your review.",
        schema=TEXT_ONLY,
        handler=handler,
    )


def failing_tool() -> Tool:
    """A tool whose handler raises, the way a real one does when the world says no."""

    async def handler(arguments: dict[str, Any]) -> str:
        raise FileNotFoundError("standards.md")

    return Tool(
        name="read_standards",
        description="The project's standards.",
        schema=NO_PARAMS,
        handler=handler,
    )


def reporting(tool: Tool, errors: list[str]) -> Tool:
    """`tool` with its failures turned into an answer, and recorded on the way.

    This is what `agent.impl.tools` does around every registered handler: a
    handler that raises is a fact about the world the model should react to, so
    it becomes an error result and the run carries on.
    """

    async def handler(arguments: dict[str, Any]) -> str:
        try:
            return await tool.handler(arguments)
        except Exception as error:  # noqa: BLE001 - the model decides what to do
            errors.append(f"{type(error).__name__}: {error}")
            return f"error: {type(error).__name__}: {error}"

    return replace(tool, handler=handler)


def spec(role: str, cwd: Path, *tools: Tool) -> AgentSpec:
    return AgentSpec(prompt="Do the work.", cwd=cwd, role=role, tools=tools)


# -- one record each -------------------------------------------------------


async def test_two_runs_bound_to_different_records_each_get_their_own(
    store: FileStore, tmp_path: Path
) -> None:
    runner = FakeAgentRunner(
        [ScriptedRun(calls=(("read_ticket", {}),)), ScriptedRun(calls=(("read_ticket", {}),))]
    )

    await runner.run(spec("review", tmp_path, read_ticket_tool(store, "T-01")))
    await runner.run(spec("review", tmp_path, read_ticket_tool(store, "T-02")))

    assert json.loads(runner.tool_results[0]) == TICKETS["T-01"]
    assert json.loads(runner.tool_results[1]) == TICKETS["T-02"]


async def test_the_tool_has_the_same_name_in_both_runs(store: FileStore, tmp_path: Path) -> None:
    # The binding is per-spec, not per-name: nothing about the name says which
    # record answered, which is why two runs can be given "read_ticket" and mean
    # different things by it.
    runner = FakeAgentRunner([ScriptedRun(), ScriptedRun()])

    await runner.run(spec("review", tmp_path, read_ticket_tool(store, "T-01")))
    await runner.run(spec("review", tmp_path, read_ticket_tool(store, "T-02")))

    assert [tool.name for call in runner.specs for tool in call.tools] == [
        "read_ticket",
        "read_ticket",
    ]


async def test_no_argument_reaches_another_record(store: FileStore, tmp_path: Path) -> None:
    # The negative, driven rather than argued: the run asks for T-02 by every
    # name it could and still gets the record its tool was built for.
    runner = FakeAgentRunner(
        {"review": ScriptedRun(calls=(("read_ticket", {"id": "T-02", "ticket": "T-03"}),))}
    )

    await runner.run(spec("review", tmp_path, read_ticket_tool(store, "T-01")))

    assert json.loads(runner.tool_results[0]) == TICKETS["T-01"]


def test_the_schema_offers_no_parameter_to_pass(store: FileStore) -> None:
    # And this is why: there is nothing in the schema for a model to fill in.
    tool = read_ticket_tool(store, "T-01")
    assert tool.schema["properties"] == {}
    assert tool.schema["additionalProperties"] is False


async def test_a_run_reaches_only_the_tools_its_spec_carried(
    store: FileStore, tmp_path: Path
) -> None:
    runner = FakeAgentRunner({"review": ScriptedRun(calls=(("save_review", {"text": "no"}),))})

    with pytest.raises(AssertionError, match="save_review"):
        await runner.run(spec("review", tmp_path, read_ticket_tool(store, "T-01")))


# -- writing back ----------------------------------------------------------


async def test_a_write_lands_at_the_key_the_closure_chose(
    store: FileStore, tmp_path: Path
) -> None:
    runner = FakeAgentRunner(
        {"review": ScriptedRun(calls=(("save_review", {"text": "Looks fine.\n"}),))}
    )

    await runner.run(spec("review", tmp_path, save_review_tool(store, "T-01")))

    assert store.list("reviews/") == ("reviews/T-01.md",)
    assert store.read("reviews/T-01.md") == "Looks fine.\n"
    assert runner.tool_results == ["saved reviews/T-01.md"]


def test_the_write_schema_takes_the_content_and_not_the_key(store: FileStore) -> None:
    assert set(save_review_tool(store, "T-01").schema["properties"]) == {"text"}


async def test_two_writers_cannot_land_on_each_other_s_key(
    store: FileStore, tmp_path: Path
) -> None:
    review = ScriptedRun(calls=(("save_review", {"text": "ok"}),))
    runner = FakeAgentRunner([review, review])

    await runner.run(spec("review", tmp_path, save_review_tool(store, "T-01")))
    await runner.run(spec("review", tmp_path, save_review_tool(store, "T-02")))

    assert store.list("reviews/") == ("reviews/T-01.md", "reviews/T-02.md")


# -- a handler that fails ---------------------------------------------------


async def test_a_reported_failure_is_visible_and_the_run_survives(tmp_path: Path) -> None:
    errors: list[str] = []
    runner = FakeAgentRunner(
        {"review": ScriptedRun("finished anyway", calls=(("read_standards", {}),))}
    )

    result = await runner.run(spec("review", tmp_path, reporting(failing_tool(), errors)))

    assert errors == ["FileNotFoundError: standards.md"]
    assert runner.tool_results == ["error: FileNotFoundError: standards.md"]
    assert result.text == "finished anyway"
    assert result.is_error is False


async def test_an_unwrapped_raise_takes_the_whole_run_down(tmp_path: Path) -> None:
    # `FakeAgentRunner` awaits the handler as given, so a raising one propagates
    # out of `run`. `ClaudeRunner` does not behave this way — every handler it
    # registers is wrapped, and the raise becomes an error result the model
    # reads. A workflow tested only against the fake would be written against
    # the wrong contract; see the report.
    runner = FakeAgentRunner({"review": ScriptedRun(calls=(("read_standards", {}),))})

    with pytest.raises(FileNotFoundError):
        await runner.run(spec("review", tmp_path, failing_tool()))
