"""`FakeAgentRunner` — the thing workflow tests will actually be written against.

It is scripted rather than mocked, so what a workflow test asserts is the same
shape of value a real run would have produced. That includes how a tool call
fails: the fake wraps a handler the way `agent.impl.tools` does, and the last
section drives the same handler through both to show they agree.
"""

from pathlib import Path
from typing import Any

import pytest

from agl.core.agent import AgentOption, AgentQuestion, AgentResult, AgentSpec, Tool
from agl.core.agent.impl.tools import build_tool_server
from tests.core.agent.test_tools import call, text_of
from tests.fakes import FakeAgentRunner, ScriptedRun, ToolResult


def spec(role: str = "implement", *tools: Tool) -> AgentSpec:
    return AgentSpec(prompt="do it", cwd=Path("/repo"), role=role, tools=tools)


def add_tool(recorder: list[dict[str, Any]] | None = None) -> Tool:
    async def handler(arguments: dict[str, Any]) -> str:
        if recorder is not None:
            recorder.append(arguments)
        return str(arguments["a"] + arguments["b"])

    return Tool(name="add", description="Add two numbers", schema={}, handler=handler)


def failing_tool() -> Tool:
    async def handler(arguments: dict[str, Any]) -> str:
        raise FileNotFoundError("standards.md")

    return Tool(name="fail", description="Always fails", schema={}, handler=handler)


def wrong_type_tool() -> Tool:
    async def handler(arguments: dict[str, Any]) -> str:
        return {"not": "a string"}  # type: ignore[return-value]

    return Tool(name="wrong", description="Returns a dict", schema={}, handler=handler)


# -- scripting -------------------------------------------------------------


async def test_a_bare_string_is_the_text_the_run_produced() -> None:
    runner = FakeAgentRunner({"implement": "the patch"})

    result = await runner.run(spec())

    assert result.text == "the patch"


async def test_a_full_result_is_returned_as_given() -> None:
    scripted = AgentResult(
        text="done",
        structured={"ok": True},
        session_id="s-9",
        cost_usd=1.5,
        num_turns=7,
        duration_ms=99,
        terminal_reason="completed",
    )
    runner = FakeAgentRunner({"implement": scripted})

    assert await runner.run(spec()) == scripted


async def test_a_script_keyed_by_role_answers_each_role() -> None:
    runner = FakeAgentRunner({"implement": "a patch", "review-quality": "looks fine"})

    assert (await runner.run(spec("implement"))).text == "a patch"
    assert (await runner.run(spec("review-quality"))).text == "looks fine"


async def test_a_role_can_be_called_more_than_once() -> None:
    # A workflow runs the same role over many items; a mapping is not consumed.
    runner = FakeAgentRunner({"implement": "a patch"})

    assert (await runner.run(spec())).text == "a patch"
    assert (await runner.run(spec())).text == "a patch"


async def test_a_list_is_consumed_in_order() -> None:
    runner = FakeAgentRunner(["first", "second"])

    assert (await runner.run(spec())).text == "first"
    assert (await runner.run(spec("anything"))).text == "second"


async def test_a_list_that_runs_out_says_so() -> None:
    runner = FakeAgentRunner(["only one"])
    await runner.run(spec())

    with pytest.raises(AssertionError, match="script"):
        await runner.run(spec())


async def test_an_unscripted_role_says_which_role_and_what_was_scripted() -> None:
    runner = FakeAgentRunner({"implement": "a patch"})

    with pytest.raises(AssertionError) as raised:
        await runner.run(spec("review-quality"))

    assert "review-quality" in str(raised.value)
    assert "implement" in str(raised.value)


# -- recording -------------------------------------------------------------


async def test_every_spec_is_recorded() -> None:
    runner = FakeAgentRunner(["one", "two"])

    await runner.run(spec("implement"))
    await runner.run(spec("review-quality"))

    assert [recorded.role for recorded in runner.specs] == ["implement", "review-quality"]
    assert runner.specs[0].prompt == "do it"


# -- invoking the spec's own tools ----------------------------------------


async def test_it_can_invoke_a_tool_the_spec_carried() -> None:
    # This is how a workflow test proves a role got the right tools and that the
    # scoping inside them holds.
    seen: list[dict[str, Any]] = []
    runner = FakeAgentRunner(
        {"implement": ScriptedRun("done", calls=(("add", {"a": 2, "b": 3}),))}
    )

    result = await runner.run(spec("implement", add_tool(seen)))

    assert seen == [{"a": 2, "b": 3}]
    assert runner.tool_results == [ToolResult("5")]
    assert result.text == "done"


async def test_calling_a_tool_the_spec_does_not_carry_says_what_it_had() -> None:
    runner = FakeAgentRunner({"implement": ScriptedRun(calls=(("delete_everything", {}),))})

    with pytest.raises(AssertionError) as raised:
        await runner.run(spec("implement", add_tool()))

    assert "delete_everything" in str(raised.value)
    assert "add" in str(raised.value)


# -- a handler that fails --------------------------------------------------


async def test_a_raising_handler_becomes_a_result_and_the_run_carries_on() -> None:
    # The failure is a fact about the world the model reacts to, not the end of
    # the run: a workflow tested against this fake is written against the same
    # contract `agent.impl.tools` gives it in production.
    calls = (("fail", {}), ("add", {"a": 1, "b": 1}))
    runner = FakeAgentRunner({"implement": ScriptedRun("finished anyway", calls=calls)})

    result = await runner.run(spec("implement", failing_tool(), add_tool()))

    assert result.text == "finished anyway"
    assert runner.tool_results[0].is_error is True


async def test_the_error_names_the_exception_type_and_its_message() -> None:
    runner = FakeAgentRunner({"implement": ScriptedRun(calls=(("fail", {}),))})

    await runner.run(spec("implement", failing_tool()))

    assert runner.tool_results == [
        ToolResult("FileNotFoundError: standards.md", is_error=True)
    ]


async def test_a_failed_call_is_distinguishable_from_a_successful_one() -> None:
    runner = FakeAgentRunner(
        {"implement": ScriptedRun(calls=(("add", {"a": 2, "b": 3}), ("fail", {})))}
    )

    await runner.run(spec("implement", add_tool(), failing_tool()))

    assert runner.tool_results == [
        ToolResult("5", is_error=False),
        ToolResult("FileNotFoundError: standards.md", is_error=True),
    ]


async def test_a_handler_returning_a_non_string_raises_rather_than_being_coerced() -> None:
    # The model cannot fix a handler that violates its own type, so this one is
    # not turned into an answer the way a raise is.
    runner = FakeAgentRunner({"implement": ScriptedRun(calls=(("wrong", {}),))})

    with pytest.raises(TypeError, match="expected str"):
        await runner.run(spec("implement", wrong_type_tool()))


# -- the fake and the registration agree ------------------------------------


async def outcomes(tool: Tool, **arguments: Any) -> tuple[ToolResult, ToolResult]:
    """One handler's outcome through the fake, and through `_register`."""
    runner = FakeAgentRunner({"implement": ScriptedRun(calls=((tool.name, arguments),))})
    await runner.run(spec("implement", tool))

    servers, _ = build_tool_server([tool])
    registered = await call(servers, tool.name, **arguments)

    return runner.tool_results[0], ToolResult(text_of(registered), bool(registered.isError))


async def test_a_successful_call_is_the_same_answer_either_way() -> None:
    through_the_fake, through_the_server = await outcomes(add_tool(), a=2, b=3)

    assert through_the_fake == through_the_server == ToolResult("5", is_error=False)


async def test_a_raise_is_the_same_error_either_way() -> None:
    through_the_fake, through_the_server = await outcomes(failing_tool())

    assert through_the_fake == through_the_server
    assert through_the_fake.is_error is True


async def test_a_non_string_return_names_the_type_violation_either_way() -> None:
    # The classification is the same — neither one hands the model an answer —
    # but the shapes differ by design: MCP's own call boundary catches the
    # `TypeError` the wrapper raises, and the fake has no such boundary, so the
    # test that owns it sees the raise.
    message = "tool 'wrong' returned dict, expected str"
    runner = FakeAgentRunner({"implement": ScriptedRun(calls=(("wrong", {}),))})

    with pytest.raises(TypeError) as raised:
        await runner.run(spec("implement", wrong_type_tool()))

    servers, _ = build_tool_server([wrong_type_tool()])
    registered = await call(servers, "wrong")

    assert str(raised.value) == message
    assert registered.isError is True
    assert text_of(registered) == message


# -- questions -------------------------------------------------------------


QUESTION = AgentQuestion(
    title="Which storage?",
    options=(
        AgentOption(label="sqlite", description="One file"),
        AgentOption(label="postgres", description="A server"),
    ),
)


async def test_it_can_fire_a_question_and_records_the_answer() -> None:
    asked: list[AgentQuestion] = []

    async def answer(question: AgentQuestion) -> str:
        asked.append(question)
        return "sqlite"

    runner = FakeAgentRunner({"implement": ScriptedRun("done", question=QUESTION)})

    await runner.run(spec(), on_question=answer)

    assert asked == [QUESTION]
    assert runner.answers == ["sqlite"]


async def test_a_scripted_question_with_nobody_to_ask_fails_loudly() -> None:
    runner = FakeAgentRunner({"implement": ScriptedRun(question=QUESTION)})

    with pytest.raises(AssertionError, match="on_question"):
        await runner.run(spec())


# -- activity --------------------------------------------------------------


async def test_it_can_fire_a_sequence_of_activity_strings() -> None:
    seen: list[str] = []
    runner = FakeAgentRunner(
        {"implement": ScriptedRun("done", activity=("Read a.py", "Edit a.py"))}
    )

    await runner.run(spec(), on_activity=seen.append)

    assert seen == ["Read a.py", "Edit a.py"]


async def test_activity_is_dropped_when_nobody_is_watching() -> None:
    runner = FakeAgentRunner({"implement": ScriptedRun("done", activity=("Read a.py",))})

    assert (await runner.run(spec())).text == "done"
