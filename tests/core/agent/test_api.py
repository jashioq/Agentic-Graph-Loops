"""The API: data types, the error hierarchy, and what an implementation owes."""

import dataclasses
import inspect
from pathlib import Path
from typing import Any

import pytest

from agl.core.agent import (
    AgentBudgetError,
    AgentError,
    AgentOption,
    AgentQuestion,
    AgentResult,
    AgentRunner,
    AgentSpec,
    Tool,
)

# -- the abstract base ----------------------------------------------------


def test_agent_runner_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        AgentRunner()  # type: ignore[abstract]


def test_run_is_the_only_method_an_implementation_owes() -> None:
    assert AgentRunner.__abstractmethods__ == frozenset({"run"})


def test_run_is_a_coroutine_with_two_optional_callbacks() -> None:
    assert inspect.iscoroutinefunction(AgentRunner.run)
    parameters = inspect.signature(AgentRunner.run).parameters
    assert list(parameters) == ["self", "spec", "on_activity", "on_question"]
    assert parameters["on_activity"].default is None
    assert parameters["on_question"].default is None


def test_the_abc_has_no_reporting_callbacks_beyond_those_two() -> None:
    # Core modules report by returning values. `agent` is the one exception, and
    # it is an exception for exactly two callbacks, not for a general event bus.
    parameters = inspect.signature(AgentRunner.run).parameters
    assert not [name for name in parameters if name.startswith("on_event")]


# -- data types -----------------------------------------------------------


def test_every_data_type_is_a_frozen_dataclass() -> None:
    for kind in (Tool, AgentSpec, AgentResult, AgentOption, AgentQuestion):
        assert dataclasses.is_dataclass(kind), kind
        assert kind.__dataclass_params__.frozen, kind  # type: ignore[attr-defined]


async def test_a_tool_is_a_name_a_description_a_schema_and_a_handler() -> None:
    async def handler(arguments: dict[str, Any]) -> str:
        return str(arguments["a"] + arguments["b"])

    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    tool = Tool(name="add", description="Add two numbers", schema=schema, handler=handler)

    assert tool.name == "add"
    assert tool.description == "Add two numbers"
    assert tool.schema == schema
    assert await tool.handler({"a": 1, "b": 2}) == "3"


def test_a_spec_needs_a_prompt_a_cwd_and_a_role_and_nothing_else() -> None:
    spec = AgentSpec(prompt="hello", cwd=Path("/repo"), role="implement")

    assert spec.tools == ()
    assert spec.system_prompt_append is None
    assert spec.add_dirs == ()
    assert spec.allowed_tools == ()
    assert spec.disallowed_tools == ()
    assert spec.permission_mode == "default"
    assert spec.model is None
    assert spec.max_turns is None
    assert spec.max_budget_usd is None
    assert spec.output_schema is None


def test_a_result_carries_the_text_and_what_the_run_cost() -> None:
    result = AgentResult(
        text="done",
        structured={"ok": True},
        session_id="s-1",
        cost_usd=0.25,
        num_turns=4,
        duration_ms=1200,
        terminal_reason="completed",
        is_error=False,
    )

    assert result.text == "done"
    assert result.structured == {"ok": True}
    assert result.session_id == "s-1"
    assert result.cost_usd == 0.25
    assert result.num_turns == 4
    assert result.duration_ms == 1200
    assert result.terminal_reason == "completed"
    assert result.is_error is False


def test_a_question_is_a_title_and_its_options() -> None:
    question = AgentQuestion(
        title="Which storage?",
        options=(
            AgentOption(label="sqlite", description="One file on disk"),
            AgentOption(label="postgres", description="A server"),
        ),
    )

    assert question.title == "Which storage?"
    assert [option.label for option in question.options] == ["sqlite", "postgres"]


def test_the_question_type_is_this_modules_own() -> None:
    # Deliberately not `terminal.Question`: core modules do not import each
    # other, and translating one into the other is the workflow's job.
    assert AgentQuestion.__module__ == "agl.core.agent.api"
    assert not [name for name in dir(AgentQuestion) if name == "header"]


# -- errors ---------------------------------------------------------------


def test_budget_exhaustion_is_an_agent_error() -> None:
    assert issubclass(AgentBudgetError, AgentError)
    assert issubclass(AgentError, Exception)
