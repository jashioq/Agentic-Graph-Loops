"""The API: data types, the error hierarchy, and what an implementation owes."""

import dataclasses
import inspect
from pathlib import Path
from typing import Any

import pytest

from agl.core.agent import (
    NO_PARAMS,
    AgentBudgetError,
    AgentError,
    AgentOption,
    AgentOutputError,
    AgentQuestion,
    AgentResult,
    AgentRunner,
    AgentSpec,
    Model,
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
    assert spec.disallowed_tools == ()
    assert spec.permission_mode == "default"
    assert spec.model is None
    assert spec.output_schema is None


def test_a_spec_takes_a_model_as_the_enum() -> None:
    spec = AgentSpec(
        prompt="hello", cwd=Path("/repo"), role="implement", model=Model.OPUS
    )

    assert spec.model is Model.OPUS


def test_a_result_carries_the_text_and_what_the_run_cost() -> None:
    result = AgentResult(
        text="done",
        structured={"ok": True},
        session_id="s-1",
        cost_usd=0.25,
        num_turns=4,
        duration_ms=1200,
        terminal_reason="completed",
    )

    assert result.text == "done"
    assert result.structured == {"ok": True}
    assert result.session_id == "s-1"
    assert result.cost_usd == 0.25
    assert result.num_turns == 4
    assert result.duration_ms == 1200
    assert result.terminal_reason == "completed"


def test_a_result_carries_no_error_flag() -> None:
    # A run that could not be completed raises. There is no such thing as a
    # result that came back saying it failed, so there is no field for it.
    assert "is_error" not in {field.name for field in dataclasses.fields(AgentResult)}


def test_a_spec_has_no_allowed_tools() -> None:
    # The permission callback allows every non-question tool, so pre-allowing
    # one bought nothing and allowing `AskUserQuestion` silently disabled asking.
    assert "allowed_tools" not in {field.name for field in dataclasses.fields(AgentSpec)}


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


# -- the model ------------------------------------------------------------


def test_the_model_enum_holds_the_four_tiers() -> None:
    assert [member.name for member in Model] == ["HAIKU", "SONNET", "OPUS", "FABLE"]


def test_the_members_are_the_cli_aliases_not_pinned_ids() -> None:
    # An alias follows the latest release of its tier; pinning `claude-opus-5`
    # is a decision to revisit on a schedule, not one to bake into the type.
    assert [member.value for member in Model] == ["haiku", "sonnet", "opus", "fable"]


def test_a_model_is_a_string() -> None:
    # `StrEnum`, so a member compares equal to its alias and needs no unwrapping
    # to be read by a person.
    assert isinstance(Model.SONNET, str)
    assert Model.SONNET == "sonnet"


# -- errors ---------------------------------------------------------------


def test_budget_exhaustion_is_an_agent_error() -> None:
    assert issubclass(AgentBudgetError, AgentError)
    assert issubclass(AgentError, Exception)


def test_output_parse_failure_is_an_agent_error_distinct_from_budget() -> None:
    # Distinct from `AgentBudgetError` even though both are never retried: they
    # are unrelated ways a call can fail, not the same failure under two names.
    assert issubclass(AgentOutputError, AgentError)
    assert not issubclass(AgentOutputError, AgentBudgetError)


# -- the no-argument schema -----------------------------------------------


def test_no_params_is_a_closed_object_schema_with_no_properties() -> None:
    # The canonical shape for a scoped tool: everything it may touch was closed
    # over at construction, so there is nothing for the model to widen.
    assert NO_PARAMS == {"type": "object", "properties": {}, "additionalProperties": False}


async def test_a_tool_can_be_built_on_no_params() -> None:
    async def handler(arguments: dict[str, Any]) -> str:
        return "the document"

    tool = Tool(name="read_it", description="Read it", schema=NO_PARAMS, handler=handler)

    assert await tool.handler({}) == "the document"
