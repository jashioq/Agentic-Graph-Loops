"""The adapter, driven against a stub `query_fn`. Never the real SDK.

Nothing here spawns the Claude Code CLI or reaches the network: `query_fn` is
the one injection seam the module has, and every test uses it.
"""

import warnings
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    CanUseToolShadowedWarning,
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
    ToolPermissionContext,
    ToolUseBlock,
)

# Private, and imported on purpose: it is the function the SDK itself calls to
# decide whether to warn about a shadowed callback, so a test that calls it
# tests the real emission rather than a guess at it. Nothing is pre-allowed any
# more, and this is what proves there is genuinely nothing to warn about rather
# than a warning being swallowed somewhere.
from claude_agent_sdk.types import _warn_if_can_use_tool_shadowed

from agl.core.agent import (
    AgentBudgetError,
    AgentError,
    AgentOutputError,
    AgentQuestion,
    AgentSpec,
    Tool,
)
from agl.core.agent.impl.claude_runner import ClaudeRunner

SPEC = AgentSpec(prompt="do the thing", cwd=Path("/repo"), role="implement")


def messages(text: str = "done", **overrides: Any) -> list[Any]:
    fields: dict[str, Any] = {
        "subtype": "success",
        "duration_ms": 10,
        "duration_api_ms": 8,
        "is_error": False,
        "num_turns": 1,
        "session_id": "s-1",
        "total_cost_usd": 0.1,
        "terminal_reason": "completed",
    }
    fields.update(overrides)
    return [
        AssistantMessage(content=[TextBlock(text=text)], model="claude-sonnet-4-5"),
        ResultMessage(**fields),
    ]


class StubQuery:
    """A `query_fn` that replays scripted outcomes and records every call.

    An entry that is an exception is raised instead of streamed, which is how
    the retry ladder is exercised without a network anywhere near it.
    """

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.options: list[ClaudeAgentOptions] = []
        self.prompts: list[Any] = []

    @property
    def calls(self) -> int:
        return len(self.options)

    def __call__(self, *, prompt: Any, options: ClaudeAgentOptions) -> AsyncIterator[Any]:
        self.options.append(options)
        self.prompts.append(prompt)
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]

        async def stream() -> AsyncIterator[Any]:
            if isinstance(outcome, Exception):
                raise outcome
            for message in outcome:
                yield message

        return stream()


async def ask(
    options: ClaudeAgentOptions, tool: str, tool_input: dict[str, Any]
) -> Any:
    """Fire the permission callback the runner installed."""
    assert options.can_use_tool is not None
    return await options.can_use_tool(tool, tool_input, ToolPermissionContext())


QUESTION_INPUT = {
    "questions": [
        {
            "question": "Which storage?",
            "header": "Storage",
            "options": [
                {"label": "sqlite", "description": "One file on disk"},
                {"label": "postgres", "description": "A server"},
            ],
        }
    ]
}


# -- the happy path --------------------------------------------------------


async def test_a_successful_call_returns_the_folded_result() -> None:
    query = StubQuery(messages("all done"))

    result = await ClaudeRunner(query_fn=query).run(SPEC)

    assert result.text == "all done"
    assert result.cost_usd == 0.1
    assert result.session_id == "s-1"
    assert query.calls == 1


async def test_activity_reaches_the_caller() -> None:
    stream = messages()
    stream.insert(
        0,
        AssistantMessage(
            content=[ToolUseBlock(id="t", name="Read", input={"file_path": "a.py"})],
            model="claude-sonnet-4-5",
        ),
    )
    seen: list[str] = []

    await ClaudeRunner(query_fn=StubQuery(stream)).run(SPEC, on_activity=seen.append)

    assert seen == ["Read a.py"]


async def test_the_prompt_is_streamed_because_the_permission_callback_needs_it() -> None:
    # `can_use_tool` is rejected outright with a string prompt, so every call is
    # made in streaming mode whether or not it will end up asking anything.
    query = StubQuery(messages())

    await ClaudeRunner(query_fn=query).run(SPEC)

    (prompt,) = query.prompts
    sent = [message async for message in prompt]
    assert sent == [
        {
            "type": "user",
            "message": {"role": "user", "content": "do the thing"},
            "parent_tool_use_id": None,
            "session_id": "default",
        }
    ]


# -- options ---------------------------------------------------------------


async def test_a_spec_with_tools_produces_options_carrying_the_server() -> None:
    async def handler(arguments: dict[str, Any]) -> str:
        return str(arguments["a"] + arguments["b"])

    spec = AgentSpec(
        prompt="add them",
        cwd=Path("/repo"),
        role="implement",
        tools=(Tool(name="add", description="Add", schema={}, handler=handler),),
    )
    query = StubQuery(messages())

    await ClaudeRunner(query_fn=query).run(spec)

    (options,) = query.options
    assert set(options.mcp_servers) == {"agl"}


async def test_a_custom_tool_is_callable_without_being_pre_allowed() -> None:
    # Nothing goes into `allowed_tools` any more. The permission callback allows
    # every tool that is not the question tool, so a registered tool is callable
    # by the only route that decides: one round trip, and yes.
    query = StubQuery(messages())
    spec = AgentSpec(prompt="p", cwd=Path("/repo"), role="r", tools=(adding(),))

    await ClaudeRunner(query_fn=query).run(spec)

    (options,) = query.options
    assert options.allowed_tools == []
    decision = await ask(options, "mcp__agl__add", {"a": 1, "b": 2})
    assert isinstance(decision, PermissionResultAllow)


async def test_disallowed_tools_still_reach_the_options() -> None:
    # The asymmetry is deliberate: a deny rule resolves ahead of the callback
    # and holds even under `bypassPermissions`, and its pattern language is the
    # CLI's, not something worth rebuilding in Python.
    query = StubQuery(messages())
    spec = AgentSpec(
        prompt="p",
        cwd=Path("/repo"),
        role="r",
        disallowed_tools=("WebFetch", "Bash(git commit:*)"),
    )

    await ClaudeRunner(query_fn=query).run(spec)

    assert query.options[0].disallowed_tools == ["WebFetch", "Bash(git commit:*)"]


async def test_a_spec_with_no_tools_still_gets_a_server() -> None:
    # An in-process MCP server is what keeps the SDK's input stream open past
    # the prompt. Without one the CLI can close stdin before the permission
    # callback fires, and a question would hang instead of being asked.
    query = StubQuery(messages())

    await ClaudeRunner(query_fn=query).run(SPEC)

    (options,) = query.options
    assert set(options.mcp_servers) == {"agl"}
    assert options.allowed_tools == []


async def test_the_settings_path_reaches_the_options() -> None:
    query = StubQuery(messages())

    await ClaudeRunner(settings_path=Path("/etc/agl.json"), query_fn=query).run(SPEC)

    assert query.options[0].settings == "/etc/agl.json"


async def test_a_relative_settings_path_reaches_the_options_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `cwd` is the target repository. A relative settings path would resolve
    # inside it, so the module whose job is sealing that repo out would read
    # its settings from it.
    monkeypatch.chdir(tmp_path)
    query = StubQuery(messages())

    await ClaudeRunner(settings_path=Path("agl-settings.json"), query_fn=query).run(SPEC)

    settings = query.options[0].settings
    assert settings is not None
    assert Path(settings).is_absolute()
    assert Path(settings).parent == Path.cwd()
    assert Path(settings).name == "agl-settings.json"


# -- no global state is touched --------------------------------------------


def adding() -> Tool:
    async def handler(arguments: dict[str, Any]) -> str:
        return "4"

    return Tool(name="add", description="Add", schema={}, handler=handler)


class WarningQuery(StubQuery):
    """A `query_fn` that runs the SDK's own shadowing check on what it is given.

    The real `query` runs it while connecting, so this is the same call at the
    same point: whatever the SDK would have said about these options, it says
    here.
    """

    def __call__(self, *, prompt: Any, options: ClaudeAgentOptions) -> AsyncIterator[Any]:
        _warn_if_can_use_tool_shadowed(options)
        return super().__call__(prompt=prompt, options=options)


async def test_a_run_warns_about_nothing_because_it_shadows_nothing() -> None:
    # Nothing is pre-allowed any more, so the SDK has no shadowing to report and
    # there is no warning to suppress.
    spec = AgentSpec(prompt="p", cwd=Path("/repo"), role="r", tools=(adding(),))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await ClaudeRunner(query_fn=WarningQuery(messages())).run(spec)

    assert [w for w in caught if issubclass(w.category, CanUseToolShadowedWarning)] == []


async def test_a_run_installs_no_process_wide_warning_filter() -> None:
    # The filters are global and agent runs overlap. A module that mutates them
    # for the lifetime of a run is hiding warnings nobody asked it to hide.
    spec = AgentSpec(prompt="p", cwd=Path("/repo"), role="r", tools=(adding(),))
    before = list(warnings.filters)

    await ClaudeRunner(query_fn=StubQuery(messages())).run(spec)

    assert list(warnings.filters) == before


async def test_bypass_permissions_is_still_reported() -> None:
    # A real problem, and the SDK's to report: under `bypassPermissions` the
    # callback never fires for anything, so the run cannot ask a question at all.
    spec = AgentSpec(
        prompt="p", cwd=Path("/repo"), role="r", permission_mode="bypassPermissions"
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await ClaudeRunner(query_fn=WarningQuery(messages())).run(spec)

    assert [w for w in caught if issubclass(w.category, CanUseToolShadowedWarning)]


# -- questions -------------------------------------------------------------


async def test_on_question_is_awaited_and_its_answer_reaches_the_decision() -> None:
    asked: list[AgentQuestion] = []

    async def answer(question: AgentQuestion) -> str:
        asked.append(question)
        return "postgres"

    query = StubQuery(messages())
    await ClaudeRunner(query_fn=query).run(SPEC, on_question=answer)

    decision = await ask(query.options[0], "AskUserQuestion", QUESTION_INPUT)

    assert isinstance(decision, PermissionResultAllow)
    assert decision.updated_input is not None
    assert decision.updated_input["answers"] == {"Which storage?": "postgres"}
    assert [question.title for question in asked] == ["Which storage?"]
    assert [option.label for option in asked[0].options] == ["sqlite", "postgres"]


async def test_every_question_in_one_call_is_asked() -> None:
    asked: list[AgentQuestion] = []

    async def answer(question: AgentQuestion) -> str:
        asked.append(question)
        return f"answer to {question.title}"

    tool_input = {
        "questions": [
            {"question": "First?", "options": [{"label": "a", "description": "A"}]},
            {"question": "Second?", "options": [{"label": "b", "description": "B"}]},
        ]
    }
    query = StubQuery(messages())
    await ClaudeRunner(query_fn=query).run(SPEC, on_question=answer)

    decision = await ask(query.options[0], "AskUserQuestion", tool_input)

    assert [question.title for question in asked] == ["First?", "Second?"]
    assert isinstance(decision, PermissionResultAllow)
    assert decision.updated_input is not None
    assert decision.updated_input["answers"] == {
        "First?": "answer to First?",
        "Second?": "answer to Second?",
    }


async def test_the_answers_are_keyed_by_the_question_text_not_the_header() -> None:
    # Up to four questions can share a header — it is a chip in the UI — and
    # keying by it would drop every answer but the last.
    async def answer(question: AgentQuestion) -> str:
        return question.title.lower()

    tool_input = {
        "questions": [
            {"question": "Which store?", "header": "Design", "options": []},
            {"question": "Which cache?", "header": "Design", "options": []},
        ]
    }
    query = StubQuery(messages())
    await ClaudeRunner(query_fn=query).run(SPEC, on_question=answer)

    decision = await ask(query.options[0], "AskUserQuestion", tool_input)

    assert isinstance(decision, PermissionResultAllow)
    assert decision.updated_input is not None
    assert decision.updated_input["answers"] == {
        "Which store?": "which store?",
        "Which cache?": "which cache?",
    }


async def test_a_repeated_question_is_asked_once_and_answered_for_both() -> None:
    # Nothing stops the model asking the same thing twice in one call, and the
    # answers map is keyed by the question text, so there is nowhere to put a
    # second answer. Asking a person the same string twice to then throw one of
    # the two replies away is worse than asking once.
    asked: list[AgentQuestion] = []

    async def answer(question: AgentQuestion) -> str:
        asked.append(question)
        return f"answer {len(asked)}"

    tool_input = {
        "questions": [
            {"question": "Which store?", "options": [{"label": "a", "description": "A"}]},
            {"question": "Which store?", "options": [{"label": "b", "description": "B"}]},
        ]
    }
    query = StubQuery(messages())
    await ClaudeRunner(query_fn=query).run(SPEC, on_question=answer)

    decision = await ask(query.options[0], "AskUserQuestion", tool_input)

    assert [question.title for question in asked] == ["Which store?"]
    assert isinstance(decision, PermissionResultAllow)
    assert decision.updated_input is not None
    assert decision.updated_input["answers"] == {"Which store?": "answer 1"}


async def test_a_repeated_question_leaves_the_questions_untouched() -> None:
    # Both entries stay in the input the tool receives: it is the tool's job to
    # match answers to questions, and it does that by text.
    async def answer(question: AgentQuestion) -> str:
        return "sqlite"

    tool_input = {
        "questions": [
            {"question": "Which store?", "options": []},
            {"question": "Which store?", "options": []},
        ]
    }
    query = StubQuery(messages())
    await ClaudeRunner(query_fn=query).run(SPEC, on_question=answer)

    decision = await ask(query.options[0], "AskUserQuestion", tool_input)

    assert isinstance(decision, PermissionResultAllow)
    assert decision.updated_input is not None
    assert decision.updated_input["questions"] == tool_input["questions"]


async def test_the_questions_survive_into_the_updated_input() -> None:
    async def answer(question: AgentQuestion) -> str:
        return "sqlite"

    query = StubQuery(messages())
    await ClaudeRunner(query_fn=query).run(SPEC, on_question=answer)

    decision = await ask(query.options[0], "AskUserQuestion", QUESTION_INPUT)

    assert isinstance(decision, PermissionResultAllow)
    assert decision.updated_input is not None
    assert decision.updated_input["questions"] == QUESTION_INPUT["questions"]


async def test_no_handler_denies_rather_than_hanging() -> None:
    query = StubQuery(messages())
    await ClaudeRunner(query_fn=query).run(SPEC, on_question=None)

    decision = await ask(query.options[0], "AskUserQuestion", QUESTION_INPUT)

    assert isinstance(decision, PermissionResultDeny)
    assert "judgment" in decision.message
    assert decision.interrupt is False


async def test_every_other_tool_is_allowed_unchanged() -> None:
    query = StubQuery(messages())
    await ClaudeRunner(query_fn=query).run(SPEC)

    decision = await ask(query.options[0], "Bash", {"command": "ls"})

    assert isinstance(decision, PermissionResultAllow)
    assert decision.updated_input is None


# -- retry -----------------------------------------------------------------


async def test_two_failures_then_a_success_returns_after_three_calls() -> None:
    query = StubQuery(
        RuntimeError("transport died"), RuntimeError("transport died"), messages("third")
    )

    result = await ClaudeRunner(query_fn=query).run(SPEC)

    assert result.text == "third"
    assert query.calls == 3


async def test_a_stub_that_always_raises_exhausts_the_attempts() -> None:
    query = StubQuery(RuntimeError("transport died"))

    with pytest.raises(AgentError) as raised:
        await ClaudeRunner(query_fn=query).run(SPEC)

    assert query.calls == 3
    assert "transport died" in str(raised.value)
    assert isinstance(raised.value.__cause__, RuntimeError)


async def test_max_attempts_is_honoured() -> None:
    query = StubQuery(RuntimeError("nope"))

    with pytest.raises(AgentError):
        await ClaudeRunner(max_attempts=5, query_fn=query).run(SPEC)

    assert query.calls == 5


@pytest.mark.parametrize("attempts", [0, -1])
def test_fewer_than_one_attempt_is_refused_at_construction(attempts: int) -> None:
    # A ladder that never runs leaves no failure to report, so the error would
    # read "failed after 0 attempts: NoneType: None".
    with pytest.raises(ValueError, match="max_attempts"):
        ClaudeRunner(max_attempts=attempts)


async def test_one_attempt_is_a_legitimate_choice() -> None:
    query = StubQuery(RuntimeError("nope"))

    with pytest.raises(AgentError, match="transport|nope"):
        await ClaudeRunner(max_attempts=1, query_fn=query).run(SPEC)

    assert query.calls == 1


async def test_a_stream_that_never_resolves_is_retried() -> None:
    query = StubQuery([], messages("second time lucky"))

    result = await ClaudeRunner(query_fn=query).run(SPEC)

    assert result.text == "second time lucky"
    assert query.calls == 2


# -- exhaustion is not retried ---------------------------------------------


async def test_budget_exhaustion_raises_immediately() -> None:
    query = StubQuery(messages(subtype="error_max_budget_usd", is_error=True))

    with pytest.raises(AgentBudgetError):
        await ClaudeRunner(query_fn=query).run(SPEC)

    # Retrying spends the budget again for the same outcome.
    assert query.calls == 1


async def test_turn_exhaustion_raises_immediately() -> None:
    query = StubQuery(messages(subtype="error_max_turns", is_error=True))

    with pytest.raises(AgentBudgetError):
        await ClaudeRunner(query_fn=query).run(SPEC)

    assert query.calls == 1


async def test_the_budget_error_says_which_limit_was_hit() -> None:
    query = StubQuery(messages(subtype="error_max_budget_usd", is_error=True))

    with pytest.raises(AgentBudgetError, match="error_max_budget_usd"):
        await ClaudeRunner(query_fn=query).run(SPEC)


# -- an output parse failure is not retried either --------------------------


async def test_an_output_parse_failure_raises_immediately() -> None:
    # A model that answered in prose instead of JSON gives the same prose back
    # on an identical retry — retrying spends the budget again for nothing.
    spec = AgentSpec(
        prompt="p", cwd=Path("/repo"), role="r", output_schema={"type": "object"}
    )
    query = StubQuery(messages("not json at all"))

    with pytest.raises(AgentOutputError):
        await ClaudeRunner(query_fn=query).run(spec)

    assert query.calls == 1


# -- an error result is a failure, and failures are retried -----------------


async def test_an_error_result_is_retried_rather_than_returned() -> None:
    # The most common SDK failure class. Returning it as a success is the ABC
    # breaking its own promise to raise when a run could not be completed.
    query = StubQuery(
        messages(subtype="error_during_execution", is_error=True), messages("second time")
    )

    result = await ClaudeRunner(query_fn=query).run(SPEC)

    assert result.text == "second time"
    assert query.calls == 2


async def test_three_error_results_exhaust_the_attempts() -> None:
    query = StubQuery(messages(subtype="error_during_execution", is_error=True))

    with pytest.raises(AgentError) as raised:
        await ClaudeRunner(query_fn=query).run(SPEC)

    assert query.calls == 3
    assert "error_during_execution" in str(raised.value)


async def test_the_error_raised_for_an_error_result_is_not_a_budget_error() -> None:
    # Retryability is the whole distinction, and `AgentBudgetError` is the one
    # thing the ladder refuses to retry.
    query = StubQuery(messages(subtype="error_during_execution", is_error=True))

    with pytest.raises(AgentError) as raised:
        await ClaudeRunner(max_attempts=1, query_fn=query).run(SPEC)

    assert not isinstance(raised.value, AgentBudgetError)
