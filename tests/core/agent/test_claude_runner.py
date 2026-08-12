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
# tests the real emission rather than a guess at its wording. If the SDK renames
# it, the message our filter matches has very likely changed too, and this
# failing is how we find out.
from claude_agent_sdk.types import _warn_if_can_use_tool_shadowed

from agl.core.agent import (
    AgentBudgetError,
    AgentError,
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
    assert options.allowed_tools == ["mcp__agl__add"]


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


# -- the shadowing warning -------------------------------------------------
#
# Our own MCP tools are in `allowed_tools`, which auto-approves them ahead of
# `can_use_tool` — the SDK calls that shadowing and says so on stderr. For our
# tools it is deliberate; the warning is not, because stderr belongs to
# `rich.Live` once a dashboard is up.


class WarningQuery(StubQuery):
    """A `query_fn` that runs the SDK's own shadowing check on what it is given.

    The real `query` runs it while connecting, so this is the same call at the
    same point: whatever the SDK would have said about these options, it says
    here.
    """

    def __call__(self, *, prompt: Any, options: ClaudeAgentOptions) -> AsyncIterator[Any]:
        _warn_if_can_use_tool_shadowed(options)
        return super().__call__(prompt=prompt, options=options)


def adding() -> Tool:
    async def handler(arguments: dict[str, Any]) -> str:
        return "4"

    return Tool(name="add", description="Add", schema={}, handler=handler)


async def test_shadowing_by_our_own_tools_is_not_reported() -> None:
    spec = AgentSpec(prompt="p", cwd=Path("/repo"), role="r", tools=(adding(),))

    with warnings.catch_warnings(record=True) as caught:
        await ClaudeRunner(query_fn=WarningQuery(messages())).run(spec)

    assert [w for w in caught if issubclass(w.category, CanUseToolShadowedWarning)] == []


async def test_a_run_with_no_tools_reports_nothing_either() -> None:
    with warnings.catch_warnings(record=True) as caught:
        await ClaudeRunner(query_fn=WarningQuery(messages())).run(SPEC)

    assert [w for w in caught if issubclass(w.category, CanUseToolShadowedWarning)] == []


async def test_shadowing_by_a_tool_we_did_not_register_is_still_reported() -> None:
    # The suppression is matched on the message, not the category: a spec that
    # allows `Read` outright really has disabled the callback for `Read`, and
    # that is information we did not have before.
    spec = AgentSpec(
        prompt="p", cwd=Path("/repo"), role="r", allowed_tools=("Read",), tools=(adding(),)
    )

    with warnings.catch_warnings(record=True) as caught:
        await ClaudeRunner(query_fn=WarningQuery(messages())).run(spec)

    shadowing = [w for w in caught if issubclass(w.category, CanUseToolShadowedWarning)]
    assert len(shadowing) == 1
    assert "Read" in str(shadowing[0].message)


async def test_bypass_permissions_is_still_reported() -> None:
    # A different message, and a real problem: under `bypassPermissions` the
    # callback never fires for anything, so the run cannot ask a question at all.
    spec = AgentSpec(
        prompt="p", cwd=Path("/repo"), role="r", permission_mode="bypassPermissions"
    )

    with warnings.catch_warnings(record=True) as caught:
        await ClaudeRunner(query_fn=WarningQuery(messages())).run(spec)

    assert [w for w in caught if issubclass(w.category, CanUseToolShadowedWarning)]


async def test_the_suppression_does_not_outlive_what_it_was_for() -> None:
    # Anyone else's shadowing warning, after a run has installed the filter, is
    # still shown.
    spec = AgentSpec(prompt="p", cwd=Path("/repo"), role="r", tools=(adding(),))

    with warnings.catch_warnings(record=True) as caught:
        await ClaudeRunner(query_fn=WarningQuery(messages())).run(spec)
        warnings.warn(
            "can_use_tool will not be invoked for: SomeoneElsesTool.",
            CanUseToolShadowedWarning,
            stacklevel=1,
        )

    assert [str(w.message) for w in caught] == [
        "can_use_tool will not be invoked for: SomeoneElsesTool."
    ]


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
    assert "judgement" in decision.message
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


async def test_an_ordinary_error_result_is_returned_not_raised() -> None:
    query = StubQuery(messages(subtype="error_during_execution", is_error=True))

    result = await ClaudeRunner(query_fn=query).run(SPEC)

    assert result.is_error is True
    assert query.calls == 1
