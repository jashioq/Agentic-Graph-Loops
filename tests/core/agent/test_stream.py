"""Folding a message stream into a result.

Pure over an iterable of messages, so the messages are the SDK's own dataclasses
built by hand — real objects, no network, no CLI.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock

from agl.core.agent import AgentBudgetError, AgentError, AgentOutputError
from agl.core.agent.impl.stream import fold, summarize_tool_use


def assistant(*blocks: Any, session_id: str | None = None) -> AssistantMessage:
    return AssistantMessage(
        content=list(blocks), model="claude-sonnet-4-5", session_id=session_id
    )


def said(text: str, session_id: str | None = None) -> AssistantMessage:
    return assistant(TextBlock(text=text), session_id=session_id)


def used(name: str, **tool_input: Any) -> AssistantMessage:
    return assistant(ToolUseBlock(id="t-1", name=name, input=tool_input))


def result(**overrides: Any) -> ResultMessage:
    fields: dict[str, Any] = {
        "subtype": "success",
        "duration_ms": 1200,
        "duration_api_ms": 900,
        "is_error": False,
        "num_turns": 3,
        "session_id": "s-1",
        "total_cost_usd": 0.42,
        "terminal_reason": "completed",
    }
    fields.update(overrides)
    return ResultMessage(**fields)


async def stream(*messages: Any) -> AsyncIterator[Any]:
    for message in messages:
        yield message


# -- text ------------------------------------------------------------------


async def test_the_final_assistant_text_wins() -> None:
    folded = await fold(stream(said("first"), said("second"), result()), None, False)
    assert folded.text == "second"


async def test_several_text_blocks_in_one_message_accumulate() -> None:
    message = assistant(TextBlock(text="one"), TextBlock(text="two"))
    folded = await fold(stream(message, result()), None, False)
    assert folded.text == "one\ntwo"


async def test_a_message_carrying_no_text_does_not_wipe_what_came_before() -> None:
    folded = await fold(
        stream(said("kept"), used("Read", file_path="README.md"), result()), None, False
    )
    assert folded.text == "kept"


# -- session ---------------------------------------------------------------


async def test_the_session_id_is_captured_from_the_first_message_carrying_one() -> None:
    folded = await fold(
        stream(said("a", session_id="s-first"), said("b", session_id="s-later"), result()),
        None,
        False,
    )
    assert folded.session_id == "s-first"


async def test_the_result_supplies_the_session_id_when_nothing_earlier_did() -> None:
    folded = await fold(stream(said("a"), result(session_id="s-9")), None, False)
    assert folded.session_id == "s-9"


# -- activity --------------------------------------------------------------


async def test_every_tool_use_fires_on_activity_once_in_order() -> None:
    seen: list[str] = []
    messages = stream(
        used("Read", file_path="README.md"),
        used("Bash", command="./gradlew build"),
        said("done"),
        result(),
    )

    await fold(messages, seen.append, False)

    assert seen == ["Read README.md", "Bash ./gradlew build"]


async def test_two_tool_uses_in_one_message_both_fire() -> None:
    seen: list[str] = []
    message = assistant(
        ToolUseBlock(id="a", name="Read", input={"file_path": "a.py"}),
        ToolUseBlock(id="b", name="Read", input={"file_path": "b.py"}),
    )

    await fold(stream(message, result()), seen.append, False)

    assert seen == ["Read a.py", "Read b.py"]


async def test_a_broken_dashboard_does_not_fail_an_agent_run() -> None:
    def explode(activity: str) -> None:
        raise RuntimeError("the footer is on fire")

    folded = await fold(stream(used("Read", file_path="a.py"), result()), explode, False)

    assert folded.terminal_reason == "completed"


# -- the result message ----------------------------------------------------


async def test_the_result_fields_map_onto_the_agent_result() -> None:
    folded = await fold(stream(said("done"), result()), None, False)

    assert folded.cost_usd == 0.42
    assert folded.num_turns == 3
    assert folded.duration_ms == 1200
    assert folded.terminal_reason == "completed"


async def test_a_missing_cost_reads_as_nothing_spent() -> None:
    folded = await fold(stream(said("done"), result(total_cost_usd=None)), None, False)
    assert folded.cost_usd == 0.0


async def test_an_error_result_raises_rather_than_coming_back_as_a_success() -> None:
    # An execution error is the most common way an SDK run fails. Handing it
    # back as a result is what let it past the retry ladder.
    with pytest.raises(AgentError, match="error_during_execution"):
        await fold(
            stream(said("nope"), result(is_error=True, subtype="error_during_execution")),
            None,
            False,
        )


async def test_the_error_carries_whatever_the_result_said_went_wrong() -> None:
    with pytest.raises(AgentError, match="the tool call blew up"):
        await fold(
            stream(
                result(
                    is_error=True,
                    subtype="error_during_execution",
                    result="the tool call blew up",
                )
            ),
            None,
            False,
        )


async def test_an_error_result_is_not_a_budget_error() -> None:
    with pytest.raises(AgentError) as raised:
        await fold(
            stream(result(is_error=True, subtype="error_during_execution")), None, False
        )

    assert not isinstance(raised.value, AgentBudgetError)


async def test_an_exhaustion_result_is_left_for_the_caller_to_classify() -> None:
    # `terminal_reason` is absent on older CLIs and says nothing about budget on
    # newer ones, so the subtype is the specific answer when it names a limit.
    # Exhaustion is not retryable and the caller says so with its own error, so
    # it comes back as a result rather than being raised here.
    folded = await fold(
        stream(said("…"), result(is_error=True, subtype="error_max_budget_usd")),
        None,
        False,
    )

    assert folded.terminal_reason == "error_max_budget_usd"


async def test_a_turn_limit_result_is_left_alone_too() -> None:
    folded = await fold(
        stream(said("…"), result(is_error=True, subtype="error_max_turns")), None, False
    )

    assert folded.terminal_reason == "error_max_turns"


async def test_an_ordinary_subtype_leaves_the_terminal_reason_alone() -> None:
    folded = await fold(stream(said("hi"), result(terminal_reason="completed")), None, False)
    assert folded.terminal_reason == "completed"


# -- structured output -----------------------------------------------------


async def test_bare_json_parses() -> None:
    folded = await fold(stream(said('{"ok": true}'), result()), None, True)
    assert folded.structured == {"ok": True}


async def test_fenced_json_parses() -> None:
    fenced = '```json\n{"ok": true, "count": 2}\n```'
    folded = await fold(stream(said(fenced), result()), None, True)
    assert folded.structured == {"ok": True, "count": 2}


async def test_an_unlabelled_fence_parses() -> None:
    folded = await fold(stream(said('```\n[1, 2, 3]\n```'), result()), None, True)
    assert folded.structured == [1, 2, 3]


async def test_malformed_json_raises() -> None:
    with pytest.raises(AgentError, match="JSON"):
        await fold(stream(said("not json at all"), result()), None, True)


async def test_malformed_json_raises_the_output_specific_error() -> None:
    # Distinct from the bare `AgentError` other failures raise here: this one
    # is not worth retrying, and the caller tells the two apart by type.
    with pytest.raises(AgentOutputError):
        await fold(stream(said("not json at all"), result()), None, True)


async def test_nothing_is_parsed_when_no_schema_was_asked_for() -> None:
    folded = await fold(stream(said('{"ok": true}'), result()), None, False)
    assert folded.structured is None


# -- a stream that never resolved -----------------------------------------


async def test_an_empty_stream_raises_rather_than_half_building_a_result() -> None:
    with pytest.raises(AgentError):
        await fold(stream(), None, False)


async def test_a_stream_that_ends_without_a_result_raises() -> None:
    with pytest.raises(AgentError, match="result"):
        await fold(stream(said("half a thought")), None, False)


# -- summarize_tool_use ----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "tool_input", "expected"),
    [
        ("Edit", {"file_path": "src/auth/TokenStore.kt"}, "Edit src/auth/TokenStore.kt"),
        ("Write", {"file_path": "notes.md"}, "Write notes.md"),
        ("Read", {"file_path": "README.md"}, "Read README.md"),
        ("Bash", {"command": "./gradlew build"}, "Bash ./gradlew build"),
        ("Grep", {"pattern": "TODO"}, "Grep TODO"),
        ("Glob", {"pattern": "**/*.kt"}, "Glob **/*.kt"),
        ("WebFetch", {"url": "https://example.com"}, "WebFetch https://example.com"),
        ("Task", {"description": "find the bug"}, "Task find the bug"),
    ],
)
def test_each_common_tool_gets_its_subject(
    name: str, tool_input: dict[str, Any], expected: str
) -> None:
    assert summarize_tool_use(name, tool_input) == expected


def test_a_custom_tool_loses_the_namespace() -> None:
    assert summarize_tool_use("mcp__agl__echo", {"text": "hello"}) == "echo hello"


def test_a_custom_tool_with_no_parameters_is_just_its_name() -> None:
    assert summarize_tool_use("mcp__agl__get_document", {}) == "get_document"


def test_an_input_with_no_obvious_subject_falls_back_to_the_bare_name() -> None:
    assert summarize_tool_use("TodoWrite", {"todos": [1, 2, 3]}) == "TodoWrite"


def test_a_multi_line_command_stays_on_one_line() -> None:
    summary = summarize_tool_use("Bash", {"command": "cd src\nmake test"})
    assert "\n" not in summary
    assert summary == "Bash cd src make test"


def test_a_long_path_is_truncated_in_the_middle_so_the_filename_survives() -> None:
    path = "src/main/kotlin/com/example/deeply/nested/package/TokenStoreImpl.kt"
    summary = summarize_tool_use("Edit", {"file_path": path})

    assert len(summary) <= 60
    assert summary.startswith("Edit src/")
    assert summary.endswith("TokenStoreImpl.kt")
    assert "…" in summary
