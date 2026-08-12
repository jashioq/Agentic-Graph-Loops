"""The SDK's message stream, folded into one `AgentResult`.

Layer: core. Everything here is a function of the messages that arrived; the
only side effect is `on_activity`, which is fire-and-forget UI and is wrapped so
a broken dashboard can never fail an agent run.

Token counts on assistant messages cover the top-level loop only and exclude
anything a subagent spent, so they understate a run that delegated. `cost_usd`
comes from the result message and is the number to trust.
"""

import json
from collections.abc import AsyncIterable, Callable, Mapping
from typing import Any

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock

from agl.core.agent.api import AgentError, AgentResult

__all__ = ["fold", "summarise_tool_use"]

MCP_PREFIX = "mcp__agl__"
SUMMARY_WIDTH = 60

# In priority order: the first one present is what the tool is doing *to*.
SUBJECT_KEYS = (
    "file_path",
    "notebook_path",
    "path",
    "command",
    "pattern",
    "url",
    "query",
    "description",
    "text",
)


def summarise_tool_use(tool: str, tool_input: Mapping[str, Any]) -> str:
    """A short one-line description of a tool call, for a status footer.

    `Edit src/auth/TokenStore.kt`, `Bash ./gradlew build`, `Read README.md`. An
    input with no obvious subject gives the bare tool name, which is still more
    informative than a blank footer.
    """
    name = tool.removeprefix(MCP_PREFIX)
    subject = next(
        (str(tool_input[key]) for key in SUBJECT_KEYS if tool_input.get(key)), ""
    )
    if not subject:
        return name

    subject = " ".join(subject.split())
    return f"{name} {_shorten(subject, SUMMARY_WIDTH - len(name) - 1)}"


def _shorten(text: str, limit: int) -> str:
    """Cut out of the middle, keeping the tail — a path's filename lives there."""
    if len(text) <= limit or limit < 3:
        return text
    keep = limit - 1
    head = keep // 3
    return f"{text[:head]}…{text[head - keep:]}"


async def fold(
    messages: AsyncIterable[Any],
    on_activity: Callable[[str], None] | None,
    expect_json: bool,
) -> AgentResult:
    """Consume the stream and return what the run produced.

    Raises `AgentError` if the stream ends without a result message — a run that
    never resolved is not a result with fields missing — or if `expect_json` was
    asked for and the final text does not parse.
    """
    text = ""
    session_id: str | None = None
    outcome: ResultMessage | None = None

    async for message in messages:
        if session_id is None:
            session_id = getattr(message, "session_id", None)

        if isinstance(message, AssistantMessage):
            blocks = [b.text for b in message.content if isinstance(b, TextBlock)]
            if blocks:
                text = "\n".join(blocks)
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    _report(on_activity, block)

        elif isinstance(message, ResultMessage):
            outcome = message

    if outcome is None:
        raise AgentError("the message stream ended without a result")

    return AgentResult(
        text=text,
        structured=_parse(text) if expect_json else None,
        session_id=session_id,
        cost_usd=outcome.total_cost_usd or 0.0,
        num_turns=outcome.num_turns,
        duration_ms=outcome.duration_ms,
        terminal_reason=_terminal_reason(outcome),
        is_error=outcome.is_error,
    )


def _report(on_activity: Callable[[str], None] | None, block: ToolUseBlock) -> None:
    """Tell the dashboard what is happening, and never let it be the reason a run fails."""
    if on_activity is None:
        return
    try:
        on_activity(summarise_tool_use(block.name, block.input))
    except Exception:  # noqa: BLE001 - a footer is not worth an agent run
        pass


def _terminal_reason(outcome: ResultMessage) -> str | None:
    """Why the loop stopped, preferring the subtype when it names a limit.

    `terminal_reason` is `None` on older CLIs and says `"completed"` or
    `"max_turns"` on newer ones — it never mentions the budget. The `error_max_`
    subtypes do, and telling budget exhaustion apart from an ordinary failure is
    the difference between retrying and not.
    """
    if outcome.subtype.startswith("error_max_"):
        return outcome.subtype
    return outcome.terminal_reason


def _parse(text: str) -> Any:
    """The text as JSON, with any code fence stripped first."""
    stripped = text.strip()
    if stripped.startswith("```"):
        _, _, rest = stripped.partition("\n")
        stripped = rest.rpartition("```")[0].strip() or rest.strip()
    try:
        return json.loads(stripped)
    except ValueError as error:
        raise AgentError(f"expected JSON output, got {text!r}: {error}") from error
