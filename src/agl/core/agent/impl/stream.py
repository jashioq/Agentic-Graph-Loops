"""The SDK's message stream, folded into one `AgentResult`.

Layer: core. A function of the messages that arrived; the only side effect is
`on_activity`, wrapped so a broken dashboard cannot fail a run.

Token counts on assistant messages exclude subagent spend, so trust `cost_usd`.
A result the SDK marked as an error is raised, since `AgentResult` cannot say
"this failed" — except exhaustion, which comes back as a result for the caller
to classify.
"""

from collections.abc import AsyncIterable, Callable, Mapping
from typing import Any

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolUseBlock

from agl.core.agent.api import AgentError, AgentResult
from agl.core.agent.impl.tools import MCP_PREFIX

__all__ = ["EXHAUSTED", "fold", "summarize_tool_use"]

# `terminal_reason` values that mean the run hit a ceiling rather than went
# wrong. `max_turns` is the SDK's own wording; the `error_max_` pair is what a
# result message reports as its subtype.
EXHAUSTED = frozenset({"max_turns", "error_max_turns", "error_max_budget_usd"})

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


def summarize_tool_use(tool: str, tool_input: Mapping[str, Any]) -> str:
    """A one-line description of a tool call, for a status footer.

    return: str - `Edit src/auth/TokenStore.kt`, or the bare tool name if it has no subject
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
) -> AgentResult:
    """Consumes the stream and returns what the run produced.

    return: AgentResult - raises `AgentError` if the stream never resolved or
        reported a non-exhaustion error
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

    reason = _terminal_reason(outcome)
    if outcome.is_error and reason not in EXHAUSTED:
        detail = (outcome.result or text).strip()
        raise AgentError(
            f"the run reported an error ({outcome.subtype}) after "
            f"{outcome.num_turns} turns and ${outcome.total_cost_usd or 0.0:.2f}"
            + (f": {detail}" if detail else "")
        )

    return AgentResult(
        text=text,
        session_id=session_id,
        cost_usd=outcome.total_cost_usd or 0.0,
        num_turns=outcome.num_turns,
        duration_ms=outcome.duration_ms,
        terminal_reason=reason,
    )


def _report(on_activity: Callable[[str], None] | None, block: ToolUseBlock) -> None:
    """Tell the dashboard what is happening, and never let it be the reason a run fails."""
    if on_activity is None:
        return
    try:
        on_activity(summarize_tool_use(block.name, block.input))
    except Exception:  # noqa: BLE001 - a footer is not worth an agent run
        pass


def _terminal_reason(outcome: ResultMessage) -> str | None:
    """Why the loop stopped, preferring the subtype when it names a limit.

    `terminal_reason` never mentions the budget; the `error_max_` subtypes do,
    and that difference is the difference between retrying and not.
    """
    if outcome.subtype.startswith("error_max_"):
        return outcome.subtype
    return outcome.terminal_reason
