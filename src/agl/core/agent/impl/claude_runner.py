"""`AgentRunner` over the Claude Agent SDK. The only code here that does I/O.

Layer: core. Thin: `options` configures, `tools` builds the server, `stream`
folds the messages; what is left is the retry ladder and the question callback.

Every call runs in streaming mode, because the SDK rejects a permission callback
on a plain-string prompt and that callback is how a question reaches the caller.
`AskUserQuestion` is unavailable to subagents, so a call that may ask must be a
top-level one. Exhaustion and a bad `output_schema` parse are never retried —
both fail identically next time; everything else goes round the ladder. Nothing
is pre-allowed: the callback allows every tool that is not the question tool.
"""

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    PermissionResult,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
    query,
)

from agl.core.agent.api import (
    AgentBudgetError,
    AgentError,
    AgentOption,
    AgentOutputError,
    AgentQuestion,
    AgentResult,
    AgentRunner,
    AgentSpec,
)
from agl.core.agent.impl.options import QUESTION_TOOL, build_options
from agl.core.agent.impl.stream import EXHAUSTED, fold
from agl.core.agent.impl.tools import build_keepalive_server, build_tool_server

__all__ = ["ClaudeRunner"]

NO_ANSWERS = (
    "Nobody is available to answer questions on this run. Proceed with your "
    "best judgment and say in your final response what you decided and why."
)


class ClaudeRunner(AgentRunner):
    """Runs a spec through the SDK, retrying what is worth retrying."""

    def __init__(
        self,
        settings_path: Path | None = None,
        max_attempts: int = 3,
        query_fn: Callable[..., AsyncIterator[Any]] | None = None,
    ) -> None:
        """Builds a runner over the SDK.

        param: settings_path - made absolute, not resolved: a run's `cwd` is the
            target repository, and a relative path would be read from inside it
        param: max_attempts - at least 1, or `ValueError`
        param: query_fn - the SDK's `query` by default; the module's one seam
        """
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")

        self._settings_path = settings_path.absolute() if settings_path is not None else None
        self._max_attempts = max_attempts
        self._query = query_fn if query_fn is not None else query

    async def run(
        self,
        spec: AgentSpec,
        on_activity: Callable[[str], None] | None = None,
        on_question: Callable[[AgentQuestion], Awaitable[str]] | None = None,
    ) -> AgentResult:
        options = self._options(spec, on_question)
        failure: Exception | None = None

        for _ in range(self._max_attempts):
            try:
                result = await self._attempt(spec, options, on_activity)
            except (AgentBudgetError, AgentOutputError):
                raise
            except Exception as error:  # noqa: BLE001 - transport, API, or ours
                failure = error
                continue
            return result

        raise AgentError(
            f"agent run {spec.role!r} failed after {self._max_attempts} attempts: "
            f"{type(failure).__name__}: {failure}"
        ) from failure

    async def _attempt(
        self,
        spec: AgentSpec,
        options: ClaudeAgentOptions,
        on_activity: Callable[[str], None] | None,
    ) -> AgentResult:
        """One call. Raises `AgentBudgetError`, `AgentOutputError`, or `AgentError`.

        Only the first two are exempt from the retry ladder.
        """
        stream = self._query(prompt=_streamed(spec.prompt), options=options)
        result = await fold(stream, on_activity, spec.output_schema is not None)

        if result.terminal_reason in EXHAUSTED:
            raise AgentBudgetError(
                f"agent run {spec.role!r} stopped at {result.terminal_reason} "
                f"after {result.num_turns} turns and ${result.cost_usd:.2f}"
            )
        return result

    def _options(
        self,
        spec: AgentSpec,
        on_question: Callable[[AgentQuestion], Awaitable[str]] | None,
    ) -> ClaudeAgentOptions:
        """The SDK options for this call, with the question callback installed."""
        servers, _names = build_tool_server(spec.tools)
        options = build_options(
            spec,
            settings_path=self._settings_path,
            mcp_servers=servers or build_keepalive_server(),
        )
        options.can_use_tool = _permission_handler(on_question)
        return options


def _streamed(prompt: str) -> AsyncIterator[dict[str, Any]]:
    """The prompt as the one-message stream the SDK's streaming mode expects."""

    async def messages() -> AsyncIterator[dict[str, Any]]:
        yield {
            "type": "user",
            "message": {"role": "user", "content": prompt},
            "parent_tool_use_id": None,
            "session_id": "default",
        }

    return messages()


def _permission_handler(
    on_question: Callable[[AgentQuestion], Awaitable[str]] | None,
) -> Callable[[str, dict[str, Any], ToolPermissionContext], Awaitable[PermissionResult]]:
    """A permission callback that intercepts questions and allows everything else.

    Not a permission policy — the options decide that. It exists only because
    the SDK routes `AskUserQuestion` through this same channel.
    """

    async def can_use_tool(
        tool: str, tool_input: dict[str, Any], context: ToolPermissionContext
    ) -> PermissionResult:
        if tool != QUESTION_TOOL:
            return PermissionResultAllow()
        if on_question is None:
            return PermissionResultDeny(message=NO_ANSWERS)

        answers: dict[str, str] = {}
        for entry in tool_input.get("questions", []):
            question = _question(entry)
            # Two questions in one call can share their text, and the answers
            # map is keyed by it — so ask once and let both take the answer
            # rather than putting the same string in front of a person twice.
            if question.title in answers:
                continue
            answers[question.title] = await on_question(question)

        return PermissionResultAllow(updated_input={**tool_input, "answers": answers})

    return can_use_tool


def _question(entry: Mapping[str, Any]) -> AgentQuestion:
    """One entry of the tool's input as this module's own question type.

    `title` must be the question text, since the answers map is keyed by it.
    """
    return AgentQuestion(
        title=str(entry.get("question") or entry.get("header") or ""),
        options=tuple(
            AgentOption(
                label=str(option.get("label", "")),
                description=str(option.get("description", "")),
            )
            for option in entry.get("options", [])
        ),
    )
