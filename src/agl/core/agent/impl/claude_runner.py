"""`AgentRunner` over the Claude Agent SDK. The only code here that does I/O.

Layer: core. Thin by design: `options` decides the configuration, `tools` builds
the server, `stream` folds the messages, and what is left is the retry ladder,
the question callback, and calling `query`.

**Every call runs in streaming mode.** The SDK rejects a permission callback
outright when the prompt is a plain string, and the callback is how a question
reaches the caller — so the prompt is always sent as a one-message stream, and
an in-process MCP server is always registered so the SDK holds stdin open until
the run resolves (see `build_keepalive_server`).

**`AskUserQuestion` is not available to subagents.** Anything spawned through
the Agent tool cannot ask, so a call that may need to ask has to be a top-level
one. That is a constraint on how callers are shaped, not a bug here.

Exhaustion and a bad `output_schema` parse are the two failures never retried:
both fail the same way next time and spend the budget again, so three attempts
would cost three times as much for the same outcome — a run out of budget
stays out of budget, and a model that answered in prose instead of JSON gives
the same prose back. Every other failure — a transport error, an SDK error
result — goes back round the ladder, because those *can* differ on a retry.

**Nothing is pre-allowed.** The permission callback allows every tool that is
not the question tool, so an `allowed_tools` entry would buy one skipped round
trip and, for `AskUserQuestion`, would skip the callback the answers ride on.
With nothing to pre-allow there is no shadowing for the SDK to warn about, and
so nothing here touches the process-wide warning filters.
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
        """`query_fn` defaults to the SDK's `query`.

        It exists so the retry ladder can be driven against a stub, and it is
        the only injection seam in this module.

        `settings_path` is made absolute here, so the guarantee is enforced
        rather than assumed: a run's `cwd` is the target repository, and a
        relative path handed to the SDK would be read from inside the very
        repository this module's configuration exists to seal out. Absolute, not
        resolved — following symlinks would hand the SDK a path the caller never
        named, and it is the anchoring that matters.

        Raises `ValueError` for fewer than one attempt: the ladder would never
        run, and a run that was never attempted has no failure to report.
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
        """One call.

        Raises `AgentBudgetError` when the run hit a ceiling, `AgentOutputError`
        when `output_schema` was set and the text did not parse, and `AgentError`
        — from `fold` — for every other way the run failed to complete. Only
        the first two are exempt from the retry ladder.
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

    This is not a permission policy. What a run may do is decided by the options
    it was built with; this callback exists because the SDK routes
    `AskUserQuestion` through the same channel, and it is the only way to get
    the question out and an answer back in.
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
            # Four questions can arrive in one call and nothing stops two of
            # them sharing their text. The map the tool reads is keyed by that
            # text, so a second answer has nowhere to go but on top of the
            # first. Asking once and letting both take the answer loses nothing
            # a second round trip would have kept, and does not put the same
            # string in front of a person twice.
            if question.title in answers:
                continue
            answers[question.title] = await on_question(question)

        return PermissionResultAllow(updated_input={**tool_input, "answers": answers})

    return can_use_tool


def _question(entry: Mapping[str, Any]) -> AgentQuestion:
    """One entry of the tool's input as this module's own question type.

    The answers map is keyed by the question text, so the title has to be that
    same string — the header is a chip in the UI, and two questions can share it.
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
