"""Agent API: one model call, described as data.

Layer: core. This is the single path through which every model call in the
system goes. It owns session configuration, budget limits, the retry ladder,
custom tool registration, and the translation from a message stream into a
result. It has never heard of a work item, a worktree, or a workflow: it takes a
prompt, a directory, and a set of opaque tools, and hands back text.

Tools are how a caller scopes what a run can reach, but the scoping happens in
the *closure*, not here. A caller that builds a tool over one document passes a
callable that only ever touches that document, so there is no parameter for the
model to widen. This module sees a name, a schema, and something to await.

Two callbacks, and only two. `on_activity` is fire-and-forget UI: a call runs
for minutes, and without it a hung run is indistinguishable from a working one.
`on_question` is interception — the run stops until it returns. Everything else
is reported by returning `AgentResult`.

`AgentQuestion` is this module's own type rather than `terminal.Question`. Core
modules do not import each other, and translating one into the other belongs to
the layer that knows why the question was asked.
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "NO_PARAMS",
    "AgentBudgetError",
    "AgentError",
    "AgentOption",
    "AgentQuestion",
    "AgentResult",
    "AgentRunner",
    "AgentSpec",
    "Tool",
]

NO_PARAMS: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}
"""The schema for a tool that takes no arguments — the canonical scoped shape.

Everything such a tool may reach was closed over when it was built, so there is
nothing left for the model to pass and nothing for it to widen. Read-only by
convention: hand it to a `Tool` rather than mutating it."""


@dataclass(frozen=True)
class Tool:
    """A callable the model may invoke during a run.

    Opaque to this module: a name, a description, a JSON schema for its
    parameters, and a handler. Whether it reads something, saves something, or
    does something else entirely is the caller's business.
    """

    name: str
    description: str
    schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[str]]


@dataclass(frozen=True)
class AgentSpec:
    """One model call: what to ask, where, with what, and under what limits."""

    prompt: str
    cwd: Path
    # What the caller is calling this run. Opaque here, and used only to name
    # the run in an error message; a caller with several kinds of call tells
    # them apart with it.
    role: str
    tools: tuple[Tool, ...] = ()
    system_prompt_append: str | None = None
    add_dirs: tuple[Path, ...] = ()
    # Deny rules only. There is no allow list: the permission callback allows
    # every tool that is not the question tool, so an allow rule would buy one
    # skipped round trip — and for the question tool it would skip the callback
    # that carries the answers. Denials are kept because they resolve ahead of
    # the callback, hold even under `bypassPermissions`, and speak the CLI's own
    # pattern language (`Bash(git commit:*)`).
    disallowed_tools: tuple[str, ...] = ()
    permission_mode: str = "default"  # "default" | "plan" | …
    model: str | None = None
    max_turns: int | None = None
    max_budget_usd: float | None = None
    output_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class AgentResult:
    """What one call produced, and what it cost to produce it.

    There is no error flag. A run that could not be completed raises, so every
    result that reaches a caller is one that finished.
    """

    text: str  # the final assistant text
    structured: Any | None  # parsed JSON when output_schema was set
    session_id: str | None
    cost_usd: float
    num_turns: int
    duration_ms: int
    terminal_reason: str | None


@dataclass(frozen=True)
class AgentOption:
    """One offered answer to an `AgentQuestion`."""

    label: str
    description: str


@dataclass(frozen=True)
class AgentQuestion:
    """A question the model asked mid-run, which it cannot continue without."""

    title: str
    options: tuple[AgentOption, ...]


class AgentError(Exception):
    """Raised when a call could not be completed."""


class AgentBudgetError(AgentError):
    """Raised when a call stopped because it ran out of budget or turns.

    Distinct because it is the one failure not worth retrying: the same call
    fails the same way and spends the budget again. Exhaustion says the task is
    too large, not that the call went wrong.
    """


class AgentRunner(ABC):
    """Runs one `AgentSpec` to completion."""

    @abstractmethod
    async def run(
        self,
        spec: AgentSpec,
        on_activity: Callable[[str], None] | None = None,
        on_question: Callable[[AgentQuestion], Awaitable[str]] | None = None,
    ) -> AgentResult:
        """Run `spec` and return what it produced.

        `on_activity` receives a short human string per tool use, for a status
        line. It is fire-and-forget: raising from it must not fail the run.

        `on_question` is awaited when the model asks the user something, and
        returns the chosen label or the user's free text. When it is `None` the
        question is refused and the model is told to use its own judgment.

        Raises `AgentBudgetError` when the run exhausted its budget or turns,
        and `AgentError` when it could not be completed for any other reason.
        """
