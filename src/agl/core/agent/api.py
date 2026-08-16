"""Agent API: one model call, described as data.

Layer: core. The single path every model call goes through — session config, the
retry ladder, tool registration, and turning a message stream into a result. It
takes a prompt, a directory and opaque tools, and hands back text: a run reports
what it produced by calling a tool, never as JSON in its final message.

Scoping happens in a tool's *closure*, not here: this module sees a name, a
schema and something to await. Two callbacks, both optional: `on_activity` is
fire-and-forget, `on_question` stops the run until it returns. `AgentQuestion`
is this module's own type because core modules do not import each other.
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
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
    "Model",
    "Tool",
]

NO_PARAMS: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}
"""The schema for a tool that takes no arguments — the canonical scoped shape.

Everything it may reach was closed over, so there is nothing to widen. Read-only."""


@dataclass(frozen=True)
class Tool:
    """A callable the model may invoke during a run. Opaque to this module."""

    name: str
    description: str
    schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[str]]


class Model(StrEnum):
    """Which model a call runs on, as the CLI's own aliases rather than pinned ids."""

    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"
    FABLE = "fable"


@dataclass(frozen=True)
class AgentSpec:
    """One model call: what to ask, where, and with what."""

    prompt: str
    cwd: Path
    # The caller's own name for this call. Opaque here; used only in messages.
    role: str
    tools: tuple[Tool, ...] = ()
    system_prompt_append: str | None = None
    add_dirs: tuple[Path, ...] = ()
    # Deny rules only, no allow list: denials resolve ahead of the permission
    # callback, hold under `bypassPermissions`, and speak the CLI's own pattern
    # language (`Bash(git commit:*)`).
    disallowed_tools: tuple[str, ...] = ()
    permission_mode: str = "default"  # "default" | "plan" | …
    # `None` leaves the choice to the CLI's own default.
    model: Model | None = None


@dataclass(frozen=True)
class AgentResult:
    """What one call produced, and what it cost. No error flag: a failed call raises."""

    text: str  # the final assistant text
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
    """Raised when the CLI ended a call because it ran out of budget or turns.

    Distinct because it is not worth retrying: the task is too large.
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
        """Runs `spec` to completion.

        param: on_activity - one short line per tool use; fire-and-forget, and
            raising from it must not fail the run
        param: on_question - awaited when the model asks the user something,
            returning a label or free text; `None` refuses the question
        return: AgentResult - raises `AgentBudgetError` on exhaustion, `AgentError` otherwise
        """
