"""Prompt files, and the assembly of one model call.

Layer: runtime. Imports `agl.core.agent` and nothing else. It has never heard
of a role: a role is a workflow's word for one of its calls, and every call
looks the same from here — a rendered prompt, a directory, some tools, and the
ceilings the run is under.

`Limits` is separate from the call for the same reason a run has one budget and
many calls: a workflow sets it once, threads it through every call it makes, and
no call site gets to decide it is the exception. `Prompts` is a directory of
templates rather than a loader per file, so a workflow points it at its own
`prompts/` once and names files after that.

Substitution is strict. `Template.substitute` raises on a placeholder nothing
filled, which is what `PromptError` reports — a `$deliverables` that reached a
model as the literal text `$deliverables` is a prompt bug that reads as a model
failure three steps later, and it is much cheaper to catch here.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from string import Template

from agl.core.agent import AgentQuestion, AgentResult, AgentRunner, AgentSpec, Model, Tool

__all__ = ["Limits", "PromptError", "Prompts", "call"]


@dataclass(frozen=True)
class Limits:
    """Ceilings threaded onto every `AgentSpec` a run builds."""

    model: Model | None = None
    max_turns: int | None = None
    max_budget_usd: float | None = None


class PromptError(Exception):
    """Raised when a prompt could not be assembled — usually a missing substitution."""


@dataclass(frozen=True)
class Prompts:
    """One directory of `$`-templated markdown files, rendered by name."""

    directory: Path

    def render(self, name: str, **substitutions: str) -> str:
        """Load `<name>.md` and substitute it, raising on anything left unfilled.

        A file that is not there raises `FileNotFoundError` rather than
        rendering empty: a prompt is the whole instruction, and an empty one
        produces a confidently wrong run instead of a failure.
        """
        path = self.directory / f"{name}.md"
        template = Template(path.read_text())
        try:
            return template.substitute(**substitutions)
        except KeyError as error:
            raise PromptError(f"{path}: missing a substitution for {error}") from error


async def call(
    runner: AgentRunner,
    *,
    role: str,
    prompt: str,
    cwd: Path,
    tools: tuple[Tool, ...] = (),
    disallowed: tuple[str, ...] = (),
    permission_mode: str = "default",
    limits: Limits,
    on_activity: Callable[[str], None] | None = None,
    ask: Callable[[AgentQuestion], Awaitable[str]] | None = None,
) -> AgentResult:
    """Build one spec and run it. The only place `Limits` meets `AgentSpec`.

    Keyword-only past the runner: a call has enough of them that positional
    order would be a coin toss at every site, and `tools` and `disallowed` in
    particular are two tuples that would read identically the wrong way round.
    """
    spec = AgentSpec(
        prompt=prompt,
        cwd=cwd,
        role=role,
        tools=tools,
        disallowed_tools=disallowed,
        permission_mode=permission_mode,
        model=limits.model,
        max_turns=limits.max_turns,
        max_budget_usd=limits.max_budget_usd,
    )
    return await runner.run(spec, on_activity, ask)
