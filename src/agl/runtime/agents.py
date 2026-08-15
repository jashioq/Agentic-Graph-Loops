"""Prompt files, and the assembly of one model call.

Layer: runtime. Imports `agl.core.agent` and nothing else. It has never heard
of a role: a role is a workflow's word for one of its calls, and every call
looks the same from here — a rendered prompt, a directory, some tools, and the
model it runs on.

The model is an argument to each call rather than something a run fixes once,
because roles differ: judgement work and execution work do not want the same
model, and the workflow is the only layer that knows which of the two a call
is. `Prompts` is a directory of templates rather than a loader per file, so a
workflow points it at its own `prompts/` once and names files after that.

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

__all__ = ["PromptError", "Prompts", "call"]


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
    model: Model,
    on_activity: Callable[[str], None] | None = None,
    ask: Callable[[AgentQuestion], Awaitable[str]] | None = None,
) -> AgentResult:
    """Build one spec and run it — the only place an `AgentSpec` is assembled.

    Keyword-only past the runner: a call has enough of them that positional
    order would be a coin toss at every site, and `tools` and `disallowed` in
    particular are two tuples that would read identically the wrong way round.

    `model` has no default on purpose: a caller that has not decided which
    model its call wants should not get one picked for it silently.
    """
    spec = AgentSpec(
        prompt=prompt,
        cwd=cwd,
        role=role,
        tools=tools,
        disallowed_tools=disallowed,
        permission_mode=permission_mode,
        model=model,
    )
    return await runner.run(spec, on_activity, ask)
