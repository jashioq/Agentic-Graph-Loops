"""Prompt files, and the assembly of one model call.

Layer: runtime. Imports `agl.core.agent` and nothing else. It has never heard of
a role: every call looks the same from here — a rendered prompt, a directory,
some tools, and the model it runs on, which each call names because only the
workflow knows whether it is judgement work or execution work. Substitution is
strict, so an unfilled `$placeholder` fails here rather than reaching a model.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from string import Template

from agl.core.agent import AgentQuestion, AgentResult, AgentRunner, AgentSpec, Model, Tool

__all__ = ["Activity", "Ask", "PromptError", "Prompts", "call"]

type Activity = Callable[[str], None]
"""Where a long-running call reports the one line it is doing right now."""

type Ask = Callable[[AgentQuestion], Awaitable[str]]
"""How a call puts its own question to whoever is watching the run."""


class PromptError(Exception):
    """Raised when a prompt could not be assembled — usually a missing substitution."""


@dataclass(frozen=True)
class Prompts:
    """One directory of `$`-templated markdown files, rendered by name."""

    directory: Path

    def render(self, name: str, **substitutions: str) -> str:
        """Loads `<name>.md` and substitutes it.

        return: str - raises `PromptError` on an unfilled placeholder,
            `FileNotFoundError` on a missing file, rather than rendering empty
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
    on_activity: Activity | None = None,
    ask: Ask | None = None,
) -> AgentResult:
    """Builds one spec and runs it — the only place an `AgentSpec` is assembled.

    Keyword-only past the runner, since `tools` and `disallowed` are two tuples
    that would read identically the wrong way round. `model` has no default.
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
