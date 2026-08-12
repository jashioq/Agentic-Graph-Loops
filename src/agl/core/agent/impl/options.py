"""One `AgentSpec` turned into one `ClaudeAgentOptions`. Pure — no I/O.

Layer: core. This is where hermetic configuration is decided, which makes it the
most important thing in the module to get right: a run has to be reproducible,
and the target project must contribute source code and nothing else.

Three options do that work, and all three are easy to lose by accident:

- `setting_sources=[]` — do not read `~/.claude/settings.json`, the project's
  `.claude/`, or anyone's local overrides. `None` means "read them all", so the
  empty list is load-bearing and is asserted on in the tests.
- `strict_mcp_config=True` — ignore the project's `.mcp.json`. Only servers
  passed in here exist.
- `settings` — our own file, when the caller has one, as an absolute path.

Not loading `project` settings also means the target repo's `CLAUDE.md` is not
loaded. That is intentional: a caller that wants it reads it and passes it
through `system_prompt_append`, which keeps the loading explicit and visible in
the prompt rather than implicit in the working directory.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast, get_args

from claude_agent_sdk import ClaudeAgentOptions, McpServerConfig, PermissionMode
from claude_agent_sdk.types import SystemPromptPreset

from agl.core.agent.api import AgentSpec

__all__ = ["QUESTION_TOOL", "build_options"]

PERMISSION_MODES: frozenset[str] = frozenset(get_args(PermissionMode))

QUESTION_TOOL = "AskUserQuestion"


def build_options(
    spec: AgentSpec,
    settings_path: Path | None,
    mcp_servers: dict[str, Any],
    tool_names: Sequence[str],
) -> ClaudeAgentOptions:
    """The options for one call, with every configuration source pinned shut.

    `mcp_servers` and `tool_names` come from `build_tool_server`; they are
    separate arguments because a caller may need a server the spec did not ask
    for — a keep-alive server, for instance — and this function should not have
    to know why.

    Raises `ValueError` when the spec's permission mode is not one the SDK
    knows. The spec types it as `str` to keep the API free of SDK types, so
    this is where the check the `Literal` would have given us happens. It also
    raises when the spec allows the question tool outright, for the reason
    below.
    """
    if spec.permission_mode not in PERMISSION_MODES:
        raise ValueError(
            f"unknown permission mode {spec.permission_mode!r}: "
            f"expected one of {', '.join(sorted(PERMISSION_MODES))}"
        )
    _refuse_allowing_the_question_tool(spec.allowed_tools)

    system_prompt: SystemPromptPreset = {"type": "preset", "preset": "claude_code"}
    if spec.system_prompt_append is not None:
        system_prompt["append"] = spec.system_prompt_append

    output_format = None
    if spec.output_schema is not None:
        output_format = {"type": "json_schema", "schema": spec.output_schema}

    return ClaudeAgentOptions(
        cwd=spec.cwd,
        add_dirs=list(spec.add_dirs),
        system_prompt=system_prompt,
        # A registered tool the model is not allowed to call is a silent dead
        # end: it appears in the listing and every call to it stops for a
        # permission prompt nobody is there to answer.
        allowed_tools=[*spec.allowed_tools, *tool_names],
        disallowed_tools=list(spec.disallowed_tools),
        permission_mode=cast(PermissionMode, spec.permission_mode),
        model=spec.model,
        max_turns=spec.max_turns,
        max_budget_usd=spec.max_budget_usd,
        output_format=output_format,
        mcp_servers=cast(dict[str, McpServerConfig], mcp_servers),
        setting_sources=[],
        strict_mcp_config=True,
        settings=str(settings_path) if settings_path is not None else None,
    )


def _refuse_allowing_the_question_tool(allowed_tools: Sequence[str]) -> None:
    """Raise if any entry allows `AskUserQuestion` as a whole tool.

    The SDK resolves a tool call in three stages and stops at the first answer:
    a deny rule, then an allow rule, then `can_use_tool`. For every other tool
    an allow rule merely skips a check that would have said yes. For this one
    the callback *is* the mechanism — the user's answers are injected by it, as
    `updated_input`, and take no other path — so allowing the tool outright
    approves a call that nobody was asked and that carries no answers. The model
    reads an empty result and guesses, and nothing on the dashboard or in the
    result says so.

    It is refused rather than quietly dropped because it is a plausible mistake:
    "the agent needs to ask questions, so I should allow the tool" reads as the
    correct thing to do, and does the opposite.
    """
    for entry in allowed_tools:
        if _whole_tool_allowed(entry) == QUESTION_TOOL:
            raise ValueError(
                f"allowed_tools entry {entry!r} allows {QUESTION_TOOL} as a whole "
                f"tool, which stops the run from ever asking anything: an allow "
                f"rule is matched before the permission callback is consulted, so "
                f"the callback never fires — and the answers are injected by that "
                f"callback. The tool would be approved with no answers in it and "
                f"the model would proceed on a guess. Drop the entry: "
                f"{QUESTION_TOOL} is answered through the callback and does not "
                f"need allowing."
            )


def _whole_tool_allowed(entry: str) -> str | None:
    """The tool an `allowed_tools` entry allows outright, or `None`.

    Mirrors the CLI's own rule parser, which is what decides whether the entry
    shadows the callback: an entry allows a whole tool when it carries no
    `(...)` specifier (`"Read"`), or when the specifier is empty or a lone
    wildcard (`"Read()"`, `"Read(*)"`). A real specifier (`"Bash(ls:*)"`) allows
    only matching invocations and leaves the rest falling through, so it is not
    the same thing. Malformed entries match nothing and are ignored.
    """
    if not entry.strip():
        return None
    open_index = entry.find("(")
    if open_index == -1:
        return entry
    if open_index == 0 or not entry.endswith(")"):
        return None
    return entry[:open_index] if entry[open_index + 1 : -1] in ("", "*") else None
