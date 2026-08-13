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
- `settings` — our own file, when the caller has one. It arrives absolute
  because `ClaudeRunner` resolves it, which is enforcement rather than an
  assumption: `cwd` is the target repository, so a relative path would be read
  from inside the very repo these three options exist to seal out.

Nothing is put in `allowed_tools`. The permission callback allows every tool
that is not the question tool, so an allow rule would only skip a round trip —
and for `AskUserQuestion` it would skip the callback that carries the user's
answers, approving a question nobody was asked. `disallowed_tools` is passed
through: deny rules resolve ahead of the callback and hold even under
`bypassPermissions`.

Not loading `project` settings also means the target repo's `CLAUDE.md` is not
loaded. That is intentional: a caller that wants it reads it and passes it
through `system_prompt_append`, which keeps the loading explicit and visible in
the prompt rather than implicit in the working directory.
"""

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
) -> ClaudeAgentOptions:
    """The options for one call, with every configuration source pinned shut.

    `mcp_servers` is a separate argument rather than being derived from the
    spec, because a caller may need a server the spec did not ask for — a
    keep-alive server, for instance — and this function should not have to know
    why.

    Raises `ValueError` when the spec's permission mode is not one the SDK
    knows. The spec types it as `str` to keep the API free of SDK types, so
    this is where the check the `Literal` would have given us happens.
    """
    if spec.permission_mode not in PERMISSION_MODES:
        raise ValueError(
            f"unknown permission mode {spec.permission_mode!r}: "
            f"expected one of {', '.join(sorted(PERMISSION_MODES))}"
        )

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
