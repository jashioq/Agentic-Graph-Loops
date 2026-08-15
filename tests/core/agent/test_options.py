"""Building the SDK options object. Pure — no fixtures, no async, no network.

Hermetic configuration is the whole point of this module: every run has to be
reproducible, and the target project must contribute source code and nothing
else. Each assertion here is one way that could quietly stop being true.
"""

from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from agl.core.agent import AgentSpec, Model
from agl.core.agent.impl.options import build_options

MINIMAL = AgentSpec(prompt="do the thing", cwd=Path("/repo"), role="implement")

FULL = AgentSpec(
    prompt="do the thing",
    cwd=Path("/repo"),
    role="implement",
    system_prompt_append="House rules go here.",
    add_dirs=(Path("/other"), Path("/third")),
    disallowed_tools=("WebFetch",),
    permission_mode="acceptEdits",
    model=Model.SONNET,
    output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
)


def build(spec: AgentSpec = MINIMAL, **overrides: Any) -> ClaudeAgentOptions:
    """`build_options` with the two arguments a test is not exercising defaulted."""
    arguments: dict[str, Any] = {
        "settings_path": None,
        "mcp_servers": {},
    }
    arguments.update(overrides)
    return build_options(spec, **arguments)


# -- everything a populated spec asks for lands on the options ------------


def test_every_field_of_a_full_spec_reaches_the_options() -> None:
    options = build(FULL, settings_path=Path("/etc/agl/settings.json"))

    assert options.cwd == Path("/repo")
    assert options.add_dirs == [Path("/other"), Path("/third")]
    assert options.permission_mode == "acceptEdits"
    assert options.model == "sonnet"
    assert options.disallowed_tools == ["WebFetch"]
    assert options.settings == "/etc/agl/settings.json"
    assert options.system_prompt == {
        "type": "preset",
        "preset": "claude_code",
        "append": "House rules go here.",
    }
    assert options.output_format == {"type": "json_schema", "schema": FULL.output_schema}


def test_the_model_reaches_the_options_as_a_plain_string() -> None:
    # The SDK is handed the alias itself, not an enum member that happens to be
    # a string: nothing of ours should travel out through the options object.
    model = build(FULL).model
    assert model == "sonnet"
    assert type(model) is str


def test_a_minimal_spec_populates_nothing_spurious() -> None:
    options = build(MINIMAL)

    assert options.model is None
    assert options.output_format is None
    assert options.settings is None
    assert options.disallowed_tools == []
    assert options.add_dirs == []
    assert options.mcp_servers == {}


# -- hermetic configuration ------------------------------------------------


def test_setting_sources_is_an_empty_list_never_none() -> None:
    # `None` means "load user, project and local settings" — the opposite of
    # what this module exists to guarantee. Only `[]` disables the lot.
    for options in (build(MINIMAL), build(FULL)):
        assert options.setting_sources == []
        assert options.setting_sources is not None


def test_strict_mcp_config_is_true() -> None:
    # Without it the target repo's `.mcp.json` is loaded and the run reaches
    # servers we never configured.
    assert build(MINIMAL).strict_mcp_config is True


def test_the_preset_is_used_bare_when_there_is_nothing_to_append() -> None:
    assert build(MINIMAL).system_prompt == {"type": "preset", "preset": "claude_code"}


def test_the_append_reaches_the_options() -> None:
    spec = AgentSpec(
        prompt="p", cwd=Path("/repo"), role="r", system_prompt_append="Extra."
    )
    system_prompt = build(spec).system_prompt
    assert isinstance(system_prompt, dict)
    assert system_prompt["append"] == "Extra."


def test_no_settings_path_leaves_the_settings_option_unset() -> None:
    assert build(MINIMAL, settings_path=None).settings is None


def test_the_settings_path_is_passed_as_an_absolute_string() -> None:
    options = build(MINIMAL, settings_path=Path("/etc/agl/settings.json"))
    assert options.settings == "/etc/agl/settings.json"
    assert isinstance(options.settings, str)


# -- tools -----------------------------------------------------------------


def test_nothing_is_ever_pre_allowed() -> None:
    # `can_use_tool` allows every tool that is not the question tool, so an
    # allow rule only skips a round trip — and for `AskUserQuestion` it skips
    # the callback that carries the answers.
    assert build(MINIMAL).allowed_tools == []
    assert build(FULL).allowed_tools == []


def test_leaving_allowed_tools_empty_does_not_read_as_allow_nothing() -> None:
    # `tools` is what restricts the built-in set; leaving it unset keeps the
    # Claude Code defaults, whereas `[]` would strip every built-in tool.
    assert build(MINIMAL).tools is None


def test_disallowed_tools_are_kept() -> None:
    # Deny rules resolve ahead of the callback and hold even under
    # `bypassPermissions`, and their pattern language is the CLI's.
    spec = AgentSpec(
        prompt="p",
        cwd=Path("/repo"),
        role="r",
        disallowed_tools=("WebFetch", "Bash(git commit:*)"),
    )
    assert build(spec).disallowed_tools == ["WebFetch", "Bash(git commit:*)"]


def test_the_server_mapping_is_passed_through_untouched() -> None:
    servers = {"agl": {"type": "sdk", "name": "agl", "instance": object()}}
    assert build(MINIMAL, mcp_servers=servers).mcp_servers == servers


# -- the permission mode is a closed set -----------------------------------


@pytest.mark.parametrize(
    "mode", ["default", "acceptEdits", "plan", "bypassPermissions", "dontAsk", "auto"]
)
def test_every_mode_the_sdk_accepts_passes_through(mode: str) -> None:
    spec = AgentSpec(prompt="p", cwd=Path("/repo"), role="r", permission_mode=mode)
    assert build(spec).permission_mode == mode


def test_an_unknown_permission_mode_is_refused_here() -> None:
    # `permission_mode` is a plain `str` on the spec so the API stays free of
    # SDK types; the check the type would have given us happens here instead.
    spec = AgentSpec(prompt="p", cwd=Path("/repo"), role="r", permission_mode="yolo")
    with pytest.raises(ValueError, match="yolo"):
        build(spec)


# -- the question tool cannot be pre-allowed by accident -------------------


def test_there_is_no_way_to_allow_the_question_tool() -> None:
    # Allowing `AskUserQuestion` outright would approve the call with no answers
    # in it — an allow rule resolves before `can_use_tool`, and the callback is
    # what injects them. There is no allow list to put it on, so the footgun is
    # unreachable rather than guarded.
    assert "allowed_tools" not in {field.name for field in fields(AgentSpec)}
    assert build(FULL).allowed_tools == []
