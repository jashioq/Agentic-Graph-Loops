"""Building the SDK options object. Pure — no fixtures, no async, no network.

Hermetic configuration is the whole point of this module: every run has to be
reproducible, and the target project must contribute source code and nothing
else. Each assertion here is one way that could quietly stop being true.
"""

from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import ClaudeAgentOptions

from agl.core.agent import AgentSpec
from agl.core.agent.impl.options import build_options

MINIMAL = AgentSpec(prompt="do the thing", cwd=Path("/repo"), role="implement")

FULL = AgentSpec(
    prompt="do the thing",
    cwd=Path("/repo"),
    role="implement",
    system_prompt_append="House rules go here.",
    add_dirs=(Path("/other"), Path("/third")),
    allowed_tools=("Read", "Edit"),
    disallowed_tools=("WebFetch",),
    permission_mode="acceptEdits",
    model="claude-sonnet-4-5",
    max_turns=12,
    max_budget_usd=1.5,
    output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
)


def build(spec: AgentSpec = MINIMAL, **overrides: Any) -> ClaudeAgentOptions:
    """`build_options` with the three arguments a test is not exercising defaulted."""
    arguments: dict[str, Any] = {
        "settings_path": None,
        "mcp_servers": {},
        "tool_names": (),
    }
    arguments.update(overrides)
    return build_options(spec, **arguments)


# -- everything a populated spec asks for lands on the options ------------


def test_every_field_of_a_full_spec_reaches_the_options() -> None:
    options = build(FULL, settings_path=Path("/etc/agl/settings.json"))

    assert options.cwd == Path("/repo")
    assert options.add_dirs == [Path("/other"), Path("/third")]
    assert options.permission_mode == "acceptEdits"
    assert options.model == "claude-sonnet-4-5"
    assert options.max_turns == 12
    assert options.max_budget_usd == 1.5
    assert options.disallowed_tools == ["WebFetch"]
    assert options.settings == "/etc/agl/settings.json"
    assert options.system_prompt == {
        "type": "preset",
        "preset": "claude_code",
        "append": "House rules go here.",
    }
    assert options.output_format == {"type": "json_schema", "schema": FULL.output_schema}


def test_a_minimal_spec_populates_nothing_spurious() -> None:
    options = build(MINIMAL)

    assert options.model is None
    assert options.max_turns is None
    assert options.max_budget_usd is None
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


def test_custom_tool_names_are_merged_into_allowed_tools() -> None:
    spec = AgentSpec(prompt="p", cwd=Path("/repo"), role="r", allowed_tools=("Read",))
    options = build(spec, tool_names=("mcp__agl__add", "mcp__agl__echo"))

    assert options.allowed_tools == ["Read", "mcp__agl__add", "mcp__agl__echo"]


def test_a_registered_tool_the_model_may_not_call_is_a_silent_dead_end() -> None:
    # So every registered name is allowed even when the spec listed none.
    options = build(MINIMAL, tool_names=("mcp__agl__add",))
    assert options.allowed_tools == ["mcp__agl__add"]


def test_no_tools_does_not_read_as_allow_nothing() -> None:
    options = build(MINIMAL, tool_names=())

    assert options.allowed_tools == []
    # `tools` is what restricts the built-in set; leaving it unset keeps the
    # Claude Code defaults, whereas `[]` would strip every built-in tool.
    assert options.tools is None


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
