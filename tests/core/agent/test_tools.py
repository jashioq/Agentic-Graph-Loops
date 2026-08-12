"""Our tools, registered as an in-process MCP server.

Tools here are throwaway — `add`, `echo`, `secret`. The module under test is not
allowed to know what a tool is for, and a test that named a real one would be
the first place that knowledge leaked in.

Handlers are driven through the server's own request handlers rather than by
calling the wrapper directly, so what is asserted is what the model would get.
"""

from typing import Any

import mcp.types as mcp
import pytest

from agl.core.agent import Tool
from agl.core.agent.impl.tools import (
    SERVER_NAME,
    build_keepalive_server,
    build_tool_server,
)

NUMBERS = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
}


def add_tool() -> Tool:
    async def handler(arguments: dict[str, Any]) -> str:
        return str(arguments["a"] + arguments["b"])

    return Tool(name="add", description="Add two numbers", schema=NUMBERS, handler=handler)


def echo_tool() -> Tool:
    async def handler(arguments: dict[str, Any]) -> str:
        return str(arguments["text"])

    return Tool(
        name="echo",
        description="Echo the text back",
        schema={"type": "object", "properties": {"text": {"type": "string"}}},
        handler=handler,
    )


# -- driving the server ---------------------------------------------------


async def listed(servers: dict[str, Any]) -> list[mcp.Tool]:
    """Every tool the server would list, as the model would see it."""
    instance = servers[SERVER_NAME]["instance"]
    request = mcp.ListToolsRequest(method="tools/list")
    result = await instance.request_handlers[mcp.ListToolsRequest](request)
    return list(result.root.tools)


async def call(servers: dict[str, Any], name: str, **arguments: Any) -> mcp.CallToolResult:
    """Invoke a tool through the server and hand back the raw result."""
    instance = servers[SERVER_NAME]["instance"]
    request = mcp.CallToolRequest(
        method="tools/call",
        params=mcp.CallToolRequestParams(name=name, arguments=arguments),
    )
    result = await instance.request_handlers[mcp.CallToolRequest](request)
    return result.root


def text_of(result: mcp.CallToolResult) -> str:
    return "\n".join(block.text for block in result.content if block.type == "text")


# -- registration ----------------------------------------------------------


def test_a_tool_is_registered_under_a_namespaced_name() -> None:
    _, names = build_tool_server([add_tool()])
    assert names == ("mcp__agl__add",)


async def test_the_server_lists_the_tool_under_its_bare_name() -> None:
    # The namespace is the SDK's doing: the server knows `add`, the model is
    # shown `mcp__agl__add`.
    servers, _ = build_tool_server([add_tool()])
    assert [tool.name for tool in await listed(servers)] == ["add"]


async def test_the_schema_and_description_reach_the_server_definition() -> None:
    servers, _ = build_tool_server([add_tool()])
    (registered,) = await listed(servers)

    assert registered.description == "Add two numbers"
    assert registered.inputSchema == NUMBERS


async def test_two_tools_both_register() -> None:
    servers, names = build_tool_server([add_tool(), echo_tool()])

    assert names == ("mcp__agl__add", "mcp__agl__echo")
    assert [tool.name for tool in await listed(servers)] == ["add", "echo"]


def test_duplicate_names_raise_at_build_time() -> None:
    with pytest.raises(ValueError, match="add"):
        build_tool_server([add_tool(), add_tool()])


def test_no_tools_means_no_server() -> None:
    servers, names = build_tool_server([])

    assert servers == {}
    assert names == ()


async def test_the_keepalive_server_is_a_real_server_with_nothing_on_it() -> None:
    # It exists only so the SDK keeps its input stream open; a caller that needs
    # questions but registered no tools would otherwise have stdin closed under
    # it before the permission callback could fire.
    servers = build_keepalive_server()

    assert set(servers) == {SERVER_NAME}
    assert servers[SERVER_NAME]["type"] == "sdk"
    assert await listed(servers) == []


# -- invocation ------------------------------------------------------------


async def test_invoking_a_handler_returns_its_string() -> None:
    servers, _ = build_tool_server([add_tool()])

    result = await call(servers, "add", a=2, b=3)

    assert text_of(result) == "5"
    assert result.isError is False


async def test_a_handler_that_closes_over_a_value_returns_that_value() -> None:
    # This is the mechanism a caller uses for scoping: the tool takes no
    # parameters, so there is nothing for the model to widen.
    async def handler(arguments: dict[str, Any]) -> str:
        return "the-one-it-was-given"

    scoped = Tool(
        name="secret",
        description="The value this tool was built around",
        schema={"type": "object", "properties": {}},
        handler=handler,
    )
    servers, _ = build_tool_server([scoped])

    assert text_of(await call(servers, "secret")) == "the-one-it-was-given"


async def test_a_raising_handler_becomes_an_error_the_model_can_read() -> None:
    async def handler(arguments: dict[str, Any]) -> str:
        raise RuntimeError("the well is dry")

    servers, _ = build_tool_server(
        [Tool(name="fail", description="Always fails", schema={}, handler=handler)]
    )

    result = await call(servers, "fail")

    assert result.isError is True
    assert "the well is dry" in text_of(result)
    assert "RuntimeError" in text_of(result)


async def test_a_raising_handler_does_not_take_the_run_down_with_it() -> None:
    async def handler(arguments: dict[str, Any]) -> str:
        raise RuntimeError("boom")

    servers, _ = build_tool_server(
        [
            Tool(name="fail", description="Always fails", schema={}, handler=handler),
            add_tool(),
        ]
    )

    await call(servers, "fail")

    # The server is still there and the next tool still works.
    assert text_of(await call(servers, "add", a=1, b=1)) == "2"


async def test_a_handler_returning_a_non_string_says_so_rather_than_coercing() -> None:
    # Not stringified: the model cannot fix a handler that violates its own
    # type, and coercing would hide the bug behind plausible-looking output. The
    # wrapper raises `TypeError`; MCP's own boundary is what turns it into a
    # result, so the message a test can see is the one the model would get.
    async def handler(arguments: dict[str, Any]) -> str:
        return {"not": "a string"}  # type: ignore[return-value]

    servers, _ = build_tool_server(
        [Tool(name="wrong", description="Returns a dict", schema={}, handler=handler)]
    )

    result = await call(servers, "wrong")

    assert result.isError is True
    assert text_of(result) == "tool 'wrong' returned dict, expected str"
