"""Our `Tool` objects, registered as one in-process MCP server.

Layer: core. The server runs inside this process, so a tool call is an `await`
and not a subprocess: handlers keep whatever they closed over, which is how a
caller scopes what a run can reach.

One server holds all of them, named `agl`, so the model sees `mcp__agl__<name>`.
`MCP_PREFIX` is that namespace, derived from the server name rather than spelled
out a second time, and the names the model will see are handed back alongside
the server so a caller never has to reconstruct them.

Two failure modes are treated as different things:

- A handler that *raises* is a fact about the world the model should react to —
  the file was not there, the command failed — so it becomes an error result
  carrying the message, and the run continues.
- A handler that returns a *non-string* violates `Tool`'s own type. The model
  cannot fix that, and stringifying it would hide the bug behind
  plausible-looking output, so it raises rather than being coerced. Registration
  cannot catch this: there is nothing to inspect until the handler has run. MCP's
  own call boundary catches the exception and reports it, so the run survives —
  what differs is that the message names a type violation rather than pretending
  the tool answered.
"""

from collections.abc import Sequence
from typing import Any

from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server

from agl.core.agent.api import Tool

__all__ = ["MCP_PREFIX", "SERVER_NAME", "build_keepalive_server", "build_tool_server"]

SERVER_NAME = "agl"

MCP_PREFIX = f"mcp__{SERVER_NAME}__"
"""What MCP puts in front of every tool name this server registers."""


def build_tool_server(
    tools: Sequence[Tool],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """The `mcp_servers` mapping for `tools`, and the names the model will see.

    No tools means no server: an empty mapping and no names. Raises `ValueError`
    on a duplicate name, which would otherwise leave one of the two silently
    unreachable.
    """
    seen: set[str] = set()
    for tool in tools:
        if tool.name in seen:
            raise ValueError(f"duplicate tool name: {tool.name!r}")
        seen.add(tool.name)

    if not tools:
        return {}, ()

    server = create_sdk_mcp_server(
        name=SERVER_NAME, tools=[_register(tool) for tool in tools]
    )
    names = tuple(f"{MCP_PREFIX}{tool.name}" for tool in tools)
    return {SERVER_NAME: server}, names


def build_keepalive_server() -> dict[str, Any]:
    """An `agl` server with no tools on it, to hold the SDK's input stream open.

    The SDK closes the CLI's stdin as soon as the prompt has been written unless
    an in-process server or a hook is registered, and the permission callback is
    answered over that same stream. A call that registers no tools still has to
    be able to ask a question, so it gets an empty server instead.
    """
    config = create_sdk_mcp_server(name=SERVER_NAME)

    # A server built with no tools registers no handlers at all, which leaves it
    # answering `tools/list` with "method not found". Saying "no tools" is a
    # different thing from not answering.
    @config["instance"].list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
    async def _no_tools() -> list[Any]:
        return []

    return {SERVER_NAME: config}


def _register(tool: Tool) -> SdkMcpTool[Any]:
    """One `Tool` as the SDK's tool definition, wrapped for error reporting."""

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await tool.handler(arguments)
        except Exception as error:  # noqa: BLE001 - the model decides what to do
            return _error(f"{type(error).__name__}: {error}")
        if not isinstance(result, str):
            raise TypeError(
                f"tool {tool.name!r} returned {type(result).__name__}, expected str"
            )
        return {"content": [{"type": "text", "text": result}]}

    return SdkMcpTool(
        name=tool.name,
        description=tool.description,
        input_schema=tool.schema,
        handler=handler,
    )


def _error(message: str) -> dict[str, Any]:
    """A tool result the model reads as a failure rather than as an answer."""
    return {"content": [{"type": "text", "text": message}], "is_error": True}
