"""Example MCP client that discovers and invokes the demo banking tools.

Run with: python -m mcp_client.client
This client launches the MCP server as a child process over stdio.
"""

import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def server_parameters() -> StdioServerParameters:
    """Build parameters for launching the local demo MCP server."""
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPOSITORY_ROOT)
        if not existing_pythonpath
        else str(REPOSITORY_ROOT) + os.pathsep + existing_pythonpath
    )
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
        env=env,
        cwd=str(REPOSITORY_ROOT),
    )


async def discover_tools(session: ClientSession) -> list[dict]:
    """Ask the MCP server which tools it exposes."""
    response = await session.list_tools()
    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.inputSchema,
        }
        for tool in response.tools
    ]


async def invoke_tool(session: ClientSession, name: str, arguments: dict) -> str:
    """Invoke one discovered MCP tool and return its text output."""
    result = await session.call_tool(name, arguments=arguments)
    text_parts = [
        item.text
        for item in result.content
        if getattr(item, "type", None) == "text"
    ]
    if result.isError:
        raise RuntimeError("\n".join(text_parts) or f"MCP tool '{name}' failed.")
    return "\n".join(text_parts)


async def main() -> None:
    """Launch the server, discover tools, and invoke one demo tool."""
    params = server_parameters()
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools = await discover_tools(session)
            print("Tools discovered from MCP server:")
            for tool in tools:
                print(f"- {tool['name']}: {tool['description']}")

            print("\nCalling account_balance through MCP:")
            output = await invoke_tool(
                session,
                "account_balance",
                {},
            )
            print(output)


if __name__ == "__main__":
    asyncio.run(main())
