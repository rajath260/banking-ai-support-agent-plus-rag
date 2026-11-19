"""Offline unit tests for MCP client helper functions."""

from types import SimpleNamespace

import pytest

from mcp_client.client import discover_tools, invoke_tool


class FakeSession:
    def __init__(self):
        self.called_name = None
        self.called_arguments = None

    async def list_tools(self):
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="account_balance",
                    description="Get a demo balance.",
                    inputSchema={"type": "object"},
                )
            ]
        )

    async def call_tool(self, name, arguments):
        self.called_name = name
        self.called_arguments = arguments
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Account balance: INR 45,000.00")
            ],
            isError=False,
        )


@pytest.mark.anyio
async def test_discover_tools_reads_server_tool_metadata():
    session = FakeSession()

    tools = await discover_tools(session)

    assert tools == [
        {
            "name": "account_balance",
            "description": "Get a demo balance.",
            "input_schema": {"type": "object"},
        }
    ]


@pytest.mark.anyio
async def test_invoke_tool_sends_arguments_and_returns_text():
    session = FakeSession()

    output = await invoke_tool(
        session,
        "account_balance",
        {},
    )

    assert session.called_name == "account_balance"
    assert session.called_arguments == {}
    assert "INR 45,000.00" in output


@pytest.mark.anyio
async def test_invoke_tool_raises_when_server_reports_error():
    class ErrorSession(FakeSession):
        async def call_tool(self, name, arguments):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="Unknown account")],
                isError=True,
            )

    with pytest.raises(RuntimeError, match="Unknown account"):
        await invoke_tool(ErrorSession(), "account_balance", {})
