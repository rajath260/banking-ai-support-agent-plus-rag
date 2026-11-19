"""Offline tests for the MCP-connected LangGraph agent and conversation memory."""

from types import SimpleNamespace

import pytest

from app.mcp_agent_graph import build_mcp_agent_graph, mcp_tools_to_llm_tools


DISCOVERED_TOOLS = [
    {
        "name": "account_balance",
        "description": "Get the fictional demo account balance.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "recent_transactions",
        "description": "Get recent transactions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 5}
            },
            "required": [],
        },
    },
]


def sdk_message(content="", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def sdk_tool_call(name, arguments, call_id="call_1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def sdk_response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeCompletions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


def fake_model_client(responses):
    completions = FakeCompletions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


class FakeMCPSession:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "recent_transactions":
            output = "Transaction A: Grocery Store (500.00 INR)\\nTransaction B: Fuel (300.00 INR)"
        else:
            output = "Account balance: INR 45,000.00"
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=output)],
            isError=False,
        )


def graph_config(thread_id="test-conversation"):
    return {"configurable": {"thread_id": thread_id}}


def test_mcp_schemas_are_converted_to_llm_function_tools():
    tools = mcp_tools_to_llm_tools(DISCOVERED_TOOLS)

    assert [item["function"]["name"] for item in tools] == [
        "account_balance",
        "recent_transactions",
    ]
    assert "limit" in tools[1]["function"]["parameters"]["properties"]


@pytest.mark.anyio
async def test_agent_uses_discovered_mcp_tool_and_returns_model_answer():
    client, completions = fake_model_client(
        [
            sdk_response(
                sdk_message(
                    tool_calls=[sdk_tool_call("account_balance", "{}")]
                )
            ),
            sdk_response(sdk_message("Your demo account balance is INR 45,000.00.")),
        ]
    )
    session = FakeMCPSession()
    graph = build_mcp_agent_graph(session, DISCOVERED_TOOLS, client=client)

    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "What is my account balance?"}]},
        config=graph_config(),
    )

    assert "INR 45,000.00" in result["final_response"]
    assert session.calls == [("account_balance", {})]
    assert len(completions.calls) == 2
    assert completions.calls[0]["tools"][0]["function"]["name"] == "account_balance"
    assert any("invoked 1 requested tool" in item for item in result["trace"])


@pytest.mark.anyio
async def test_agent_answers_general_question_without_mcp_call():
    client, completions = fake_model_client(
        [sdk_response(sdk_message("A savings account is used to save money."))]
    )
    session = FakeMCPSession()
    graph = build_mcp_agent_graph(session, DISCOVERED_TOOLS, client=client)

    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "What is a savings account?"}]},
        config=graph_config(),
    )

    assert result["final_response"] == "A savings account is used to save money."
    assert session.calls == []
    assert len(completions.calls) == 1


@pytest.mark.anyio
async def test_agent_rejects_tool_name_not_discovered_from_mcp_server():
    client, _ = fake_model_client(
        [
            sdk_response(
                sdk_message(
                    tool_calls=[sdk_tool_call("delete_account", "{}", "call_unknown")]
                )
            ),
            sdk_response(sdk_message("I could not perform that operation.")),
        ]
    )
    session = FakeMCPSession()
    graph = build_mcp_agent_graph(session, DISCOVERED_TOOLS, client=client)

    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "Delete my account"}]},
        config=graph_config(),
    )

    assert session.calls == []
    assert "not discovered" in result["tool_results"][0]["content"]


@pytest.mark.anyio
async def test_agent_remembers_previous_turn_and_understands_follow_up():
    client, completions = fake_model_client(
        [
            sdk_response(
                sdk_message(
                    tool_calls=[sdk_tool_call("account_balance", "{}", "call_balance")]
                )
            ),
            sdk_response(sdk_message("Your demo account balance is INR 45,000.00.")),
            sdk_response(
                sdk_message(
                    tool_calls=[
                        sdk_tool_call(
                            "recent_transactions",
                            '{"limit": 2}',
                            "call_transactions",
                        )
                    ]
                )
            ),
            sdk_response(
                sdk_message(
                    "Here are your two latest transactions: Grocery Store and Fuel."
                )
            ),
        ]
    )
    session = FakeMCPSession()
    graph = build_mcp_agent_graph(session, DISCOVERED_TOOLS, client=client)

    first = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "What is my account balance?"}]},
        config=graph_config("same-user-thread"),
    )
    second = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "Show my last two transactions."}]},
        config=graph_config("same-user-thread"),
    )

    assert "INR 45,000.00" in first["final_response"]
    assert "Grocery Store and Fuel" in second["final_response"]
    assert session.calls == [
        ("account_balance", {}),
        ("recent_transactions", {"limit": 2}),
    ]

    # The second LLM request contains the earlier user turn and assistant answer.
    second_turn_messages = completions.calls[2]["messages"]
    assert any(
        item.get("role") == "user"
        and item.get("content") == "What is my account balance?"
        for item in second_turn_messages
    )
    assert any(
        item.get("role") == "assistant"
        and item.get("content") == "Your demo account balance is INR 45,000.00."
        for item in second_turn_messages
    )
    assert second_turn_messages[-1]["content"] == "Show my last two transactions."
