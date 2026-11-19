"""Offline tests for the LangGraph tool-calling agent."""

from types import SimpleNamespace

from app.agent_graph import build_agent_graph


def sdk_message(content="", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def sdk_response(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeCompletions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


def fake_client(responses):
    completions = FakeCompletions(responses)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return client, completions


def test_graph_executes_balance_tool_without_hugging_face():
    tool_call = SimpleNamespace(
        id="call_balance_1",
        function=SimpleNamespace(
            name="get_account_balance",
            arguments="{}",
        ),
    )
    client, completions = fake_client(
        [
            sdk_response(sdk_message(tool_calls=[tool_call])),
            sdk_response(sdk_message("Your demo account balance is INR 45,000.00.")),
        ]
    )

    graph = build_agent_graph(client=client)
    result = graph.invoke(
        {
            "user_query": "What is my account balance?",
            "account_id": "DEMO-001",
            "trace": [],
        }
    )

    assert "INR 45,000.00" in result["final_response"]
    assert len(completions.calls) == 2
    assert completions.calls[0]["tool_choice"] == "auto"
    assert completions.calls[1]["tool_choice"] == "none"
    assert any("executed 1 approved tool" in item for item in result["trace"])


def test_graph_returns_general_answer_without_calling_tools():
    client, completions = fake_client(
        [sdk_response(sdk_message("A savings account earns interest on deposits."))]
    )

    graph = build_agent_graph(client=client)
    result = graph.invoke(
        {
            "user_query": "What is a savings account?",
            "account_id": "DEMO-001",
            "trace": [],
        }
    )

    assert result["final_response"] == "A savings account earns interest on deposits."
    assert len(completions.calls) == 1
    assert result.get("tool_results", []) == []
    assert len(result["trace"]) == 2
