"""Tests for the initial LangGraph milestone."""

from app.graph import build_graph


def run_query(query: str) -> dict:
    return build_graph().invoke(
        {"user_query": query, "account_id": "DEMO-001", "trace": []}
    )


def test_balance_tool_path():
    result = run_query("Check my account balance")
    assert "INR 45,000.00" in result["final_response"]
    assert result["intent"] == "balance"


def test_transactions_tool_path():
    result = run_query("Show recent transactions")
    assert "Grocery Store" in result["final_response"]


def test_card_status_tool_path():
    result = run_query("Is my card active?")
    assert "ACTIVE" in result["final_response"]


def test_general_path_does_not_call_tool():
    result = run_query("How do I reset my password?")
    assert "fictional data only" in result["final_response"]
    assert "tool_result" not in result
