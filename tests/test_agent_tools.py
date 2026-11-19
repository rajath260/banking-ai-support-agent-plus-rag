"""Unit tests for the banking tool allow-list and argument validation."""

import pytest

from app.agent_tools import execute_banking_tool


def test_balance_tool_returns_demo_balance():
    result = execute_banking_tool("get_account_balance", {}, "DEMO-001")
    assert "INR 45,000.00" in result


def test_transactions_tool_respects_limit():
    result = execute_banking_tool(
        "get_recent_transactions", {"limit": 1}, "DEMO-001"
    )
    assert "Grocery Store" in result
    assert "Salary Credit" not in result


def test_card_status_tool_returns_demo_status():
    result = execute_banking_tool("get_card_status", {}, "DEMO-001")
    assert "ACTIVE" in result


def test_unknown_tool_is_rejected():
    with pytest.raises(ValueError, match="not allowed"):
        execute_banking_tool("delete_account", {}, "DEMO-001")


def test_model_cannot_override_account_id():
    with pytest.raises(ValueError, match="does not accept arguments"):
        execute_banking_tool(
            "get_account_balance", {"account_id": "SOMEONE-ELSE"}, "DEMO-001"
        )


def test_invalid_transaction_limit_is_rejected():
    with pytest.raises(ValueError, match="between 1 and 5"):
        execute_banking_tool(
            "get_recent_transactions", {"limit": 99}, "DEMO-001"
        )
