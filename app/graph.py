"""Inspectable LangGraph workflow using deterministic intent routing.

This first milestone does not use an LLM. Keyword routing is used to make
StateGraph mechanics easy to understand before adding model tool calling.
"""

from langgraph.graph import END, START, StateGraph

from app.banking_tools import get_account_balance, get_card_status, get_recent_transactions
from app.state import BankingState


def classify_request(state: BankingState) -> dict:
    """Classify the customer query into one supported demo intent."""
    query = state["user_query"].lower()
    if "balance" in query:
        intent = "balance"
    elif "transaction" in query:
        intent = "transactions"
    elif "card" in query:
        intent = "card_status"
    else:
        intent = "general"
    return {"intent": intent, "trace": [f"Classified request: {intent}"]}


def route_by_intent(state: BankingState) -> str:
    """Return the route label used by the conditional edge."""
    if state.get("intent") in {"balance", "transactions", "card_status"}:
        return "run_banking_tool"
    return "general_response"


def run_banking_tool(state: BankingState) -> dict:
    """Execute one approved, read-only mock function."""
    account_id = state.get("account_id", "DEMO-001")
    intent = state["intent"]

    if intent == "balance":
        result = f"Demo account balance: {get_account_balance(account_id)}"
    elif intent == "transactions":
        transactions = get_recent_transactions(account_id)
        lines = [
            f'{item["date"]}: {item["description"]} ({item["amount"]:,.2f} INR)'
            for item in transactions
        ]
        result = "Recent demo transactions:\n" + "\n".join(lines)
    elif intent == "card_status":
        result = f"Demo debit card status: {get_card_status(account_id)}"
    else:
        result = "No supported tool was selected."

    return {"tool_result": result, "trace": [f"Executed mock tool: {intent}"]}


def general_response(state: BankingState) -> dict:
    """Handle a query that does not need account data."""
    return {
        "final_response": (
            "I can help with demo balance, recent transactions and card status. "
            "This prototype uses fictional data only."
        ),
        "trace": ["Answered without calling an account tool"],
    }


def format_tool_response(state: BankingState) -> dict:
    """Copy the tool result into the final response field."""
    return {
        "final_response": state.get("tool_result", "No result returned."),
        "trace": ["Formatted tool result"],
    }


def build_graph():
    """Build and compile the StateGraph."""
    builder = StateGraph(BankingState)
    builder.add_node("classify_request", classify_request)
    builder.add_node("run_banking_tool", run_banking_tool)
    builder.add_node("general_response", general_response)
    builder.add_node("format_tool_response", format_tool_response)

    builder.add_edge(START, "classify_request")
    builder.add_conditional_edges(
        "classify_request",
        route_by_intent,
        {
            "run_banking_tool": "run_banking_tool",
            "general_response": "general_response",
        },
    )
    builder.add_edge("run_banking_tool", "format_tool_response")
    builder.add_edge("format_tool_response", END)
    builder.add_edge("general_response", END)
    return builder.compile()
