"""Tool schemas and safe dispatch for the fictional banking demo."""

from app.banking_tools import (
    get_account_balance,
    get_card_status,
    get_recent_transactions,
)

# These JSON schemas are sent to the model. Notice that account_id is not
# exposed as a model-controlled argument; the application supplies it.
BANKING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_account_balance",
            "description": "Retrieve the balance for the authenticated demo account.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_transactions",
            "description": "Retrieve recent transactions for the authenticated demo account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of transactions to return, from 1 to 5.",
                        "minimum": 1,
                        "maximum": 5,
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_card_status",
            "description": "Retrieve the debit card status for the authenticated demo account.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def execute_banking_tool(
    tool_name: str, arguments: dict, account_id: str
) -> str:
    """Validate and execute one explicitly allow-listed read-only demo tool."""
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")

    if tool_name == "get_account_balance":
        if arguments:
            raise ValueError("get_account_balance does not accept arguments.")
        return f"Account balance: {get_account_balance(account_id)}"

    if tool_name == "get_recent_transactions":
        unexpected = set(arguments) - {"limit"}
        if unexpected:
            raise ValueError("Unsupported transaction tool arguments.")
        limit = arguments.get("limit", 3)
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("limit must be an integer.")
        transactions = get_recent_transactions(account_id, limit)
        return "\n".join(
            f'{item["date"]}: {item["description"]} '
            f'({item["amount"]:,.2f} INR)'
            for item in transactions
        ) or "No recent transactions found."

    if tool_name == "get_card_status":
        if arguments:
            raise ValueError("get_card_status does not accept arguments.")
        return f"Debit card status: {get_card_status(account_id)}"

    raise ValueError(f"Tool is not allowed: {tool_name}")
