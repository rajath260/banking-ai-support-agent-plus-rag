"""MCP server publishing fictional banking tools over stdio.

The demo account is fixed in the server so the model cannot choose an account ID.
"""

from mcp.server.fastmcp import FastMCP

from app.banking_tools import get_account_balance, get_card_status, get_recent_transactions

mcp = FastMCP("demo-banking-tools")

DEMO_ACCOUNT_ID = "DEMO-001"


@mcp.tool()
def account_balance() -> str:
    """Get the balance of the authenticated fictional demo account."""
    return get_account_balance(DEMO_ACCOUNT_ID)


@mcp.tool()
def recent_transactions(limit: int = 3) -> list[dict]:
    """Get recent transactions for the authenticated demo account. Limit must be from 1 to 5."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 5:
        raise ValueError("limit must be an integer between 1 and 5")
    return get_recent_transactions(DEMO_ACCOUNT_ID, limit)


@mcp.tool()
def debit_card_status() -> str:
    """Get the debit card status for the authenticated fictional demo account."""
    return get_card_status(DEMO_ACCOUNT_ID)


if __name__ == "__main__":
    mcp.run(transport="stdio")
