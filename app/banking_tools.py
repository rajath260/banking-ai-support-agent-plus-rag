"""Read-only mock banking tools. All data in this module is fictional."""

MOCK_ACCOUNTS = {
    "DEMO-001": {
        "balance": 45000.00,
        "currency": "INR",
        "card_status": "ACTIVE",
        "transactions": [
            {"date": "2026-11-15", "description": "Grocery Store", "amount": -1250.00},
            {"date": "2026-11-14", "description": "Salary Credit", "amount": 55000.00},
            {"date": "2026-11-12", "description": "Electricity Bill", "amount": -1800.00},
        ],
    }
}


def _account(account_id: str) -> dict:
    """Return a demo account or raise a generic error."""
    try:
        return MOCK_ACCOUNTS[account_id]
    except KeyError as exc:
        raise ValueError("Demo account was not found.") from exc


def get_account_balance(account_id: str = "DEMO-001") -> str:
    """Return the balance for a fictional demo account."""
    account = _account(account_id)
    return f'{account["currency"]} {account["balance"]:,.2f}'


def get_recent_transactions(account_id: str = "DEMO-001", limit: int = 3) -> list[dict]:
    """Return up to five recent fictional transactions."""
    if not 1 <= limit <= 5:
        raise ValueError("limit must be between 1 and 5.")
    return _account(account_id)["transactions"][:limit]


def get_card_status(account_id: str = "DEMO-001") -> str:
    """Return the fictional demo debit card status."""
    return _account(account_id)["card_status"]
