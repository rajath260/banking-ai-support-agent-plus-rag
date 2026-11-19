"""Shared state schema and reducer example for the LangGraph workflow."""

from operator import add
from typing import Annotated, TypedDict


class BankingState(TypedDict, total=False):
    """Shared data passed between graph nodes.

    Nodes return partial updates. The trace reducer appends new trace entries
    instead of replacing the existing list.
    """

    user_query: str
    account_id: str
    intent: str
    tool_result: str
    final_response: str
    trace: Annotated[list[str], add]
