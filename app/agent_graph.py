"""LangGraph workflow that lets a Hugging Face model request banking tools."""

import json
from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent_tools import BANKING_TOOLS, execute_banking_tool
from app.llm import MODEL_ID, get_client


SYSTEM_PROMPT = (
    "You are a banking support assistant for a fictional demo. "
    "Use the supplied tools whenever the user asks for account-specific "
    "information such as balance, transactions, or card status. "
    "Never guess account information. Do not claim a tool was run unless "
    "you receive its result. The tools are read-only and operate on demo data."
)


class AgentState(TypedDict, total=False):
    user_query: str
    account_id: str
    tool_calls: list[dict]
    assistant_message: dict
    model_text: str
    tool_results: list[dict]
    final_response: str
    trace: Annotated[list[str], add]


def _normalise_tool_calls(message) -> list[dict]:
    """Convert SDK tool-call objects into plain dictionaries."""
    calls = []
    for call in message.tool_calls or []:
        raw_arguments = call.function.arguments or {}
        if isinstance(raw_arguments, str):
            raw_arguments = json.loads(raw_arguments)
        if not isinstance(raw_arguments, dict):
            raise ValueError("The model returned invalid tool arguments.")
        calls.append(
            {
                "id": str(call.id),
                "name": call.function.name,
                "arguments": raw_arguments,
            }
        )
    return calls


def build_agent_graph(client=None):
    """Build the model -> optional tool -> model response graph.

    A client can be injected for tests. In normal use, credentials are loaded
    lazily from the local .env file by get_client().
    """
    client = client or get_client()

    def call_model(state: AgentState) -> dict:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": state["user_query"]},
            ],
            tools=BANKING_TOOLS,
            tool_choice="auto",
            max_tokens=300,
            temperature=0.1,
        )
        message = response.choices[0].message
        calls = _normalise_tool_calls(message)
        assistant_message = {
            "role": "assistant",
            "content": message.content or "",
        }
        if calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"]),
                    },
                }
                for call in calls
            ]
        return {
            "tool_calls": calls,
            "assistant_message": assistant_message,
            "model_text": message.content or "",
            "trace": ["Hugging Face model evaluated the user request"],
        }

    def route_after_model(state: AgentState) -> str:
        return "execute_tools" if state.get("tool_calls") else "finish"

    def execute_tools(state: AgentState) -> dict:
        account_id = state.get("account_id", "DEMO-001")
        results = []
        for call in state.get("tool_calls", []):
            try:
                result = execute_banking_tool(
                    call["name"], call["arguments"], account_id
                )
            except (ValueError, KeyError, TypeError) as exc:
                result = f"Tool request rejected: {exc}"
            results.append({"tool_call_id": call["id"], "content": result})
        return {
            "tool_results": results,
            "trace": [f'Application executed {len(results)} approved tool call(s)'],
        }

    def finish(state: AgentState) -> dict:
        if not state.get("tool_calls"):
            return {
                "final_response": state.get("model_text", "I could not generate a response."),
                "trace": ["Model answered without requesting a banking tool"],
            }

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state["user_query"]},
            state["assistant_message"],
        ]
        messages.extend(
            {
                "role": "tool",
                "tool_call_id": item["tool_call_id"],
                "content": item["content"],
            }
            for item in state.get("tool_results", [])
        )
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            tool_choice="none",
            max_tokens=250,
            temperature=0.1,
        )
        answer = response.choices[0].message.content or ""
        return {
            "final_response": answer,
            "trace": ["Hugging Face model generated the final response from tool output"],
        }

    builder = StateGraph(AgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tools", execute_tools)
    builder.add_node("finish", finish)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        route_after_model,
        {"execute_tools": "execute_tools", "finish": "finish"},
    )
    builder.add_edge("execute_tools", "finish")
    builder.add_edge("finish", END)
    return builder.compile()
