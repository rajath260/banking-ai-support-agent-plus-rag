"""LangGraph agent with MCP tools and checkpoint-backed conversation memory."""

import asyncio
import json
from operator import add
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.llm import MODEL_ID, get_client


SYSTEM_PROMPT = (
    "You are a banking support assistant for a fictional demo. "
    "Use the available MCP tools for account-specific questions. "
    "Never invent balances, transactions or card status. "
    "The MCP server exposes read-only tools for one fixed fictional demo account. "
    "Use the conversation history to understand follow-up questions. "
    "Do not claim a tool was used unless its result is provided."
)


class MCPAgentState(TypedDict, total=False):
    messages: Annotated[list[dict], add]
    tool_calls: list[dict]
    assistant_message: dict
    model_text: str
    tool_results: list[dict]
    final_response: str
    trace: Annotated[list[str], add]


def mcp_tools_to_llm_tools(discovered_tools: list[dict]) -> list[dict]:
    """Convert MCP tool metadata into OpenAI-compatible function schemas."""
    converted = []
    for tool in discovered_tools:
        schema = tool.get("input_schema") or {
            "type": "object",
            "properties": {},
        }
        parameters = {
            key: schema[key]
            for key in ("type", "properties", "required", "additionalProperties")
            if key in schema
        }
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": parameters,
                },
            }
        )
    return converted


def _normalise_tool_calls(message) -> list[dict]:
    """Convert SDK tool-call objects into validated plain dictionaries."""
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


def _mcp_result_text(result) -> str:
    """Extract text content from an MCP CallToolResult."""
    parts = [
        item.text
        for item in result.content
        if getattr(item, "type", None) == "text"
    ]
    text = "\n".join(parts)
    if result.isError:
        raise RuntimeError(text or "MCP server reported a tool execution error.")
    return text or "The MCP tool returned no text content."


def build_mcp_agent_graph(
    session,
    discovered_tools: list[dict],
    client=None,
    checkpointer=None,
):
    """Build an async MCP agent graph with thread-scoped checkpoint memory.

    Keep the MCP session open while invoking the graph. Reuse the same graph
    and configurable thread_id for every turn in one conversation.
    """
    client = client or get_client()
    checkpointer = checkpointer or MemorySaver()
    llm_tools = mcp_tools_to_llm_tools(discovered_tools)
    allowed_tool_names = {tool["name"] for tool in discovered_tools}

    async def call_model(state: MCPAgentState) -> dict:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *state.get("messages", []),
            ],
            tools=llm_tools,
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
            "messages": [assistant_message],
            "trace": ["LLM evaluated the conversation using checkpointed history"],
        }

    def route_after_model(state: MCPAgentState) -> str:
        return "execute_mcp_tools" if state.get("tool_calls") else "finish"

    async def execute_mcp_tools(state: MCPAgentState) -> dict:
        results = []
        tool_messages = []
        for call in state.get("tool_calls", []):
            if call["name"] not in allowed_tool_names:
                output = (
                    f"Tool request rejected: '{call['name']}' "
                    "was not discovered from the MCP server."
                )
            else:
                try:
                    mcp_result = await session.call_tool(
                        call["name"], arguments=call["arguments"]
                    )
                    output = _mcp_result_text(mcp_result)
                except Exception as exc:
                    output = f"MCP tool execution failed: {exc}"
            results.append({"tool_call_id": call["id"], "content": output})
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": output,
                }
            )
        return {
            "tool_results": results,
            "messages": tool_messages,
            "trace": [f"MCP client invoked {len(results)} requested tool call(s)"],
        }

    async def finish(state: MCPAgentState) -> dict:
        if not state.get("tool_calls"):
            answer = state.get(
                "model_text", "I could not generate a response."
            )
            return {
                "final_response": answer,
                "trace": ["Model answered without requesting an MCP tool"],
            }

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *state.get("messages", []),
            ],
            tool_choice="none",
            max_tokens=250,
            temperature=0.1,
        )
        answer = response.choices[0].message.content or ""
        return {
            "final_response": answer,
            "messages": [{"role": "assistant", "content": answer}],
            "trace": ["LLM generated its final response using MCP tool output"],
        }

    builder = StateGraph(MCPAgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_mcp_tools", execute_mcp_tools)
    builder.add_node("finish", finish)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "execute_mcp_tools": "execute_mcp_tools",
            "finish": "finish",
        },
    )
    builder.add_edge("execute_mcp_tools", "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer)
