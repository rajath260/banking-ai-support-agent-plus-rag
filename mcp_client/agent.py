"""Interactive multi-turn agent using dynamically discovered MCP tools."""

import asyncio

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from app.llm import get_client
from app.mcp_agent_graph import build_mcp_agent_graph
from mcp_client.client import discover_tools, server_parameters


async def run_turn(graph, question: str, thread_id: str = "local-demo-chat") -> dict:
    """Run one turn using the same checkpoint thread as earlier turns."""
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    return await graph.ainvoke(
        {"messages": [{"role": "user", "content": question.strip()}]},
        config={"configurable": {"thread_id": thread_id}},
    )


async def main() -> None:
    print("Banking AI Support Agent (MCP-connected demo data)")
    print("Conversation memory is active for this session.")
    print("Type 'exit' or 'quit' to end the conversation.")

    model_client = get_client()
    async with stdio_client(server_parameters()) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await discover_tools(session)
            print("\nDiscovered MCP tools:")
            for tool in tools:
                print(f"- {tool['name']}: {tool['description']}")

            # Keep one graph, checkpointer and MCP session alive across turns.
            graph = build_mcp_agent_graph(
                session=session,
                discovered_tools=tools,
                client=model_client,
            )
            thread_id = "local-demo-chat"

            while True:
                question = input("\nYou: ").strip()
                if question.lower() in {"exit", "quit"}:
                    print("Conversation ended.")
                    break
                if not question:
                    print("Please enter a question.")
                    continue

                result = await run_turn(graph, question, thread_id)
                print("\nAssistant:", result.get("final_response", "No response generated."))


if __name__ == "__main__":
    asyncio.run(main())
