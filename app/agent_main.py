"""Run the Hugging Face + LangGraph banking support agent."""

from app.agent_graph import build_agent_graph


def main() -> None:
    graph = build_agent_graph()
    print("Banking AI Support Agent (fictional demo data)")
    print("Try: balance, recent transactions, or debit card status.")
    question = input("\nYou: ").strip()
    if not question:
        print("Please enter a question.")
        return

    result = graph.invoke(
        {
            "user_query": question,
            # In a real application this must come from authenticated server
            # context, never from an untrusted model-generated argument.
            "account_id": "DEMO-001",
            "trace": [],
        }
    )
    print("\nAssistant:", result.get("final_response", "No response generated."))
    print("\nExecution trace:")
    for item in result.get("trace", []):
        print(" -", item)


if __name__ == "__main__":
    main()
