"""Run example customer queries through the LangGraph workflow."""

from app.graph import build_graph


def main() -> None:
    graph = build_graph()
    questions = [
        "What is my account balance?",
        "Show my recent transactions.",
        "Is my debit card active?",
        "How do I reset my banking password?",
    ]

    for question in questions:
        result = graph.invoke(
            {"user_query": question, "account_id": "DEMO-001", "trace": []}
        )
        print("\n" + "=" * 60)
        print("Customer:", question)
        print("Agent:", result["final_response"])
        print("Trace:")
        for item in result.get("trace", []):
            print(" -", item)


if __name__ == "__main__":
    main()
