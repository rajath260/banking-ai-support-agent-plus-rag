"""Run a local end-to-end indexing and semantic retrieval demonstration.

Usage:
    python -m app.rag.demo
"""
from pathlib import Path

from app.rag.embeddings import LocalEmbeddingModel
from app.rag.pipeline import index_knowledge_base, retrieve_relevant_chunks
from app.rag.vector_store import ChromaVectorStore


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    knowledge_base = project_root / "knowledge_base"
    persist_directory = project_root / "data" / "chroma"

    print("Loading the local embedding model...")
    embedding_model = LocalEmbeddingModel()
    vector_store = ChromaVectorStore(persist_directory=persist_directory)

    indexed_count = index_knowledge_base(
        knowledge_base,
        embedding_model,
        vector_store,
    )
    print(f"Indexed or updated {indexed_count} FAQ chunks.")
    print("The Chroma collection is persisted under data/chroma.")

    while True:
        query = input("\nAsk a question about the demo banking FAQs (or type 'exit'): ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            print("Please enter a question.")
            continue

        matches = retrieve_relevant_chunks(
            query,
            embedding_model,
            vector_store,
            top_k=3,
        )
        if not matches:
            print("No matching FAQ chunks were found.")
            continue

        print("\nRetrieved FAQ context (lower cosine distance is a closer match):")
        for rank, match in enumerate(matches, start=1):
            source = match["metadata"].get("source", "unknown")
            distance = match["distance"]
            print(f"\n{rank}. Source: {source} | Cosine distance: {distance:.4f}")
            print(match["document"])


if __name__ == "__main__":
    main()
