"""Orchestrate document indexing and semantic retrieval for the RAG demo."""
from __future__ import annotations

from pathlib import Path

from app.rag.document_loader import load_and_chunk_documents


def index_knowledge_base(
    knowledge_base_dir: str | Path,
    embedding_model,
    vector_store,
    chunk_size: int = 500,
    overlap: int = 75,
) -> int:
    """Load, chunk, embed and persist all Markdown knowledge-base documents."""
    chunks = load_and_chunk_documents(
        knowledge_base_dir,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if not chunks:
        return 0

    embeddings = embedding_model.embed_documents([chunk.content for chunk in chunks])
    return vector_store.upsert_chunks(chunks, embeddings)


def retrieve_relevant_chunks(
    query: str,
    embedding_model,
    vector_store,
    top_k: int = 3,
) -> list[dict]:
    """Embed a user's question and retrieve the closest stored FAQ chunks."""
    if not query.strip():
        raise ValueError("Query cannot be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    query_embedding = embedding_model.embed_query(query)
    return vector_store.search(query_embedding, top_k=top_k)
