"""Persistent Chroma vector storage for the RAG pipeline.

The module accepts embeddings from the embedding adapter rather than creating
them internally. This keeps vector storage independent from the model runtime
and makes the storage layer straightforward to test offline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from app.rag.document_loader import Document


DEFAULT_COLLECTION_NAME = "banking_faq"


class ChromaVectorStore:
    """Persist document chunks and retrieve them by vector similarity."""

    def __init__(
        self,
        persist_directory: str | Path = "data/chroma",
        collection_name: str = DEFAULT_COLLECTION_NAME,
        collection=None,
    ) -> None:
        if collection is not None:
            self.collection = collection
            return

        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "Chroma storage requires the optional dependency. "
                "Install it with: pip install -r requirements-rag.txt"
            ) from exc

        client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(
        self,
        chunks: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Insert or update chunks using stable source-and-index identifiers."""
        if len(chunks) != len(embeddings):
            raise ValueError("The number of chunks must match the number of embeddings")
        if not chunks:
            return 0

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, str | int]] = []
        vectors: list[list[float]] = []

        for position, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if not chunk.content.strip():
                raise ValueError(f"Chunk at position {position} cannot be empty")
            if not embedding:
                raise ValueError(f"Embedding at position {position} cannot be empty")

            source = str(chunk.metadata.get("source", "unknown"))
            chunk_index = int(chunk.metadata.get("chunk_index", position))
            ids.append(f"{source}::chunk-{chunk_index}")
            documents.append(chunk.content)
            metadatas.append(dict(chunk.metadata))
            vectors.append([float(value) for value in embedding])

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=vectors,
        )
        return len(ids)

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 3,
    ) -> list[dict]:
        """Return the closest chunks; Chroma's cosine distance is lower-is-better."""
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        result = self.collection.query(
            query_embeddings=[[float(value) for value in query_embedding]],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        matches: list[dict] = []
        for index, document in enumerate(documents):
            matches.append(
                {
                    "id": ids[index],
                    "document": document,
                    "metadata": metadatas[index],
                    "distance": distances[index],
                }
            )
        return matches
