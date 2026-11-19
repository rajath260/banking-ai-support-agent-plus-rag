"""Local text embeddings for the RAG pipeline.

The model is downloaded once from Hugging Face and then runs locally. No hosted
inference API or paid inference credits are used by this module.
"""
from __future__ import annotations

from typing import Sequence


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbeddingModel:
    """Create dense vectors for documents and user queries."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Local embeddings require the optional dependency. "
                "Install it with: pip install -r requirements-rag.txt"
            ) from exc

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode document chunks as normalized vectors."""
        if not texts:
            return []

        vectors = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Encode one user query using the same embedding space."""
        if not text.strip():
            raise ValueError("Query text cannot be empty")

        vectors = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors[0].tolist()
