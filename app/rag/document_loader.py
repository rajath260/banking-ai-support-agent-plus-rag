"""Load simple knowledge-base documents and split them into overlapping chunks.

This first RAG milestone uses only the Python standard library. Chunking is
character-based for transparency; a later milestone can introduce token-aware
chunking and embeddings.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """A source document with text and traceable metadata."""

    content: str
    metadata: dict[str, str | int]


def load_markdown_documents(directory: str | Path) -> list[Document]:
    """Load every Markdown file in a directory in stable filename order."""
    source_dir = Path(directory)
    documents: list[Document] = []

    for path in sorted(source_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        if content:
            documents.append(Document(content=content, metadata={"source": path.name}))

    return documents


def chunk_document(document: Document, chunk_size: int = 500, overlap: int = 75) -> list[Document]:
    """Split text into character windows and retain source metadata.

    Adjacent chunks share overlap characters so context near a boundary is less
    likely to be lost. These sizes are character counts, not token counts.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = document.content.strip()
    if not text:
        return []

    chunks: list[Document] = []
    start = 0
    chunk_index = 0
    step = chunk_size - overlap

    while start < len(text):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(Document(content=piece, metadata={**document.metadata, "chunk_index": chunk_index}))
            chunk_index += 1
        if start + chunk_size >= len(text):
            break
        start += step

    return chunks


def load_and_chunk_documents(directory: str | Path, chunk_size: int = 500, overlap: int = 75) -> list[Document]:
    """Load all Markdown files and return their chunks."""
    chunks: list[Document] = []
    for document in load_markdown_documents(directory):
        chunks.extend(chunk_document(document, chunk_size, overlap))
    return chunks
