from pathlib import Path

import pytest

from app.rag.document_loader import Document, chunk_document, load_and_chunk_documents, load_markdown_documents


def test_load_markdown_documents_reads_files_and_source_metadata(tmp_path: Path):
    (tmp_path / "b.md").write_text("# Second\nContent B", encoding="utf-8")
    (tmp_path / "a.md").write_text("# First\nContent A", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("Not Markdown", encoding="utf-8")

    documents = load_markdown_documents(tmp_path)

    assert len(documents) == 2
    assert documents[0].metadata["source"] == "a.md"
    assert documents[1].metadata["source"] == "b.md"


def test_chunk_document_preserves_source_and_adds_chunk_index():
    document = Document(content="ABCDEFGHIJKLMNOPQRSTUVWXYZ", metadata={"source": "sample.md"})
    chunks = chunk_document(document, chunk_size=10, overlap=2)

    assert [chunk.content for chunk in chunks] == ["ABCDEFGHIJ", "IJKLMNOPQR", "QRSTUVWXYZ"]
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 2]
    assert all(chunk.metadata["source"] == "sample.md" for chunk in chunks)


def test_chunk_document_returns_empty_for_empty_content():
    assert chunk_document(Document(content="  ", metadata={})) == []


@pytest.mark.parametrize("chunk_size,overlap", [(0, 0), (10, 10), (10, 11), (10, -1)])
def test_chunk_document_rejects_invalid_sizes(chunk_size: int, overlap: int):
    with pytest.raises(ValueError):
        chunk_document(Document(content="some text", metadata={}), chunk_size=chunk_size, overlap=overlap)


def test_load_and_chunk_documents_reads_demo_knowledge_base():
    knowledge_base = Path(__file__).parents[1] / "knowledge_base"
    chunks = load_and_chunk_documents(knowledge_base, chunk_size=300, overlap=40)

    assert chunks
    assert {chunk.metadata["source"] for chunk in chunks} == {
        "debit_card_faq.md", "digital_payments_faq.md", "account_security_faq.md"
    }
