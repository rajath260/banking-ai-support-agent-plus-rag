import pytest

from app.rag.pipeline import index_knowledge_base, retrieve_relevant_chunks


class FakeEmbeddingModel:
    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    def embed_documents(self, texts):
        self.document_calls.append(texts)
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, text):
        self.query_calls.append(text)
        return [0.3, 0.4]


class FakeVectorStore:
    def __init__(self):
        self.upsert_call = None
        self.search_call = None

    def upsert_chunks(self, chunks, embeddings):
        self.upsert_call = (chunks, embeddings)
        return len(chunks)

    def search(self, query_embedding, top_k=3):
        self.search_call = (query_embedding, top_k)
        return [{"document": "Relevant FAQ text", "metadata": {"source": "faq.md"}, "distance": 0.1}]


def test_index_knowledge_base_loads_chunks_embeds_and_stores(tmp_path):
    (tmp_path / "cards.md").write_text(
        "# Cards\n\nBlock a lost debit card immediately using official support.",
        encoding="utf-8",
    )
    embedder = FakeEmbeddingModel()
    store = FakeVectorStore()

    count = index_knowledge_base(tmp_path, embedder, store, chunk_size=200, overlap=20)

    assert count == 1
    assert len(embedder.document_calls) == 1
    assert "Block a lost debit card" in embedder.document_calls[0][0]
    assert store.upsert_call[1] == [[0.1, 0.2]]


def test_index_knowledge_base_does_not_embed_empty_directory(tmp_path):
    embedder = FakeEmbeddingModel()
    store = FakeVectorStore()

    count = index_knowledge_base(tmp_path, embedder, store)

    assert count == 0
    assert embedder.document_calls == []
    assert store.upsert_call is None


def test_retrieve_relevant_chunks_embeds_query_and_searches():
    embedder = FakeEmbeddingModel()
    store = FakeVectorStore()

    matches = retrieve_relevant_chunks(
        "How can I stop a stolen ATM card?",
        embedder,
        store,
        top_k=2,
    )

    assert embedder.query_calls == ["How can I stop a stolen ATM card?"]
    assert store.search_call == ([0.3, 0.4], 2)
    assert matches[0]["metadata"]["source"] == "faq.md"


def test_retrieve_rejects_blank_query():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        retrieve_relevant_chunks("   ", FakeEmbeddingModel(), FakeVectorStore())


def test_retrieve_rejects_non_positive_top_k():
    with pytest.raises(ValueError, match="top_k"):
        retrieve_relevant_chunks("lost card", FakeEmbeddingModel(), FakeVectorStore(), top_k=0)
