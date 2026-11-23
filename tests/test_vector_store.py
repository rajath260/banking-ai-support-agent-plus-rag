import pytest

from app.rag.document_loader import Document
from app.rag.vector_store import ChromaVectorStore


class FakeCollection:
    def __init__(self):
        self.upsert_call = None
        self.query_call = None
        self.query_result = {
            "ids": [["debit_card_faq.md::chunk-0"]],
            "documents": [["Immediately block a lost or stolen debit card."]],
            "metadatas": [[{"source": "debit_card_faq.md", "chunk_index": 0}]],
            "distances": [[0.12]],
        }

    def upsert(self, **kwargs):
        self.upsert_call = kwargs

    def query(self, **kwargs):
        self.query_call = kwargs
        return self.query_result


def sample_chunks():
    return [
        Document(
            content="Immediately block a lost or stolen debit card.",
            metadata={"source": "debit_card_faq.md", "chunk_index": 0},
        ),
        Document(
            content="Contact support if a payment is pending.",
            metadata={"source": "digital_payments_faq.md", "chunk_index": 1},
        ),
    ]


def test_upsert_chunks_sends_documents_metadata_and_embeddings():
    collection = FakeCollection()
    store = ChromaVectorStore(collection=collection)

    count = store.upsert_chunks(sample_chunks(), [[0.1, 0.2], [0.3, 0.4]])

    assert count == 2
    assert collection.upsert_call["ids"] == [
        "debit_card_faq.md::chunk-0",
        "digital_payments_faq.md::chunk-1",
    ]
    assert collection.upsert_call["documents"][0].startswith("Immediately block")
    assert collection.upsert_call["metadatas"][0]["source"] == "debit_card_faq.md"
    assert collection.upsert_call["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]


def test_upsert_chunks_returns_zero_for_empty_input():
    collection = FakeCollection()
    store = ChromaVectorStore(collection=collection)

    assert store.upsert_chunks([], []) == 0
    assert collection.upsert_call is None


def test_upsert_chunks_rejects_mismatched_counts():
    store = ChromaVectorStore(collection=FakeCollection())

    with pytest.raises(ValueError, match="number of chunks"):
        store.upsert_chunks(sample_chunks(), [[0.1, 0.2]])


def test_upsert_chunks_rejects_empty_embedding():
    store = ChromaVectorStore(collection=FakeCollection())

    with pytest.raises(ValueError, match="Embedding at position 0"):
        store.upsert_chunks(sample_chunks(), [[], [0.3, 0.4]])


def test_search_returns_retrieved_text_metadata_and_distance():
    collection = FakeCollection()
    store = ChromaVectorStore(collection=collection)

    matches = store.search([0.5, 0.6], top_k=2)

    assert len(matches) == 1
    assert matches[0]["document"] == "Immediately block a lost or stolen debit card."
    assert matches[0]["metadata"]["source"] == "debit_card_faq.md"
    assert matches[0]["distance"] == 0.12
    assert collection.query_call["n_results"] == 2
    assert collection.query_call["query_embeddings"] == [[0.5, 0.6]]


def test_search_rejects_empty_query_embedding():
    store = ChromaVectorStore(collection=FakeCollection())

    with pytest.raises(ValueError, match="Query embedding"):
        store.search([])


def test_search_rejects_non_positive_top_k():
    store = ChromaVectorStore(collection=FakeCollection())

    with pytest.raises(ValueError, match="top_k"):
        store.search([0.1, 0.2], top_k=0)


def test_search_handles_no_matches():
    collection = FakeCollection()
    collection.query_result = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    store = ChromaVectorStore(collection=collection)

    assert store.search([0.1, 0.2]) == []
