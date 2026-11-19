import sys
import types

import pytest

from app.rag.embeddings import LocalEmbeddingModel


class FakeVector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeVectors:
    def __init__(self, rows):
        self.rows = rows

    def tolist(self):
        return self.rows

    def __getitem__(self, index):
        return FakeVector(self.rows[index])


class FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return FakeVectors([[0.1, 0.2, 0.3] for _ in texts])


def install_fake_sentence_transformers(monkeypatch):
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)


def test_local_embedding_model_encodes_document_chunks(monkeypatch):
    install_fake_sentence_transformers(monkeypatch)
    embedder = LocalEmbeddingModel(model_name="test-model")

    vectors = embedder.embed_documents(["lost card", "pending payment"])

    assert vectors == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert embedder.model.model_name == "test-model"
    assert embedder.model.calls[0][1]["normalize_embeddings"] is True


def test_local_embedding_model_encodes_query(monkeypatch):
    install_fake_sentence_transformers(monkeypatch)
    embedder = LocalEmbeddingModel()

    vector = embedder.embed_query("How do I block my card?")

    assert vector == [0.1, 0.2, 0.3]


def test_local_embedding_model_returns_empty_for_no_documents(monkeypatch):
    install_fake_sentence_transformers(monkeypatch)
    embedder = LocalEmbeddingModel()

    assert embedder.embed_documents([]) == []
    assert embedder.model.calls == []


def test_local_embedding_model_rejects_blank_query(monkeypatch):
    install_fake_sentence_transformers(monkeypatch)
    embedder = LocalEmbeddingModel()

    with pytest.raises(ValueError, match="Query text cannot be empty"):
        embedder.embed_query("   ")
