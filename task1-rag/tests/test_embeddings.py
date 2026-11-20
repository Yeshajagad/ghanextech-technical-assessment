from types import SimpleNamespace

import numpy as np
import ollama
import pytest

from rag import config
from rag.embeddings import BATCH_SIZE, Embedder, EmbeddingError


class FakeClient:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def embed(self, model, input):
        if self.error:
            raise self.error
        self.calls.append((model, list(input)))
        # 3 numbers per text, deliberately not unit length
        return SimpleNamespace(embeddings=[[3.0, 4.0, 0.0] for _ in input])


def test_vectors_are_unit_length():
    vectors = Embedder(client=FakeClient()).embed(["hello"])
    assert vectors.dtype == np.float32
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0)


def test_texts_are_sent_in_batches():
    client = FakeClient()
    vectors = Embedder(client=client).embed(["x"] * (BATCH_SIZE + 4))
    assert len(vectors) == BATCH_SIZE + 4
    assert [len(call[1]) for call in client.calls] == [BATCH_SIZE, 4]
    assert client.calls[0][0] == config.OLLAMA_EMBED_MODEL


def test_ollama_not_running_gives_friendly_error():
    embedder = Embedder(client=FakeClient(error=ConnectionError("refused")))
    with pytest.raises(EmbeddingError, match="Cannot reach Ollama"):
        embedder.embed(["hello"])


def test_missing_model_tells_the_user_what_to_pull():
    embedder = Embedder(client=FakeClient(error=ollama.ResponseError("not found", 404)))
    with pytest.raises(EmbeddingError, match="ollama pull"):
        embedder.embed(["hello"])


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "EMBED_PROVIDER", "banana")
    with pytest.raises(EmbeddingError, match="EMBED_PROVIDER"):
        Embedder()