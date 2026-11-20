import numpy as np

from rag.store import VectorStore


class FakeEmbedder:
    """Word-count vectors, so the store can be tested without Ollama."""

    vocab = ["cat", "dog", "bank", "river", "money", "water"]

    def embed(self, texts):
        rows = []
        for text in texts:
            words = text.lower().split()
            row = np.array([words.count(w) for w in self.vocab], dtype="float32")
            norm = np.linalg.norm(row)
            rows.append(row / norm if norm else row)
        return np.vstack(rows).astype("float32")


def test_search_returns_most_similar_chunk_first():
    store = VectorStore(FakeEmbedder())
    store.build(
        [
            {"text": "the cat and the dog", "source": "a.pdf", "page": 1},
            {"text": "money in the bank", "source": "a.pdf", "page": 2},
            {"text": "water in the river", "source": "b.pdf", "page": 1},
        ]
    )
    results = store.search("river water", k=2)
    assert results[0]["source"] == "b.pdf"
    assert results[0]["score"] > results[1]["score"]


def test_build_replaces_the_old_index():
    store = VectorStore(FakeEmbedder())
    store.build([{"text": "cat cat", "source": "old.pdf", "page": 1}])
    store.build([{"text": "bank money", "source": "new.pdf", "page": 1}])
    assert [c["source"] for c in store.chunks] == ["new.pdf"]
    assert store.search("cat", k=3)[0]["source"] == "new.pdf"


def test_k_larger_than_chunk_count_is_fine():
    store = VectorStore(FakeEmbedder())
    store.build([{"text": "cat", "source": "a.pdf", "page": 1}])
    assert len(store.search("cat", k=8)) == 1


def test_search_on_empty_store_returns_nothing():
    assert VectorStore(FakeEmbedder()).search("anything") == []