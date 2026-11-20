"""A small in-memory vector store built on FAISS."""

import faiss


class VectorStore:
    def __init__(self, embedder):
        self.embedder = embedder
        self.index = None
        self.chunks = []

    def build(self, chunks):
        """Throw away any old index and build a new one from `chunks`."""
        self.index = None
        self.chunks = []
        if not chunks:
            return

        vectors = self.embedder.embed([c["text"] for c in chunks])
        # Vectors are unit length, so inner product equals cosine similarity.
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self.index = index
        self.chunks = [dict(c, chunk_id=i) for i, c in enumerate(chunks)]

    def search(self, question, k=4):
        """Return up to k chunks, best first, each with a 'score' added."""
        if self.index is None:
            return []

        query = self.embedder.embed([question])
        k = min(k, len(self.chunks))
        scores, ids = self.index.search(query, k)

        results = []
        for score, i in zip(scores[0], ids[0]):
            if i == -1:
                continue
            results.append(dict(self.chunks[i], score=float(score)))
        return results