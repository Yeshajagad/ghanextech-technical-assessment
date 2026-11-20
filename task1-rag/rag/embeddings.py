"""Turn text into vectors.

Local: Ollama running bge-m3.
Deployed demo: all-MiniLM-L6-v2 through fastembed, which needs no PyTorch (EMBED_PROVIDER=minilm).
"""

import numpy as np
import ollama

from rag import config

BATCH_SIZE = 16


class EmbeddingError(Exception):
    """Raised with a message that is safe to show to the user."""


class Embedder:
    def __init__(self, client=None):
        self.provider = config.EMBED_PROVIDER
        self.client = None
        self.model = None

        if self.provider == "ollama":
            self.client = client or ollama.Client(
                host=config.OLLAMA_HOST, timeout=config.OLLAMA_TIMEOUT
            )
        elif self.provider == "minilm":
            # Imported here so a local setup does not need this package installed.
            from fastembed import TextEmbedding

            self.model = TextEmbedding(config.MINILM_MODEL)
        else:
            raise EmbeddingError(
                f"EMBED_PROVIDER must be 'ollama' or 'minilm' (got '{self.provider}')."
            )

    @property
    def model_name(self):
        return config.OLLAMA_EMBED_MODEL if self.provider == "ollama" else config.MINILM_MODEL

    def embed(self, texts):
        """Return a float32 array of unit-length vectors, one row per text."""
        if self.provider == "ollama":
            vectors = self._embed_ollama(texts)
        else:
            vectors = list(self.model.embed(texts, batch_size=32))

        vectors = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid dividing an all-zero vector by zero
        return vectors / norms

    def _embed_ollama(self, texts):
        vectors = []
        try:
            for start in range(0, len(texts), BATCH_SIZE):
                batch = texts[start:start + BATCH_SIZE]
                response = self.client.embed(model=config.OLLAMA_EMBED_MODEL, input=batch)
                vectors.extend(response.embeddings)
        except ConnectionError:
            raise EmbeddingError(
                f"Cannot reach Ollama at {config.OLLAMA_HOST}. "
                "Start it with `ollama serve` (or open the Ollama app)."
            )
        except ollama.ResponseError as error:
            if error.status_code == 404:
                raise EmbeddingError(
                    f"The embedding model '{config.OLLAMA_EMBED_MODEL}' is not installed. "
                    f"Run: ollama pull {config.OLLAMA_EMBED_MODEL}"
                )
            raise EmbeddingError(f"Ollama returned an error while embedding: {error.error}")
        return vectors