"""Prompt embedding for the semantic cache.

`Embedder` is a structural protocol so the cache can take either the real MiniLM embedder or a
fake deterministic one in tests (keeps CI hermetic — no torch weights download). Embeddings are
**L2-normalized**, so a cosine-similarity cache reduces to inner-product search in FAISS.
"""

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized row embeddings."""
        ...


class SentenceTransformerEmbedder:
    """Local `all-MiniLM-L6-v2` embedder (CPU, 384-dim). The model is loaded lazily on the first
    embed, so importing this module never triggers the ~80MB weights download — only the first real
    cache lookup pays it once."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype="float32")
