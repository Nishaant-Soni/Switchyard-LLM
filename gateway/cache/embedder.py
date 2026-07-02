"""Prompt embedding for the semantic cache.

`Embedder` is a structural protocol so the cache can take either the real MiniLM embedder or a
fake deterministic one in tests (keeps CI hermetic — no torch weights download). Embeddings are
**L2-normalized**, so a cosine-similarity cache reduces to inner-product search in FAISS.
"""

import threading
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
        self._lock = threading.Lock()

    def _ensure_model(self):
        # Double-checked lock: embed runs in a threadpool (see main.py), so concurrent first calls
        # must not each load the ~80MB model. Fast path skips the lock once loaded.
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_model()
        vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype="float32")
