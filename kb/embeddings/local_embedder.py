"""Local sentence-transformers embedding provider."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from kb.embeddings.provider import EmbeddingProvider

_model = None
_model_name: Optional[str] = None


def _load_model(model_name: str):
    global _model, _model_name
    if _model is None or _model_name != model_name:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_name)
        _model_name = model_name
    return _model


class LocalEmbedder(EmbeddingProvider):
    """Wraps sentence-transformers for local, offline embedding."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
        self._model_name = model_name
        self._batch_size = batch_size
        self._dim: Optional[int] = None

    def _get_model(self):
        return _load_model(self._model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Batch-encode texts. Returns float32 array of shape (N, dim)."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        model = self._get_model()
        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,   # cosine similarity via inner product
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text. Returns float32 array of shape (dim,)."""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        if self._dim is None:
            model = self._get_model()
            self._dim = model.get_sentence_embedding_dimension()
        return self._dim


def get_local_embedder(model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64) -> LocalEmbedder:
    return LocalEmbedder(model_name=model_name, batch_size=batch_size)
