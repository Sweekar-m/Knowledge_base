"""Abstract embedding provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EmbeddingProvider(ABC):

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return a 2D float32 array of shape (len(texts), dim)."""
        ...

    @abstractmethod
    def embed_one(self, text: str) -> np.ndarray:
        """Return a 1D float32 array of shape (dim,)."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...
