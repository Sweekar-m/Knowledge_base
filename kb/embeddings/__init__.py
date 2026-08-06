"""kb.embeddings package — embedding provider factory."""

from __future__ import annotations

from typing import Optional

from kb.embeddings.provider import EmbeddingProvider

_provider: Optional[EmbeddingProvider] = None


def get_embedder() -> EmbeddingProvider:
    """Return the configured embedding provider (lazy singleton)."""
    global _provider
    if _provider is None:
        from kb.config.settings import get_settings
        settings = get_settings()
        cfg = settings.embeddings

        if cfg.provider == "local":
            from kb.embeddings.local_embedder import LocalEmbedder
            _provider = LocalEmbedder(model_name=cfg.model, batch_size=cfg.batch_size)
        else:
            # Fallback to local if unknown provider
            from kb.embeddings.local_embedder import LocalEmbedder
            _provider = LocalEmbedder(model_name="all-MiniLM-L6-v2", batch_size=cfg.batch_size)

    return _provider
