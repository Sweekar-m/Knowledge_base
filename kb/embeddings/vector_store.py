"""FAISS vector store — persistent, updatable index for chunk embeddings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_faiss = None


def _get_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


class VectorStore:
    """
    FAISS IndexFlatIP (inner product) store.
    
    Since embeddings are L2-normalized, inner product == cosine similarity.
    Supports: upsert, search, delete, save, load.
    """

    def __init__(self, dimension: int, index_path: Optional[Path] = None):
        self._dim = dimension
        self._index_path = index_path
        self._index = None
        self._id_to_chunk: Dict[int, int] = {}   # faiss_id → db chunk_id
        self._chunk_to_id: Dict[int, int] = {}   # db chunk_id → faiss_id
        self._next_id: int = 0
        self._deleted: set = set()               # faiss IDs marked deleted
        self._dirty: bool = False

        if index_path and index_path.exists():
            self.load(index_path)
        else:
            self._build_index()

    def _build_index(self):
        faiss = _get_faiss()
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self._dim))

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert(self, chunk_id: int, vector: np.ndarray) -> int:
        """Insert or update a chunk embedding. Returns the faiss ID assigned."""
        faiss = _get_faiss()

        # Remove old entry if updating
        if chunk_id in self._chunk_to_id:
            old_fid = self._chunk_to_id[chunk_id]
            self._deleted.add(old_fid)
            del self._id_to_chunk[old_fid]

        fid = self._next_id
        self._next_id += 1

        vec = vector.reshape(1, -1).astype(np.float32)
        ids = np.array([fid], dtype=np.int64)
        self._index.add_with_ids(vec, ids)

        self._id_to_chunk[fid] = chunk_id
        self._chunk_to_id[chunk_id] = fid
        self._dirty = True
        return fid

    def upsert_batch(self, chunk_ids: List[int], vectors: np.ndarray):
        """Batch upsert. vectors shape: (N, dim)."""
        faiss = _get_faiss()

        # Remove old entries
        for chunk_id in chunk_ids:
            if chunk_id in self._chunk_to_id:
                old_fid = self._chunk_to_id[chunk_id]
                self._deleted.add(old_fid)
                del self._id_to_chunk[old_fid]

        fids = list(range(self._next_id, self._next_id + len(chunk_ids)))
        self._next_id += len(chunk_ids)

        ids_arr = np.array(fids, dtype=np.int64)
        vecs = vectors.astype(np.float32)
        self._index.add_with_ids(vecs, ids_arr)

        for fid, chunk_id in zip(fids, chunk_ids):
            self._id_to_chunk[fid] = chunk_id
            self._chunk_to_id[chunk_id] = fid

        self._dirty = True

    def delete(self, chunk_ids: List[int]):
        """Mark chunk embeddings as deleted."""
        for chunk_id in chunk_ids:
            if chunk_id in self._chunk_to_id:
                fid = self._chunk_to_id.pop(chunk_id)
                self._deleted.add(fid)
                self._id_to_chunk.pop(fid, None)
        self._dirty = True

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for top_k nearest neighbours.
        Returns list of (chunk_id, score) pairs sorted by descending score.
        """
        if self._index.ntotal == 0:
            return []

        k = min(top_k + len(self._deleted), self._index.ntotal, top_k * 3)
        vec = query_vector.reshape(1, -1).astype(np.float32)

        scores, ids = self._index.search(vec, k)

        results = []
        for score, fid in zip(scores[0], ids[0]):
            if fid == -1:
                continue
            if fid in self._deleted:
                continue
            chunk_id = self._id_to_chunk.get(int(fid))
            if chunk_id is not None:
                results.append((chunk_id, float(score)))
            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None):
        """Save index and metadata to disk."""
        import pickle

        save_path = path or self._index_path
        if save_path is None:
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        faiss = _get_faiss()
        faiss.write_index(self._index, str(save_path))

        meta_path = save_path.with_suffix(".meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump({
                "id_to_chunk": self._id_to_chunk,
                "chunk_to_id": self._chunk_to_id,
                "next_id": self._next_id,
                "deleted": self._deleted,
                "dim": self._dim,
            }, f)

        self._dirty = False

    def load(self, path: Path):
        """Load index and metadata from disk."""
        import pickle

        faiss = _get_faiss()
        self._index = faiss.read_index(str(path))

        meta_path = path.with_suffix(".meta.pkl")
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self._id_to_chunk = meta["id_to_chunk"]
            self._chunk_to_id = meta["chunk_to_id"]
            self._next_id = meta["next_id"]
            self._deleted = meta.get("deleted", set())
            self._dim = meta.get("dim", self._dim)
        self._dirty = False

    def save_if_dirty(self):
        if self._dirty:
            self.save()

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal - len(self._deleted) if self._index else 0


# ---------------------------------------------------------------------------
# Singleton store
# ---------------------------------------------------------------------------

_store: Optional[VectorStore] = None


def get_vector_store(dimension: int = 384, index_path: Optional[Path] = None) -> VectorStore:
    """Return the global VectorStore, loading from disk if available."""
    global _store
    if _store is None:
        from kb.config.settings import get_settings
        settings = get_settings()
        _dim = dimension or settings.embeddings.dimension
        _path = index_path or settings.faiss_path
        _store = VectorStore(dimension=_dim, index_path=_path)
    return _store
