"""Semantic retrieval via FAISS vector similarity search."""

from __future__ import annotations

from typing import List, Tuple

from kb.database.db import get_session
from kb.database.models import Chunk, File


def semantic_search(
    query: str,
    top_k: int = 10,
    project_id: Optional[int] = None,
) -> List[Tuple[int, float, str, str]]:
    """
    Embed the query and find the top_k most similar chunks.
    
    Returns:
        List of (chunk_id, score, content, file_path) tuples.
    """
    from kb.embeddings import get_embedder
    from kb.embeddings.vector_store import get_vector_store

    embedder = get_embedder()
    store = get_vector_store()

    query_vec = embedder.embed_one(query)
    fetch_k = top_k * 10 if project_id is not None else top_k
    results = store.search(query_vec, top_k=fetch_k)

    if not results:
        return []

    chunk_ids = [r[0] for r in results]
    scores = {r[0]: r[1] for r in results}

    with get_session() as session:
        q = session.query(Chunk).join(File).filter(Chunk.id.in_(chunk_ids))
        if project_id is not None:
            q = q.filter(File.project_id == project_id)
        chunks = q.all()
        chunk_map = {c.id: c for c in chunks}

        output = []
        for chunk_id in chunk_ids:
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                continue
            file_path = ""
            if chunk.file:
                file_path = chunk.file.relative_path
            output.append((
                chunk_id,
                scores[chunk_id],
                chunk.content,
                file_path,
            ))
            if len(output) >= top_k:
                break

    return output
