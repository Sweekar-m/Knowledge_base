"""Hybrid search using Reciprocal Rank Fusion (RRF) of semantic + keyword results."""

from __future__ import annotations

from typing import List, Tuple

from kb.retriever.keyword import keyword_search
from kb.retriever.semantic import semantic_search


def hybrid_search(
    query: str,
    top_k: int = 10,
    project_id: Optional[int] = None,
    semantic_weight: float = 0.6,
    keyword_weight: float = 0.4,
    rrf_k: int = 60,
) -> List[Tuple[int, float, str, str]]:
    """
    Merge semantic and keyword results using Reciprocal Rank Fusion (RRF).
    
    RRF score = Σ weight_i / (k + rank_i)
    
    Args:
        query: User query string.
        top_k: Number of results to return.
        project_id: Optional DB project ID for isolating results.
        semantic_weight: Weight for semantic results.
        keyword_weight: Weight for keyword results.
        rrf_k: RRF constant (typically 60).
    
    Returns:
        Deduplicated list of (chunk_id, score, content, file_path) sorted by RRF score.
    """
    fetch_k = max(top_k * 2, 20)

    semantic_results = semantic_search(query, top_k=fetch_k, project_id=project_id)
    keyword_results = keyword_search(query, top_k=fetch_k, project_id=project_id)

    # Map chunk_id -> (content, file_path)
    chunk_info: dict[int, tuple] = {}
    for cid, score, content, fp in semantic_results + keyword_results:
        chunk_info[cid] = (content, fp)

    # Compute RRF scores
    rrf_scores: dict[int, float] = {}

    for rank, (cid, score, content, fp) in enumerate(semantic_results):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + semantic_weight / (rrf_k + rank + 1)

    for rank, (cid, score, content, fp) in enumerate(keyword_results):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + keyword_weight / (rrf_k + rank + 1)

    # Sort by descending RRF score
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    results = []
    for cid in sorted_ids[:top_k]:
        content, fp = chunk_info.get(cid, ("", ""))
        results.append((cid, rrf_scores[cid], content, fp))

    return results
