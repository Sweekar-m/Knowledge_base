"""Keyword search using SQLite FTS5."""

from __future__ import annotations

from typing import List, Tuple

from sqlalchemy import text

from kb.database.db import get_session


def keyword_search(
    query: str,
    top_k: int = 10,
    project_id: Optional[int] = None,
) -> List[Tuple[int, float, str, str]]:
    """
    Full-text search using SQLite FTS5.
    
    Returns:
        List of (chunk_id, score, content, file_path) tuples.
    """
    from kb.database.models import Chunk, File

    # Sanitize query for FTS5 (escape special chars)
    safe_query = _sanitize_fts_query(query)
    if not safe_query:
        return []

    with get_session() as session:
        # BM25 score — FTS5 returns negative rank (lower = better match)
        try:
            sql = """
                SELECT c.id, -rank AS score, c.content, f.relative_path
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.rowid = c.id
                JOIN files f ON c.file_id = f.id
                WHERE chunks_fts MATCH :query
            """
            params = {"query": safe_query, "limit": top_k}
            if project_id is not None:
                sql += " AND f.project_id = :project_id"
                params["project_id"] = project_id
            sql += " ORDER BY rank LIMIT :limit"

            rows = session.execute(text(sql), params).fetchall()
        except Exception:
            # Fallback to LIKE search if FTS fails
            like_query = f"%{query[:100]}%"
            sql = """
                SELECT c.id, 1.0 AS score, c.content, f.relative_path
                FROM chunks c
                JOIN files f ON c.file_id = f.id
                WHERE c.content LIKE :query
            """
            params = {"query": like_query, "limit": top_k}
            if project_id is not None:
                sql += " AND f.project_id = :project_id"
                params["project_id"] = project_id
            sql += " LIMIT :limit"

            rows = session.execute(text(sql), params).fetchall()

        return [(row[0], float(row[1]), row[2], row[3]) for row in rows]


def _sanitize_fts_query(query: str) -> str:
    """
    Make a query safe for FTS5 MATCH.
    Strips special characters that would cause FTS5 syntax errors.
    """
    # Remove FTS5 special chars
    for char in ['"', "'", "(", ")", "*", "^", "-", "+"]:
        query = query.replace(char, " ")
    # Collapse whitespace and strip
    parts = [p.strip() for p in query.split() if p.strip() and len(p.strip()) > 1]
    if not parts:
        return ""
    # Join as OR search for broader recall
    return " OR ".join(parts[:10])
