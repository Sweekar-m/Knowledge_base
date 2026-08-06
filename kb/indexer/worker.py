"""Background indexing worker using ThreadPoolExecutor."""

from __future__ import annotations

import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from kb.config.settings import get_settings
from kb.database.db import get_session, init_db
from kb.database.models import Chunk, File, Project
from kb.embeddings import get_embedder
from kb.embeddings.vector_store import get_vector_store
from kb.parser import parse_file
from kb.parser.chunker import chunk_file
from kb.utils.display import console, make_progress
from kb.utils.hashing import hash_file


def _index_single_file(
    file_path: Path,
    root: Path,
    project_id: int,
) -> Optional[dict]:
    """
    Parse, chunk, and embed a single file.
    Returns a dict with results or None on failure.
    """
    settings = get_settings()
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    # Parse
    parsed = parse_file(file_path, content)

    # Chunk
    chunks = chunk_file(
        content=content,
        file_path=str(file_path.relative_to(root)),
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
    )

    file_hash = hash_file(file_path)
    try:
        rel_path = str(file_path.relative_to(root))
    except ValueError:
        rel_path = str(file_path)

    return {
        "file_path": file_path,
        "rel_path": rel_path,
        "project_id": project_id,
        "parsed": parsed,
        "chunks": chunks,
        "file_hash": file_hash,
        "size_bytes": file_path.stat().st_size,
        "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime),
    }


def _save_file_to_db(result: dict) -> List[int]:
    """
    Upsert a file and its chunks to the database.
    Returns list of (chunk_id, chunk_content) tuples for embedding.
    """
    parsed = result["parsed"]
    chunks = result["chunks"]

    with get_session() as session:
        # Upsert File record
        file_rec = session.query(File).filter_by(
            project_id=result["project_id"],
            path=str(result["file_path"]),
        ).first()

        if file_rec is None:
            file_rec = File(
                project_id=result["project_id"],
                path=str(result["file_path"]),
                relative_path=result["rel_path"],
            )
            session.add(file_rec)
            session.flush()
        else:
            # Delete old chunks
            session.query(Chunk).filter_by(file_id=file_rec.id).delete()

        file_rec.language = parsed.language
        file_rec.file_hash = result["file_hash"]
        file_rec.size_bytes = result["size_bytes"]
        file_rec.last_indexed = datetime.utcnow()
        file_rec.last_modified = result["last_modified"]
        file_rec.imports_json = json.dumps(parsed.imports)
        file_rec.exports_json = json.dumps(parsed.exports)
        file_rec.classes_json = json.dumps(parsed.classes)
        file_rec.functions_json = json.dumps(parsed.functions)
        file_rec.interfaces_json = json.dumps(parsed.interfaces)
        file_rec.enums_json = json.dumps(parsed.enums)
        file_rec.todos_json = json.dumps(parsed.todos)
        file_rec.summary = parsed.summary

        # Insert new chunks
        chunk_ids = []
        for tc in chunks:
            chunk = Chunk(
                file_id=file_rec.id,
                content=tc.content,
                start_line=tc.start_line,
                end_line=tc.end_line,
                chunk_index=tc.chunk_index,
                token_count=tc.token_count,
            )
            session.add(chunk)
            session.flush()
            chunk_ids.append((chunk.id, tc.content))

        return chunk_ids


def _embed_chunks(chunk_data: List[tuple]) -> List[tuple]:
    """
    Embed a list of (chunk_id, content) pairs.
    Returns list of (chunk_id, vector) tuples.
    """
    if not chunk_data:
        return []

    embedder = get_embedder()
    ids = [c[0] for c in chunk_data]
    texts = [c[1] for c in chunk_data]
    vectors = embedder.embed(texts)
    return list(zip(ids, vectors))


def run_indexing(
    files: List[Path],
    root: Path,
    project_id: int,
    workers: int = 4,
    progress_callback: Optional[Callable[[int], None]] = None,
):
    """
    Index a list of files: parse → chunk → save to DB → embed → upsert FAISS.
    
    Args:
        files: Files to index.
        root: Project root (for relative paths).
        project_id: DB project ID.
        workers: Number of parse threads.
        progress_callback: Called with number of files completed.
    """
    settings = get_settings()
    store = get_vector_store(
        dimension=settings.embeddings.dimension,
        index_path=settings.faiss_path,
    )

    all_chunk_data: List[tuple] = []  # (chunk_id, content)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_index_single_file, f, root, project_id): f
            for f in files
        }

        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                if result:
                    chunk_ids = _save_file_to_db(result)
                    all_chunk_data.extend(chunk_ids)
            except Exception as e:
                file = futures[future]
                console.print(f"[error]Error indexing {file}: {e}[/error]")

            if progress_callback:
                progress_callback(i + 1)

    # Batch embed all new/updated chunks
    if all_chunk_data:
        embedded = _embed_chunks(all_chunk_data)
        import numpy as np
        ids = [e[0] for e in embedded]
        vectors = np.stack([e[1] for e in embedded])
        store.upsert_batch(ids, vectors)

        # Update vector_id in DB
        with get_session() as session:
            for chunk_id, vec in embedded:
                chunk = session.query(Chunk).get(chunk_id)
                if chunk:
                    chunk.vector_id = chunk_id   # use chunk_id as stable vector key

    store.save_if_dirty()


def delete_files_from_index(relative_paths: List[str], project_id: int):
    """Remove deleted files from DB and vector store."""
    store = get_vector_store()
    with get_session() as session:
        for rel_path in relative_paths:
            file_rec = session.query(File).filter_by(
                project_id=project_id,
                relative_path=rel_path,
            ).first()
            if file_rec:
                chunk_ids = [c.id for c in file_rec.chunks]
                store.delete(chunk_ids)
                session.delete(file_rec)

    store.save_if_dirty()
