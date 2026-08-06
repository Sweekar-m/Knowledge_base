"""Automatic incremental scanner helper — ensures index is up to date before operations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from kb.config.settings import get_settings
from kb.indexer.scanner import FileDiff, scan_directory
from kb.indexer.worker import delete_files_from_index, run_indexing
from kb.utils.display import console, info, success


def auto_scan_if_needed(
    root: Path,
    silent: bool = False,
    workers: Optional[int] = None,
) -> FileDiff:
    """
    Check if files in `root` have changed since last scan and perform an
    incremental scan automatically if modifications, additions, or deletions exist.

    Args:
        root: Project directory path.
        silent: If True, suppress progress prints unless indexing occurs.
        workers: Number of worker threads for indexing.

    Returns:
        FileDiff object containing change details.
    """
    from kb.database.db import get_session, init_db
    from kb.database.models import File, Project
    from collections import Counter
    from datetime import datetime

    init_db()

    abs_path = str(root.resolve())
    with get_session() as session:
        project = session.query(Project).filter_by(path=abs_path).first()
        if project is None:
            name = root.resolve().name
            project = Project(name=name, path=abs_path)
            session.add(project)
            session.flush()
        project_id = project.id

        files = session.query(File).filter_by(project_id=project_id).all()
        existing_hashes = {f.relative_path: f.file_hash for f in files}

    settings = get_settings()
    n_workers = workers or settings.indexer.workers

    diff = scan_directory(
        root=root,
        existing_hashes=existing_hashes,
        max_file_size_mb=settings.indexer.max_file_size_mb,
    )

    to_index = diff.new_files + diff.modified_files
    has_changes = bool(to_index or diff.deleted_paths)

    if not has_changes:
        if not silent:
            console.print("[dim][>] Knowledge base is up to date.[/dim]")
        return diff

    if not silent:
        info(
            f"Auto-scan detected changes ([green]+{len(diff.new_files)}[/green] "
            f"[yellow]~{len(diff.modified_files)}[/yellow] [red]-{len(diff.deleted_paths)}[/red]). Update in progress..."
        )

    if diff.deleted_paths:
        delete_files_from_index(diff.deleted_paths, project_id)

    if to_index:
        run_indexing(
            files=to_index,
            root=root,
            project_id=project_id,
            workers=n_workers,
        )

    # Recalculate stats
    with get_session() as session:
        files = session.query(File).filter_by(project_id=project_id).all()
        lang_counts = Counter(f.language for f in files if f.language)
        proj = session.query(Project).get(project_id)
        if proj:
            proj.file_count = len(files)
            proj.set_language_stats(dict(lang_counts))
            proj.last_scanned = datetime.utcnow()

    if not silent:
        success(f"Auto-scan complete! Processed {len(to_index)} file(s).")

    return diff
