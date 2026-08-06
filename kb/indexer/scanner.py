"""File system scanner with incremental change detection."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from kb.config.settings import get_settings
from kb.utils.hashing import hash_file

# Files to always include regardless of extension
_ALWAYS_INCLUDE = {
    "package.json", "README.md", "Dockerfile",
    "docker-compose.yml", ".env.example", "Makefile",
    "CMakeLists.txt", "Cargo.toml", "go.mod", "go.sum",
    "requirements.txt", "setup.py", "setup.cfg",
    "pyproject.toml", ".gitignore", ".dockerignore",
}


@dataclass
class FileDiff:
    """Result of scanning a directory against the existing DB state."""
    new_files: List[Path] = field(default_factory=list)
    modified_files: List[Path] = field(default_factory=list)
    deleted_paths: List[str] = field(default_factory=list)    # relative paths
    unchanged_count: int = 0
    total_scanned: int = 0


def _is_ignored(path: Path, ignored_dirs: Set[str], root: Path) -> bool:
    """Return True if path is inside any ignored directory."""
    try:
        rel = path.relative_to(root)
        parts = rel.parts
        for part in parts[:-1]:  # directory parts only
            if part in ignored_dirs or part.startswith("."):
                # allow .env.example etc at root
                if len(parts) == 1:
                    continue
                return True
        return False
    except ValueError:
        return False


def _should_include(path: Path, supported_exts: Set[str], ignored_exts: Set[str]) -> bool:
    """Return True if this file should be indexed."""
    name = path.name
    suffix = path.suffix.lower()

    if name in _ALWAYS_INCLUDE:
        return True
    if suffix in ignored_exts:
        return False
    if suffix in supported_exts:
        return True
    return False


def scan_directory(
    root: Path,
    existing_hashes: Dict[str, str],  # relative_path -> hash
    max_file_size_mb: float = 5.0,
) -> FileDiff:
    """
    Walk root directory and compute a FileDiff against existing_hashes.
    
    Args:
        root: Directory to scan.
        existing_hashes: Dict mapping relative file paths to their last-known hash.
        max_file_size_mb: Skip files larger than this.
    
    Returns:
        FileDiff with new, modified, deleted lists.
    """
    settings = get_settings()
    ignored_dirs: Set[str] = set(settings.ignored_dirs)
    supported_exts: Set[str] = set(settings.supported_extensions)
    ignored_exts: Set[str] = set(settings.ignored_extensions)
    max_bytes = int(max_file_size_mb * 1024 * 1024)

    diff = FileDiff()
    seen_paths: Set[str] = set()

    for dirpath, dirnames, filenames in os.walk(root):
        dir_path = Path(dirpath)

        # Prune ignored directories in-place (prevents os.walk from descending)
        dirnames[:] = [
            d for d in dirnames
            if d not in ignored_dirs and not d.startswith(".")
            or d == ".github"   # allow .github
        ]

        for filename in filenames:
            file_path = dir_path / filename

            if _is_ignored(file_path, ignored_dirs, root):
                continue
            if not _should_include(file_path, supported_exts, ignored_exts):
                continue

            try:
                file_size = file_path.stat().st_size
            except OSError:
                continue

            if file_size > max_bytes:
                continue

            diff.total_scanned += 1
            try:
                rel_path = str(file_path.relative_to(root))
            except ValueError:
                rel_path = str(file_path)

            seen_paths.add(rel_path)
            new_hash = hash_file(file_path)

            if rel_path not in existing_hashes:
                diff.new_files.append(file_path)
            elif existing_hashes[rel_path] != new_hash:
                diff.modified_files.append(file_path)
            else:
                diff.unchanged_count += 1

    # Detect deleted files
    for rel_path in existing_hashes:
        if rel_path not in seen_paths:
            diff.deleted_paths.append(rel_path)

    return diff
