"""Fast file hashing with xxhash."""

from __future__ import annotations

from pathlib import Path

import xxhash


def hash_file(path: Path, chunk_size: int = 65536) -> str:
    """Return xxh64 hex digest for a file. Reads in chunks for large files."""
    h = xxhash.xxh64()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
    except (OSError, PermissionError):
        return ""
    return h.hexdigest()


def hash_string(s: str) -> str:
    """Return xxh64 hex digest for a string."""
    return xxhash.xxh64(s.encode()).hexdigest()
