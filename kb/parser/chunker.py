"""Text chunker that splits file content into overlapping token-bounded chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from kb.utils.tokenizer import count_tokens


@dataclass
class TextChunk:
    content: str
    start_line: int
    end_line: int
    chunk_index: int
    token_count: int


def chunk_text(
    content: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[TextChunk]:
    """
    Split content into overlapping chunks that respect line boundaries.

    Strategy:
    1. Split into lines.
    2. Accumulate lines until we hit chunk_size tokens.
    3. Back up by overlap tokens for the next chunk.
    """
    if not content.strip():
        return []

    lines = content.splitlines(keepends=True)
    if not lines:
        return []

    chunks: List[TextChunk] = []
    chunk_index = 0
    start_line = 0

    while start_line < len(lines):
        # Accumulate lines up to chunk_size
        current_lines = []
        current_tokens = 0
        i = start_line

        while i < len(lines):
            line = lines[i]
            line_tokens = count_tokens(line)
            if current_tokens + line_tokens > chunk_size and current_lines:
                break
            current_lines.append(line)
            current_tokens += line_tokens
            i += 1

        if not current_lines:
            # Single line exceeds chunk_size; take it anyway to avoid infinite loop
            current_lines = [lines[start_line]]
            current_tokens = count_tokens(current_lines[0])
            i = start_line + 1

        end_line = start_line + len(current_lines) - 1
        content_str = "".join(current_lines)

        chunks.append(TextChunk(
            content=content_str,
            start_line=start_line + 1,   # 1-indexed
            end_line=end_line + 1,
            chunk_index=chunk_index,
            token_count=current_tokens,
        ))
        chunk_index += 1

        # Move forward, backing up by overlap
        if i >= len(lines):
            break

        # Calculate overlap lines
        overlap_tokens = 0
        overlap_lines = 0
        for line in reversed(current_lines):
            overlap_tokens += count_tokens(line)
            overlap_lines += 1
            if overlap_tokens >= chunk_overlap:
                break

        start_line = max(start_line + 1, i - overlap_lines)

    return chunks


def chunk_file(
    content: str,
    file_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[TextChunk]:
    """Chunk a file, prepending the file path to the first chunk for context."""
    chunks = chunk_text(content, chunk_size, chunk_overlap)
    if chunks:
        # Prepend file path to first chunk so retrieval results show provenance
        chunks[0].content = f"# {file_path}\n" + chunks[0].content
    return chunks
