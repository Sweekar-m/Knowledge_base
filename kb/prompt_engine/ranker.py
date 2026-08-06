"""Token-budget ranker — trims context to fit within the allowed token window."""

from __future__ import annotations

from typing import List, Tuple

from kb.utils.tokenizer import count_tokens, truncate_to_tokens


def rank_and_trim(
    chunks: List[Tuple[int, float, str, str]],
    max_tokens: int = 12000,
    reserve_for_system: int = 500,
) -> List[Tuple[int, float, str, str]]:
    """
    Given ranked chunks (chunk_id, score, content, file_path),
    return only as many as fit within max_tokens.
    
    Already sorted by descending relevance — trim from the tail.
    """
    budget = max_tokens - reserve_for_system
    kept = []
    used = 0

    for chunk in chunks:
        tokens = count_tokens(chunk[2])
        if used + tokens > budget:
            # Try to fit a truncated version of this chunk
            remaining = budget - used
            if remaining > 100:  # only bother if >100 tokens left
                truncated = truncate_to_tokens(chunk[2], remaining)
                kept.append((chunk[0], chunk[1], truncated, chunk[3]))
                used += remaining
            break
        kept.append(chunk)
        used += tokens

    return kept


def trim_string(text: str, max_tokens: int) -> str:
    """Trim a string to a maximum token count."""
    return truncate_to_tokens(text, max_tokens)
