"""Token counting helpers using tiktoken."""

from __future__ import annotations

import tiktoken

_encoder = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        # cl100k_base works well for most modern LLMs
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """Return approximate token count for the given text."""
    if not text:
        return 0
    return len(_get_encoder().encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens tokens."""
    enc = _get_encoder()
    tokens = enc.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])
