"""NVIDIA Nemotron streaming LLM client — exact API as specified."""

from __future__ import annotations

import re
from typing import Generator, List, Optional, Tuple

from openai import OpenAI

from kb.config.settings import get_settings
from kb.utils.display import console

_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)

_TOOL_OPEN = re.compile(r"<(?:tool_call|invoke|function)[^>]*>", re.IGNORECASE)
_TOOL_CLOSE = re.compile(r"<\/(?:tool_call|invoke|function|parameter)[^>]*>", re.IGNORECASE)


def _get_client() -> OpenAI:
    settings = get_settings()
    cfg = settings.nvidia
    return OpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
    )


from rich.text import Text


def _safe_print(text: str, style: str = ""):
    t = Text(text, style=style) if style else text
    try:
        console.print(t, end="", markup=False)
    except Exception:
        safe_text = text.encode("ascii", errors="replace").decode("ascii")
        t_safe = Text(safe_text, style=style) if style else safe_text
        console.print(t_safe, end="", markup=False)


_UNK_RE = re.compile(r"<unk>|<\|.*?\|>|\[UNK\]", re.IGNORECASE)


def stream_response(
    messages: List[dict],
    status_context: Optional[object] = None,
) -> Tuple[str, str]:
    """
    Stream a response from the NVIDIA Nemotron API.
    
    Prints reasoning tokens in dim magenta and answer tokens in white.
    Suppresses any raw XML tool call blocks and <unk> special tokens emitted by the model.
    
    Args:
        messages: OpenAI-format message list (system + user + history).
        status_context: Optional Rich Status object to stop on first token.
    
    Returns:
        (thinking_text, answer_text) tuple.
    """
    settings = get_settings()
    cfg = settings.nvidia
    client = _get_client()

    completion = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_tokens,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": cfg.enable_thinking,
            },
            "reasoning_budget": cfg.reasoning_budget,
        },
        stream=True,
    )

    thinking_buffer = []
    answer_buffer = []
    in_think_block = False
    in_tool_block = False
    printed_answer_header = False
    unk_counter = 0

    for chunk in completion:
        choice = chunk.choices[0] if chunk.choices else None
        if choice is None:
            continue

        delta = choice.delta
        content = delta.content or ""
        if not content:
            continue

        # Filter out <unk> and special control tokens
        clean_content = _UNK_RE.sub("", content)
        if not clean_content and ("<unk>" in content.lower() or "[unk]" in content.lower()):
            unk_counter += 1
            if unk_counter >= 3:
                # Stop streaming if model enters repeating UNK loop
                break
            continue
        else:
            unk_counter = 0
            content = clean_content

        if not content:
            continue

        # Suppress raw XML tool calls if model attempts function calls
        if _TOOL_OPEN.search(content):
            in_tool_block = True

        if in_tool_block:
            if _TOOL_CLOSE.search(content):
                in_tool_block = False
            continue

        # Stop loading spinner when first valid token arrives
        if status_context is not None:
            try:
                status_context.stop()
            except Exception:
                pass
            status_context = None
            console.print()

        # Detect <think> block boundaries
        if _THINK_OPEN.search(content):
            in_think_block = True
            # Print reasoning header once
            _safe_print("\n>> Reasoning...\n", style="dim")

        if in_think_block:
            # Strip XML tags before printing
            display = _THINK_OPEN.sub("", content)
            display = _THINK_CLOSE.sub("", display)
            thinking_buffer.append(content)
            if display:
                _safe_print(display, style="thinking")

            if _THINK_CLOSE.search(content):
                in_think_block = False
                console.print()  # newline after reasoning
        else:
            answer_buffer.append(content)
            if not printed_answer_header:
                _safe_print("\n>> Answer\n\n", style="accent")
                printed_answer_header = True
            _safe_print(content, style="answer")

    console.print()  # final newline

    thinking_text = "".join(thinking_buffer)
    answer_text = "".join(answer_buffer)
    return thinking_text, answer_text


def simple_completion(prompt: str, system: str = "") -> str:
    """
    Non-streaming completion for internal use (e.g. generating summaries).
    Returns only the answer text.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    settings = get_settings()
    cfg = settings.nvidia
    client = _get_client()

    completion = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=2048,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
        stream=False,
    )

    content = completion.choices[0].message.content or ""
    # Strip think tags if present
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return content.strip()
