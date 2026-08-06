"""Prompt builder — assembles the complete optimised system prompt."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from kb.database.db import get_session
from kb.database.models import File, Project
from kb.prompt_engine.ranker import rank_and_trim, trim_string
from kb.utils.tokenizer import count_tokens


_SYSTEM_PERSONA = """You are an expert AI software engineer and system architect with deep knowledge of this codebase.
You have access to the complete project structure, conversation history, git state, and architectural decisions.

CRITICAL INSTRUCTIONS:
- You DO NOT have execution tools or function-calling capabilities. Do NOT output `<tool_call>`, `<invoke>`, `<function>`, `read_file`, or JSON/XML function call payloads under any circumstances.
- All relevant project files, structures, summaries, and source code are ALREADY retrieved and provided to you in the prompt sections below.
- Always answer the user's question directly, completely, and accurately in clear Markdown format.
- Ground your answers in the actual code and context provided. Cite file paths and line numbers when referencing code."""


def build_system_prompt(
    query: str,
    project_id: int,
    project_path: Path,
    top_k: int = 10,
    max_context_tokens: int = 12000,
    chat_history: Optional[List[dict]] = None,
    git_context_str: Optional[str] = None,
) -> Tuple[str, List[int]]:
    """
    Build the complete system prompt for a user query.
    
    Sections assembled (in priority order):
    1. Persona & project summary + file structure
    2. Architecture notes (long-term memory)
    3. Git state
    4. Relevant source code chunks
    5. Relevant past conversations
    
    Returns:
        (system_prompt_string, list_of_retrieved_chunk_ids)
    """
    from kb.memory.long_term import format_notes_for_context, get_relevant_notes
    from kb.memory.conversation import get_relevant_past_messages
    from kb.retriever.hybrid import hybrid_search

    sections = []
    retrieved_chunk_ids = []

    # ── 1. Persona + Project Summary & File Map ──────────────────────────────
    with get_session() as session:
        project = session.query(Project).get(project_id)
        if project:
            lang_stats = project.get_language_stats()
            lang_summary = ", ".join(
                f"{lang}: {count}" for lang, count in
                sorted(lang_stats.items(), key=lambda x: -x[1])[:8]
            )
            proj_summary = (
                f"Project: {project.name}\n"
                f"Path: {project.path}\n"
                f"Files indexed: {project.file_count}\n"
                f"Languages: {lang_summary or 'N/A'}\n"
                f"Last scanned: {project.last_scanned or 'never'}"
            )

            # List indexed files with summaries for complete file awareness
            files = session.query(File).filter_by(project_id=project_id).order_by(File.relative_path).all()
            file_lines = []
            for f in files[:100]:  # Cap at 100 files to stay within token budget
                summary_str = f" — {f.summary}" if f.summary else ""
                file_lines.append(f"  • {f.relative_path} ({f.language or 'file'}, {f.size_bytes}B){summary_str}")
            if len(files) > 100:
                file_lines.append(f"  ... and {len(files) - 100} more files.")

            file_map_str = "\n".join(file_lines)
            proj_summary += f"\n\n[PROJECT FILE STRUCTURE & SUMMARIES]\n{file_map_str}"
        else:
            proj_summary = f"Project at: {project_path}"

    sections.append(f"[PROJECT SUMMARY]\n{proj_summary}")

    # ── 2. Architecture Notes ────────────────────────────────────────────────
    try:
        notes = get_relevant_notes(query, project_id, top_k=4)
        if notes:
            notes_text = format_notes_for_context(notes)
            sections.append(f"[ARCHITECTURE NOTES]\n{notes_text}")
    except Exception:
        pass

    # ── 3. Git State ─────────────────────────────────────────────────────────
    if git_context_str:
        sections.append(f"[GIT STATE]\n{git_context_str}")

    # ── 4. Relevant Source Files ─────────────────────────────────────────────
    try:
        raw_results = hybrid_search(query, top_k=top_k * 2, project_id=project_id)
        trimmed = rank_and_trim(raw_results, max_tokens=max_context_tokens - 2000)

        if trimmed:
            code_blocks = []
            for chunk_id, score, content, file_path in trimmed:
                retrieved_chunk_ids.append(chunk_id)
                code_blocks.append(f"--- {file_path} ---\n{content}")
            sections.append("[RELEVANT SOURCE FILES]\n" + "\n\n".join(code_blocks))
    except Exception:
        pass

    # ── 5. Relevant Past Conversations ───────────────────────────────────────
    try:
        past = get_relevant_past_messages(query, project_id, limit=3)
        if past:
            conv_lines = []
            for user_q, asst_a in past:
                conv_lines.append(f"Q: {user_q}\nA: {asst_a}")
            sections.append("[RELEVANT PAST CONVERSATIONS]\n" + "\n\n---\n\n".join(conv_lines))
    except Exception:
        pass

    # ── Assemble system prompt ───────────────────────────────────────────────
    body = "\n\n".join(sections)
    system_prompt = f"{_SYSTEM_PERSONA}\n\n{body}"

    # Final token trim to stay within budget
    total = count_tokens(system_prompt)
    if total > max_context_tokens:
        system_prompt = trim_string(system_prompt, max_context_tokens)

    return system_prompt, retrieved_chunk_ids
