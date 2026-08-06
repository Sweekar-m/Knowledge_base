"""Antigravity Super-Prompt Engine — builds comprehensive prompts for Antigravity AI agent."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from kb.database.db import get_session
from kb.database.models import File, Project
from kb.prompt_engine.ranker import rank_and_trim
from kb.utils.tokenizer import count_tokens


_ANTIGRAVITY_HEADER = """# 🚀 ANTIGRAVITY TASK INSTRUCTIONS & CODEBASE CONTEXT

You are **Antigravity**, an expert agentic AI coding assistant pair programming with the developer.
Your goal is to solve the following issue/task with high accuracy, clean architecture, and minimal regression risk.
Below is the curated, up-to-date context from the project knowledge base including relevant code snippets, file structure, git state, and architecture notes.
"""

_ANTIGRAVITY_PROTOCOL = """
---

## ⚡ RESOLUTION PROTOCOL & EXPECTED DELIVERABLES

When answering, please follow these steps:
1. **Root Cause & Scope Analysis**: Briefly explain what component/file needs to change and why.
2. **Implementation Strategy**: Outline the exact changes to be made.
3. **Precise Code Modifications**: Provide complete, production-ready code edits or diffs with exact file paths.
4. **Verification Plan**: List specific tests or commands to verify that the fix works correctly without regressions.
"""


def build_antigravity_prompt(
    issue_query: str,
    project_id: int,
    project_path: Path,
    top_k: int = 15,
    git_context_str: Optional[str] = None,
    chat_history: Optional[List[dict]] = None,
) -> str:
    """
    Build a super-detailed prompt optimized for Antigravity to solve a specific issue.

    Args:
        issue_query: Description of the issue or task to solve.
        project_id: Database project ID.
        project_path: Path to the project root.
        top_k: Number of relevant code chunks to retrieve.
        git_context_str: Optional formatted git status string.
        chat_history: Optional recent chat history context.

    Returns:
        Formatted Markdown prompt string ready to give to Antigravity.
    """
    from kb.memory.long_term import format_notes_for_context, get_relevant_notes, get_tasks
    from kb.retriever.hybrid import hybrid_search
    from kb.git.reader import read_git_state

    sections = [_ANTIGRAVITY_HEADER]

    # 1. Issue Objective
    query_str = issue_query.strip() or "Resolve codebase issue based on recent chat context."
    sections.append(f"## 🎯 ISSUE OBJECTIVE / TASK\n{query_str}")

    # 2. Project Environment & File Structure
    with get_session() as session:
        project = session.query(Project).get(project_id)
        if project:
            lang_stats = project.get_language_stats()
            lang_summary = ", ".join(
                f"{lang}: {count}" for lang, count in
                sorted(lang_stats.items(), key=lambda x: -x[1])[:8]
            )
            proj_summary = (
                f"- **Project Name**: {project.name}\n"
                f"- **Root Path**: {project.path}\n"
                f"- **Indexed Files**: {project.file_count}\n"
                f"- **Primary Languages**: {lang_summary or 'N/A'}\n"
                f"- **Last Scanned**: {project.last_scanned or 'Just now'}"
            )

            files = session.query(File).filter_by(project_id=project_id).order_by(File.relative_path).all()
            file_lines = []
            for f in files[:80]:
                summary_str = f" — {f.summary}" if f.summary else ""
                file_lines.append(f"  • `{f.relative_path}` ({f.language or 'file'}){summary_str}")
            if len(files) > 80:
                file_lines.append(f"  ... and {len(files) - 80} more files.")

            file_map_str = "\n".join(file_lines)
            proj_summary += f"\n\n### 📁 Key Project Files:\n{file_map_str}"
        else:
            proj_summary = f"- **Project Path**: {project_path}"

    sections.append(f"## 🏢 PROJECT OVERVIEW\n{proj_summary}")

    # 3. Relevant Code Context via Hybrid Search
    try:
        raw_results = hybrid_search(query_str, top_k=top_k * 2, project_id=project_id)
        trimmed = rank_and_trim(raw_results, max_tokens=10000)

        if trimmed:
            code_blocks = []
            for chunk_id, score, content, file_path in trimmed:
                code_blocks.append(f"### File: `{file_path}` (relevance score: {score:.3f})\n```\n{content}\n```")
            sections.append("## 🔍 RELEVANT SOURCE CODE CONTEXT\n" + "\n\n".join(code_blocks))
        else:
            sections.append("## 🔍 RELEVANT SOURCE CODE CONTEXT\n*No specific code chunks matched the query directly.*")
    except Exception as e:
        sections.append(f"## 🔍 RELEVANT SOURCE CODE CONTEXT\n*Retrieval notice: {e}*")

    # 4. Architecture Notes & Tasks
    try:
        notes = get_relevant_notes(query_str, project_id, top_k=5)
        tasks = get_tasks(project_id, status="pending")
        memory_parts = []
        if notes:
            memory_parts.append("### Architecture Notes:\n" + format_notes_for_context(notes))
        if tasks:
            task_lines = [f"- [ ] #{t.id} ({t.priority}): {t.title}" for t in tasks[:5]]
            memory_parts.append("### Pending Project Tasks:\n" + "\n".join(task_lines))
        if memory_parts:
            sections.append("## 📝 ARCHITECTURE & TASKS MEMORY\n" + "\n\n".join(memory_parts))
    except Exception:
        pass

    # 5. Git Context
    if not git_context_str:
        try:
            state = read_git_state(project_path)
            if state.is_git_repo:
                git_context_str = state.to_context_string()
        except Exception:
            pass

    if git_context_str:
        sections.append(f"## 🌿 GIT STATE & UNCOMMITTED CHANGES\n{git_context_str}")

    # 6. Chat History Context (if provided)
    if chat_history:
        history_lines = []
        for msg in chat_history[-6:]:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "").strip()
            if content:
                history_lines.append(f"**{role}**: {content[:300]}")
        if history_lines:
            sections.append("## 💬 RECENT CONVERSATION CONTEXT\n" + "\n\n".join(history_lines))

    # 7. Add Deliverables Protocol
    sections.append(_ANTIGRAVITY_PROTOCOL)

    return "\n\n".join(sections)


def copy_to_clipboard(text: str) -> bool:
    """
    Attempt to copy text to system clipboard.
    Supports pyperclip, Windows `clip`, macOS `pbcopy`, and Linux `xclip`.
    """
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        pass

    # Windows fallback
    if os.name == "nt":
        try:
            p = subprocess.Popen(["clip"], stdin=subprocess.PIPE, shell=True)
            p.communicate(input=text.encode("utf-16"))
            return p.returncode == 0
        except Exception:
            pass

    # macOS fallback
    if sys.platform == "darwin":
        try:
            p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            p.communicate(input=text.encode("utf-8"))
            return p.returncode == 0
        except Exception:
            pass

    # Linux fallback
    if sys.platform.startswith("linux"):
        for cmd in (["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]):
            try:
                p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                p.communicate(input=text.encode("utf-8"))
                return p.returncode == 0
            except Exception:
                continue

    return False
