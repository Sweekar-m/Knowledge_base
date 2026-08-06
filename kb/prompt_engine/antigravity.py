"""Antigravity Super-Prompt Engine — builds comprehensive, LLM-analyzed prompts for Antigravity AI agent."""

from __future__ import annotations

import os
import re
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

_EXCLUDED_PATTERNS = re.compile(
    r"(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|Cargo\.lock|go\.sum|\.min\.js|\.min\.css|\.map$)",
    re.IGNORECASE,
)


def _expand_query(query: str) -> List[str]:
    q_lower = query.lower()
    queries = [query]
    if any(w in q_lower for w in ("security", "vulnerability", "auth", "login", "secret", "token", "jwt", "permission")):
        queries.append("auth login JWT secret token encryption middleware permission sanitize authorization apiResponse")
    if any(w in q_lower for w in ("database", "db", "model", "query", "sql", "orm")):
        queries.append("database schema model query ORM migration table SQL session entity")
    if any(w in q_lower for w in ("api", "route", "endpoint", "controller")):
        queries.append("api route endpoint controller handler request response middleware REST")
    if any(w in q_lower for w in ("ui", "component", "editor", "style", "page")):
        queries.append("component props state JSX TSX render handler form input")
    return queries


def _synthesize_llm_analysis(
    issue_query: str,
    file_map_text: str,
    code_context_text: str,
) -> str:
    """Use Nemotron LLM to synthesize a deep architectural & security analysis for Antigravity."""
    from kb.llm.nvidia import simple_completion

    system_prompt = (
        "You are a senior principal software engineer and security architect. "
        "Your task is to analyze the user's issue request against the provided project file structure and source code snippets, "
        "and generate a clear, highly technical analysis and action plan specifically tailored for Antigravity AI."
    )

    user_prompt = f"""[USER TASK / ISSUE]
{issue_query}

[PROJECT FILE MAP]
{file_map_text[:3000]}

[RETRIEVED SOURCE CODE SNIPPETS]
{code_context_text[:6000]}

---

Based on the above context, provide a technical breakdown for Antigravity:
1. **Target Subsystem & Root Cause Analysis**: What specific files/modules are directly involved or vulnerable?
2. **Key Security & Architecture Concerns**: What specific risks or code patterns need attention?
3. **Recommended Execution Steps**: Step-by-step instructions for Antigravity to solve this issue cleanly.

Keep your output direct, concise, and structured in Markdown format.
"""

    try:
        return simple_completion(prompt=user_prompt, system=system_prompt)
    except Exception as e:
        return f"*Automated LLM analysis unavailable ({e}). Refer to retrieved source code snippets below.*"


def build_antigravity_prompt(
    issue_query: str,
    project_id: int,
    project_path: Path,
    top_k: int = 15,
    git_context_str: Optional[str] = None,
    chat_history: Optional[List[dict]] = None,
) -> str:
    """
    Build a super-detailed, LLM-analyzed prompt optimized for Antigravity to solve a specific issue.

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
    file_map_lines = []
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
            for f in files[:80]:
                summary_str = f" — {f.summary}" if f.summary else ""
                file_map_lines.append(f"  • `{f.relative_path}` ({f.language or 'file'}){summary_str}")
            if len(files) > 80:
                file_map_lines.append(f"  ... and {len(files) - 80} more files.")

            file_map_str = "\n".join(file_map_lines)
            proj_summary += f"\n\n### 📁 Key Project Files:\n{file_map_str}"
        else:
            proj_summary = f"- **Project Path**: {project_path}"

    sections.append(f"## 🏢 PROJECT OVERVIEW\n{proj_summary}")

    # 3. Hybrid Code Search with Filtering & Query Expansion
    raw_results = []
    search_queries = _expand_query(query_str)
    seen_chunk_ids = set()

    for q in search_queries:
        try:
            hits = hybrid_search(q, top_k=top_k * 2, project_id=project_id)
            for cid, score, content, file_path in hits:
                if cid in seen_chunk_ids:
                    continue
                # Exclude lockfiles and minified files
                if _EXCLUDED_PATTERNS.search(file_path):
                    continue
                seen_chunk_ids.add(cid)
                raw_results.append((cid, score, content, file_path))
        except Exception:
            pass

    trimmed = rank_and_trim(raw_results, max_tokens=10000)
    code_blocks = []
    if trimmed:
        for chunk_id, score, content, file_path in trimmed:
            code_blocks.append(f"### File: `{file_path}`\n```\n{content}\n```")
        code_context_text = "\n\n".join(code_blocks)
    else:
        code_context_text = "*No specific source files retrieved.*"

    sections.append("## 🔍 RELEVANT SOURCE CODE CONTEXT\n" + code_context_text)

    # 4. LLM Synthesis: Codebase & Task Analysis for Antigravity
    ai_analysis = _synthesize_llm_analysis(
        issue_query=query_str,
        file_map_text="\n".join(file_map_lines),
        code_context_text=code_context_text,
    )
    sections.insert(2, f"## 🧠 AI CODEBASE & TASK ANALYSIS\n{ai_analysis}")

    # 5. Architecture Notes & Tasks
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

    # 6. Git Context
    if not git_context_str:
        try:
            state = read_git_state(project_path)
            if state.is_git_repo:
                git_context_str = state.to_context_string()
        except Exception:
            pass

    if git_context_str:
        sections.append(f"## 🌿 GIT STATE & UNCOMMITTED CHANGES\n{git_context_str}")

    # 7. Chat History Context
    if chat_history:
        history_lines = []
        for msg in chat_history[-6:]:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "").strip()
            if content:
                history_lines.append(f"**{role}**: {content[:300]}")
        if history_lines:
            sections.append("## 💬 RECENT CONVERSATION CONTEXT\n" + "\n\n".join(history_lines))

    # 8. Resolution Protocol
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
