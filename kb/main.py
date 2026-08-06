from __future__ import annotations

"""
kb -- Local AI Knowledge Base & Prompt Engine
CLI entry point using Typer.

Commands:
  kb scan     -- Index or re-index a project directory
  kb chat     -- Start interactive chat session
  kb search   -- Hybrid search without LLM
  kb status   -- Show index stats
  kb git      -- Show current git context
  kb memory   -- Show/add architecture notes and tasks
  kb rebuild  -- Force full rebuild
  kb stats    -- Detailed statistics
  kb watch    -- Live file watcher
"""
import os
import warnings
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
warnings.filterwarnings("ignore", category=UserWarning)


import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from kb.utils.display import (
    console,
    error,
    header,
    info,
    make_progress,
    print_panel,
    print_table,
    success,
    warning,
)

app = typer.Typer(
    name="kb",
    help="[KB] Local AI Knowledge Base & Prompt Engine",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_db():
    """Initialize database on first run."""
    from kb.database.db import init_db
    init_db()


def _get_or_create_project(path: Path) -> int:
    """Return project_id for the given path, creating if needed."""
    from kb.database.db import get_session
    from kb.database.models import Project

    abs_path = str(path.resolve())

    with get_session() as session:
        project = session.query(Project).filter_by(path=abs_path).first()
        if project is None:
            name = path.resolve().name
            project = Project(name=name, path=abs_path)
            session.add(project)
            session.flush()
        return project.id


def _get_project_id(path: Path) -> Optional[int]:
    """Return project_id for a path, or None if not indexed."""
    from kb.database.db import get_session
    from kb.database.models import Project

    abs_path = str(path.resolve())
    with get_session() as session:
        project = session.query(Project).filter_by(path=abs_path).first()
        return project.id if project else None


def _get_existing_hashes(project_id: int) -> dict:
    """Return {relative_path: hash} for all indexed files in a project."""
    from kb.database.db import get_session
    from kb.database.models import File

    with get_session() as session:
        files = session.query(File).filter_by(project_id=project_id).all()
        return {f.relative_path: f.file_hash for f in files}


def _update_project_stats(project_id: int, root: Path):
    """Recalculate and save file count + language breakdown."""
    from kb.database.db import get_session
    from kb.database.models import File, Project
    from collections import Counter

    with get_session() as session:
        files = session.query(File).filter_by(project_id=project_id).all()
        lang_counts = Counter(f.language for f in files if f.language)
        project = session.query(Project).get(project_id)
        if project:
            project.file_count = len(files)
            project.set_language_stats(dict(lang_counts))
            project.last_scanned = datetime.utcnow()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def scan(
    path: Optional[str] = typer.Argument(None, help="Project directory to scan (default: CWD)"),
    rebuild: bool = typer.Option(False, "--rebuild", "-r", help="Force full rebuild"),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Indexer threads"),
):
    """Scan and index a project directory (incremental by default)."""
    from kb.config.settings import get_settings
    from kb.indexer.scanner import scan_directory
    from kb.indexer.worker import delete_files_from_index, run_indexing

    settings = get_settings()
    root = Path(path or settings.project.default_path).resolve()

    if not root.exists():
        error(f"Path does not exist: {root}")
        raise typer.Exit(1)

    _ensure_db()
    project_id = _get_or_create_project(root)
    n_workers = workers or settings.indexer.workers

    header(f"Scanning: {root}")

    # Get existing hashes for incremental indexing
    existing_hashes = {} if rebuild else _get_existing_hashes(project_id)

    if rebuild and existing_hashes:
        warning("Rebuild requested — clearing existing index...")
        from kb.database.db import get_session
        from kb.database.models import File
        with get_session() as session:
            session.query(File).filter_by(project_id=project_id).delete()
        existing_hashes = {}

    info(f"Detecting changes (existing: {len(existing_hashes)} files)...")
    diff = scan_directory(
        root=root,
        existing_hashes=existing_hashes,
        max_file_size_mb=settings.indexer.max_file_size_mb,
    )

    console.print(
        f"  [green]New:[/green] {len(diff.new_files)}  "
        f"[yellow]Modified:[/yellow] {len(diff.modified_files)}  "
        f"[red]Deleted:[/red] {len(diff.deleted_paths)}  "
        f"[dim]Unchanged:[/dim] {diff.unchanged_count}"
    )

    # Delete removed files
    if diff.deleted_paths:
        info(f"Removing {len(diff.deleted_paths)} deleted files...")
        delete_files_from_index(diff.deleted_paths, project_id)

    # Index new + modified
    to_index = diff.new_files + diff.modified_files
    if not to_index:
        success("Everything up to date!")
        _update_project_stats(project_id, root)
        return

    info(f"Indexing {len(to_index)} files with {n_workers} workers...")

    with make_progress() as progress:
        task = progress.add_task("Indexing...", total=len(to_index))

        def _progress_cb(n: int):
            progress.update(task, completed=n)

        run_indexing(
            files=to_index,
            root=root,
            project_id=project_id,
            workers=n_workers,
            progress_callback=_progress_cb,
        )

    _update_project_stats(project_id, root)
    success(f"Done! Indexed {len(to_index)} file(s).")


@app.command()
def chat(
    path: Optional[str] = typer.Argument(None, help="Project directory (default: CWD)"),
    resume: Optional[int] = typer.Option(None, "--resume", help="Resume chat ID"),
):
    """Start an interactive AI chat session with full codebase context."""
    from kb.config.settings import get_settings
    from kb.database.db import get_session
    from kb.database.models import Chat, Project
    from kb.git.reader import read_git_state, save_git_state
    from kb.indexer.auto_scan import auto_scan_if_needed
    from kb.indexer.watcher import FileWatcher
    from kb.llm.nvidia import stream_response
    from kb.memory.conversation import (
        add_message, format_chat_history_for_context,
        get_chat_messages, new_chat, set_chat_title,
    )
    from kb.prompt_engine.builder import build_system_prompt

    settings = get_settings()
    root = Path(path or settings.project.default_path).resolve()

    _ensure_db()
    
    # Auto-scan changes before launching chat session
    auto_scan_if_needed(root, silent=False)

    project_id = _get_project_id(root)
    if project_id is None:
        project_id = _get_or_create_project(root)

    # Start real-time background watcher thread
    watcher: Optional[FileWatcher] = None
    try:
        watcher = FileWatcher(project_id=project_id, root=root)
        watcher.start()
    except Exception:
        watcher = None

    # Read git state
    git_state = read_git_state(root)
    git_context = git_state.to_context_string()
    git_json = git_state.to_json()
    if git_state.is_git_repo:
        save_git_state(git_state, project_id)

    # Create or resume chat
    if resume:
        chat_id = resume
        history = format_chat_history_for_context(chat_id, max_messages=10)
        info(f"Resumed chat #{chat_id} with {len(history)} messages")
    else:
        chat_id = new_chat(project_id, git_state_json=git_json)
        history = []

    header("KB Chat — Type 'exit' or Ctrl+C to quit")
    info(f"Project: {root.name} | Chat #{chat_id} | Branch: {git_state.branch}")
    console.print("[dim]Slash commands: /prompt [issue] | /note <text> | /task <text> | /search <query> | /history | exit[/dim]\n")

    first_message = True
    try:
        while True:
            try:
                user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Bye![/dim]")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break

            # --- Slash commands ---
            if user_input.startswith("/prompt") or user_input.startswith("/p "):
                issue_text = user_input[7:].strip() if user_input.startswith("/prompt") else user_input[3:].strip()
                _handle_prompt_command(issue_text, project_id, root, git_context, history)
                continue
            if user_input.startswith("/note "):
                _handle_note(user_input[6:].strip(), project_id)
                continue
            if user_input.startswith("/task "):
                _handle_task(user_input[6:].strip(), project_id)
                continue
            if user_input.startswith("/search "):
                _handle_inline_search(user_input[8:].strip(), project_id)
                continue
            if user_input == "/history":
                _show_chat_history(chat_id)
                continue

            # Save user message
            add_message(chat_id, "user", user_input)

            # Auto-title on first message
            if first_message:
                title = user_input[:100]
                set_chat_title(chat_id, title)
                first_message = False

            status = console.status("[bold cyan]Thinking & retrieving context...[/bold cyan]", spinner="dots")
            status.start()

            try:
                # Build optimised system prompt
                system_prompt, retrieved_ids = build_system_prompt(
                    query=user_input,
                    project_id=project_id,
                    project_path=root,
                    top_k=settings.retrieval.top_k,
                    max_context_tokens=settings.retrieval.max_context_tokens,
                    git_context_str=git_context,
                )

                # Assemble messages (system + history + current)
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(history[-8:])   # include recent history
                messages.append({"role": "user", "content": user_input})

                # Stream response (status will be stopped on first token)
                thinking, answer = stream_response(messages, status_context=status)
            finally:
                try:
                    status.stop()
                except Exception:
                    pass

            # Save assistant response
            add_message(
                chat_id, "assistant", answer,
                thinking=thinking,
                retrieved_chunk_ids=retrieved_ids,
            )

            # Update history for next turn
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": answer})
    finally:
        if watcher:
            watcher.stop()


def _handle_prompt_command(issue_text: str, project_id: int, root: Path, git_context_str: str, history: list):
    from kb.indexer.auto_scan import auto_scan_if_needed
    from kb.prompt_engine.antigravity import build_antigravity_prompt, copy_to_clipboard

    auto_scan_if_needed(root, silent=True)

    if not issue_text:
        user_msgs = [m["content"] for m in history if m["role"] == "user"]
        if user_msgs:
            issue_text = user_msgs[-1]
        else:
            issue_text = Prompt.ask("[bold cyan]Enter issue description for Antigravity prompt[/bold cyan]")

    status = console.status("[bold cyan]Generating Antigravity Super-Prompt...[/bold cyan]", spinner="dots")
    status.start()
    try:
        super_prompt = build_antigravity_prompt(
            issue_query=issue_text,
            project_id=project_id,
            project_path=root,
            top_k=15,
            git_context_str=git_context_str,
            chat_history=history,
        )
    finally:
        try:
            status.stop()
        except Exception:
            pass

    copied = copy_to_clipboard(super_prompt)
    copy_status = " [bold green](Copied to clipboard!)[/bold green]" if copied else " [dim](Clipboard copy unavailable)[/dim]"

    console.print(Panel(
        Markdown(super_prompt),
        title=f"🚀 [bold cyan]Antigravity Super-Prompt[/bold cyan]{copy_status}",
        border_style="cyan",
        expand=False,
    ))


def _handle_note(text: str, project_id: int):
    from kb.memory.long_term import save_note
    note_id = save_note(project_id, "general", text[:100], text)
    success(f"Note saved (#{note_id})")


def _handle_task(text: str, project_id: int):
    from kb.memory.long_term import save_task
    task_id = save_task(project_id, text)
    success(f"Task saved (#{task_id})")


def _handle_inline_search(query: str, project_id: int):
    from kb.retriever.hybrid import hybrid_search
    results = hybrid_search(query, top_k=5, project_id=project_id)
    if not results:
        warning("No results found.")
        return
    for _, score, content, fp in results:
        console.print(f"\n[dim]{fp}[/dim]")
        console.print(content[:400])


def _show_chat_history(chat_id: int):
    from kb.memory.conversation import get_chat_messages
    msgs = get_chat_messages(chat_id)
    for m in msgs[-10:]:
        role_style = "cyan" if m.role == "user" else "green"
        console.print(f"[{role_style}]{m.role.upper()}[/{role_style}]: {m.content[:200]}")


@app.command()
def prompt(
    issue: str = typer.Argument(..., help="Issue or task description"),
    path: Optional[str] = typer.Option(None, "--path", "-p", help="Project path"),
    copy: bool = typer.Option(True, "--copy/--no-copy", help="Copy prompt to clipboard"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save prompt to markdown file"),
):
    """Generate a super-strong detailed prompt for Antigravity to solve an issue."""
    from kb.config.settings import get_settings
    from kb.git.reader import read_git_state
    from kb.indexer.auto_scan import auto_scan_if_needed
    from kb.prompt_engine.antigravity import build_antigravity_prompt, copy_to_clipboard

    settings = get_settings()
    root = Path(path or settings.project.default_path).resolve()
    _ensure_db()

    auto_scan_if_needed(root, silent=True)
    project_id = _get_project_id(root)
    if project_id is None:
        project_id = _get_or_create_project(root)

    git_state = read_git_state(root)
    git_context = git_state.to_context_string()

    header(f"Generating Antigravity Prompt: {issue!r}")
    super_prompt = build_antigravity_prompt(
        issue_query=issue,
        project_id=project_id,
        project_path=root,
        top_k=settings.retrieval.top_k,
        git_context_str=git_context,
    )

    if output:
        out_path = Path(output).resolve()
        out_path.write_text(super_prompt, encoding="utf-8")
        success(f"Prompt saved to {out_path}")

    copied = False
    if copy:
        copied = copy_to_clipboard(super_prompt)

    copy_msg = " [bold green](Copied to clipboard!)[/bold green]" if copied else ""
    console.print(Panel(
        Markdown(super_prompt),
        title=f"🚀 [bold cyan]Antigravity Super-Prompt[/bold cyan]{copy_msg}",
        border_style="cyan",
        expand=False,
    ))


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="Search mode: hybrid|semantic|keyword"),
    path: Optional[str] = typer.Option(None, "--path", "-p", help="Project path"),
):
    """Search the knowledge base (no LLM call)."""
    from kb.config.settings import get_settings
    from kb.indexer.auto_scan import auto_scan_if_needed

    settings = get_settings()
    root = Path(path or settings.project.default_path).resolve()
    _ensure_db()
    
    auto_scan_if_needed(root, silent=True)
    project_id = _get_project_id(root)

    header(f"Search: {query!r} [{mode}]")

    if mode == "semantic":
        from kb.retriever.semantic import semantic_search
        results = semantic_search(query, top_k=top_k, project_id=project_id)
    elif mode == "keyword":
        from kb.retriever.keyword import keyword_search
        results = keyword_search(query, top_k=top_k, project_id=project_id)
    else:
        from kb.retriever.hybrid import hybrid_search
        results = hybrid_search(query, top_k=top_k, project_id=project_id)

    if not results:
        warning("No results found.")
        return

    for i, (cid, score, content, fp) in enumerate(results, 1):
        console.print(Panel(
            content[:600],
            title=f"[{i}] [cyan]{fp}[/cyan]  score={score:.4f}",
            border_style="dim blue",
            expand=False,
        ))


@app.command()
def status(
    path: Optional[str] = typer.Argument(None, help="Project directory"),
):
    """Show index status and project statistics."""
    from kb.config.settings import get_settings
    from kb.database.db import get_session
    from kb.database.models import Chunk, File, Project
    from kb.indexer.auto_scan import auto_scan_if_needed

    settings = get_settings()
    root = Path(path or settings.project.default_path).resolve()
    _ensure_db()
    
    auto_scan_if_needed(root, silent=True)

    with get_session() as session:
        project = session.query(Project).filter_by(path=str(root)).first()

        if project is None:
            warning(f"No project indexed at {root}")
            info("Run [bold]kb scan[/bold] to index.")
            return

        file_count = session.query(File).filter_by(project_id=project.id).count()
        chunk_count = session.query(Chunk).join(File).filter(File.project_id == project.id).count()

        header("Project Status")
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Project", project.name)
        table.add_row("Path", project.path)
        table.add_row("Files indexed", str(file_count))
        table.add_row("Chunks", str(chunk_count))
        table.add_row("Last scanned", str(project.last_scanned or "never"))

        lang_stats = project.get_language_stats()
        if lang_stats:
            lang_str = ", ".join(
                f"{k}:{v}" for k, v in sorted(lang_stats.items(), key=lambda x: -x[1])[:10]
            )
            table.add_row("Languages", lang_str)

        console.print(table)

    # Vector store stats
    try:
        from kb.embeddings.vector_store import get_vector_store
        store = get_vector_store()
        info(f"Vector store: {store.total_vectors} vectors")
    except Exception:
        pass


@app.command()
def git(
    path: Optional[str] = typer.Argument(None, help="Project directory"),
):
    """Show current git state for the project."""
    from kb.config.settings import get_settings
    from kb.git.reader import read_git_state

    settings = get_settings()
    root = Path(path or settings.project.default_path).resolve()
    header("Git State")

    state = read_git_state(root)
    if not state.is_git_repo:
        warning("Not a git repository.")
        return

    console.print(state.to_context_string())


@app.command()
def memory(
    path: Optional[str] = typer.Argument(None, help="Project directory"),
    add: Optional[str] = typer.Option(None, "--add", help="Add a note"),
    category: str = typer.Option("general", "--category", "-c", help="Note category"),
    tasks: bool = typer.Option(False, "--tasks", "-t", help="Show tasks"),
):
    """View and add long-term memory notes."""
    from kb.config.settings import get_settings
    from kb.memory.long_term import (
        NOTE_CATEGORIES, get_notes, get_tasks, save_note, save_task,
    )

    settings = get_settings()
    root = Path(path or settings.project.default_path).resolve()
    _ensure_db()
    project_id = _get_project_id(root)

    if project_id is None:
        warning("No project indexed. Run [bold]kb scan[/bold] first.")
        raise typer.Exit(1)

    if add:
        if tasks:
            tid = save_task(project_id, add)
            success(f"Task #{tid} saved: {add}")
        else:
            nid = save_note(project_id, category, add[:100], add)
            success(f"Note #{nid} saved [{category}]: {add[:60]}")
        return

    header("Long-Term Memory")

    if tasks:
        task_list = get_tasks(project_id)
        if not task_list:
            info("No tasks.")
            return
        rows = [[t.id, t.status, t.priority, t.title[:60]] for t in task_list]
        print_table("Tasks", ["ID", "Status", "Priority", "Title"], rows)
        return

    notes = get_notes(project_id)
    if not notes:
        info("No notes yet. Use [bold]--add[/bold] to add one.")
        return

    for note in notes:
        cat = NOTE_CATEGORIES.get(note.category, note.category)
        console.print(Panel(
            note.content,
            title=f"[{note.id}] [cyan]{cat}[/cyan] — {note.title}",
            border_style="blue",
            expand=False,
        ))


@app.command()
def rebuild(
    path: Optional[str] = typer.Argument(None, help="Project directory"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Force a full rebuild of the index (drops and re-indexes everything)."""
    from kb.config.settings import get_settings

    settings = get_settings()
    root = Path(path or settings.project.default_path).resolve()

    if not yes:
        if not Confirm.ask(f"[yellow]Full rebuild of {root}? This will re-index all files.[/yellow]"):
            raise typer.Exit()

    # Delegate to scan with --rebuild
    ctx = typer.Context(scan)
    scan(path=str(root), rebuild=True, workers=None)


@app.command()
def stats(
    path: Optional[str] = typer.Argument(None, help="Project directory"),
):
    """Detailed statistics: tokens, chunks, chats, notes."""
    from kb.config.settings import get_settings
    from kb.database.db import get_session
    from kb.database.models import ArchitectureNote, Chat, Chunk, File, Message, Task

    settings = get_settings()
    root = Path(path or settings.project.default_path).resolve()
    _ensure_db()
    project_id = _get_project_id(root)

    if project_id is None:
        warning("No project indexed.")
        return

    header("Detailed Statistics")

    with get_session() as session:
        file_count = session.query(File).filter_by(project_id=project_id).count()
        chunk_count = session.query(Chunk).join(File).filter(File.project_id == project_id).count()
        total_tokens = session.execute(
            __import__("sqlalchemy").text(
                "SELECT COALESCE(SUM(c.token_count), 0) FROM chunks c "
                "JOIN files f ON c.file_id = f.id WHERE f.project_id = :pid"
            ),
            {"pid": project_id},
        ).scalar()
        chat_count = session.query(Chat).filter_by(project_id=project_id).count()
        msg_count = session.query(Message).join(Chat).filter(Chat.project_id == project_id).count()
        note_count = session.query(ArchitectureNote).filter_by(project_id=project_id).count()
        task_count = session.query(Task).filter_by(project_id=project_id).count()

    rows = [
        ["Files indexed", str(file_count)],
        ["Text chunks", str(chunk_count)],
        ["Total tokens indexed", str(total_tokens)],
        ["Chat sessions", str(chat_count)],
        ["Total messages", str(msg_count)],
        ["Architecture notes", str(note_count)],
        ["Tasks", str(task_count)],
    ]
    print_table("Knowledge Base Stats", ["Metric", "Value"], rows)


@app.command()
def watch(
    path: Optional[str] = typer.Argument(None, help="Project directory"),
):
    """Watch for file changes and auto-update the index."""
    from kb.config.settings import get_settings
    from kb.indexer.watcher import FileWatcher

    settings = get_settings()
    root = Path(path or settings.project.default_path).resolve()
    _ensure_db()
    project_id = _get_project_id(root)

    if project_id is None:
        warning("No project indexed. Run [bold]kb scan[/bold] first.")
        raise typer.Exit(1)

    watcher = FileWatcher(project_id=project_id, root=root)
    watcher.run_forever()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
