"""Long-term memory — architecture notes, preferences, tasks, ideas."""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

from kb.database.db import get_session
from kb.database.models import ArchitectureNote, Task, UserPreference

# Valid note categories
NOTE_CATEGORIES = {
    "decision": "Architecture Decision",
    "preference": "Coding Preference",
    "convention": "Naming Convention",
    "bug": "Known Bug",
    "idea": "Future Idea",
    "task": "Task / TODO",
    "general": "General Note",
}


def save_note(
    project_id: int,
    category: str,
    title: str,
    content: str,
    tags: Optional[List[str]] = None,
) -> int:
    """Save an architecture note. Returns the note ID."""
    with get_session() as session:
        note = ArchitectureNote(
            project_id=project_id,
            category=category,
            title=title,
            content=content,
            tags=json.dumps(tags or []),
        )
        session.add(note)
        session.flush()
        return note.id


def get_notes(project_id: int, category: Optional[str] = None) -> List[ArchitectureNote]:
    """Retrieve architecture notes, optionally filtered by category."""
    with get_session() as session:
        q = session.query(ArchitectureNote).filter_by(project_id=project_id)
        if category:
            q = q.filter_by(category=category)
        return q.order_by(ArchitectureNote.created_at.desc()).all()


def get_relevant_notes(query: str, project_id: int, top_k: int = 5) -> List[ArchitectureNote]:
    """Find semantically relevant architecture notes."""
    from kb.embeddings import get_embedder
    import numpy as np

    with get_session() as session:
        notes = session.query(ArchitectureNote).filter_by(project_id=project_id).all()
        if not notes:
            return []

        embedder = get_embedder()
        query_vec = embedder.embed_one(query)
        texts = [f"{n.title} {n.content}"[:512] for n in notes]
        vecs = embedder.embed(texts)

        scores = vecs @ query_vec
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [notes[i] for i in top_idx]


def save_task(project_id: int, title: str, description: str = "", priority: str = "medium") -> int:
    """Create a new task. Returns the task ID."""
    with get_session() as session:
        task = Task(
            project_id=project_id,
            title=title,
            description=description,
            priority=priority,
            status="pending",
        )
        session.add(task)
        session.flush()
        return task.id


def get_tasks(project_id: int, status: Optional[str] = None) -> List[Task]:
    """Return tasks for a project."""
    with get_session() as session:
        q = session.query(Task).filter_by(project_id=project_id)
        if status:
            q = q.filter_by(status=status)
        return q.order_by(Task.created_at.desc()).all()


def update_task_status(task_id: int, status: str):
    """Update a task status."""
    with get_session() as session:
        task = session.query(Task).get(task_id)
        if task:
            task.status = status
            task.updated_at = datetime.utcnow()


def get_preference(key: str) -> Optional[str]:
    """Retrieve a user preference value."""
    with get_session() as session:
        pref = session.query(UserPreference).filter_by(key=key).first()
        return pref.value if pref else None


def set_preference(key: str, value: str):
    """Set a user preference."""
    with get_session() as session:
        pref = session.query(UserPreference).filter_by(key=key).first()
        if pref:
            pref.value = value
        else:
            pref = UserPreference(key=key, value=value)
            session.add(pref)


def format_notes_for_context(notes: List[ArchitectureNote]) -> str:
    """Format notes as a human-readable context block."""
    if not notes:
        return ""
    lines = []
    for note in notes:
        cat = NOTE_CATEGORIES.get(note.category, note.category)
        lines.append(f"[{cat}] {note.title}")
        lines.append(note.content.strip())
        lines.append("")
    return "\n".join(lines).strip()
