"""Conversation memory — store and retrieve chat sessions."""

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional, Tuple

from kb.database.db import get_session
from kb.database.models import Chat, Message


def new_chat(project_id: int, git_state_json: str = "{}") -> int:
    """Create a new chat session. Returns the chat ID."""
    with get_session() as session:
        chat = Chat(
            project_id=project_id,
            git_state_json=git_state_json,
        )
        session.add(chat)
        session.flush()
        return chat.id


def set_chat_title(chat_id: int, title: str):
    """Update the chat title (typically derived from first message)."""
    with get_session() as session:
        chat = session.query(Chat).get(chat_id)
        if chat:
            chat.title = title[:200]


def add_message(
    chat_id: int,
    role: str,
    content: str,
    thinking: str = "",
    retrieved_chunk_ids: Optional[List[int]] = None,
    token_count: int = 0,
):
    """Append a message to a chat session."""
    with get_session() as session:
        msg = Message(
            chat_id=chat_id,
            role=role,
            content=content,
            thinking=thinking,
            retrieved_chunks_json=json.dumps(retrieved_chunk_ids or []),
            token_count=token_count,
        )
        session.add(msg)
        # Update chat timestamp
        chat = session.query(Chat).get(chat_id)
        if chat:
            chat.updated_at = datetime.utcnow()


def get_chat_messages(chat_id: int) -> List[Message]:
    """Return all messages for a chat, ordered by timestamp."""
    with get_session() as session:
        return session.query(Message).filter_by(chat_id=chat_id).order_by(Message.timestamp).all()


def get_recent_chats(project_id: int, limit: int = 5) -> List[Chat]:
    """Return the most recent chat sessions for a project."""
    with get_session() as session:
        return (
            session.query(Chat)
            .filter_by(project_id=project_id)
            .order_by(Chat.updated_at.desc())
            .limit(limit)
            .all()
        )


def get_relevant_past_messages(
    query: str,
    project_id: int,
    limit: int = 5,
) -> List[Tuple[str, str]]:
    """
    Retrieve relevant past user/assistant pairs via semantic search over messages.
    
    Returns:
        List of (user_msg, assistant_msg) tuples.
    """
    from kb.embeddings import get_embedder
    import numpy as np

    with get_session() as session:
        # Get all user messages from this project
        user_msgs = (
            session.query(Message)
            .join(Chat)
            .filter(Chat.project_id == project_id, Message.role == "user")
            .order_by(Message.timestamp.desc())
            .limit(200)
            .all()
        )

        if not user_msgs:
            return []

        embedder = get_embedder()
        query_vec = embedder.embed_one(query)
        msg_texts = [m.content[:512] for m in user_msgs]
        msg_vecs = embedder.embed(msg_texts)

        # Cosine similarity (vectors are normalized)
        scores = msg_vecs @ query_vec
        top_indices = np.argsort(scores)[::-1][:limit]

        results = []
        for idx in top_indices:
            user_msg = user_msgs[idx]
            # Find the assistant reply
            assistant = (
                session.query(Message)
                .filter(
                    Message.chat_id == user_msg.chat_id,
                    Message.role == "assistant",
                    Message.timestamp > user_msg.timestamp,
                )
                .order_by(Message.timestamp)
                .first()
            )
            if assistant:
                results.append((user_msg.content[:400], assistant.content[:600]))

        return results


def format_chat_history_for_context(
    chat_id: int,
    max_messages: int = 10,
) -> List[dict]:
    """Return the last N messages as OpenAI-format dicts for multi-turn context."""
    messages = get_chat_messages(chat_id)
    recent = messages[-max_messages:] if len(messages) > max_messages else messages
    return [{"role": m.role, "content": m.content} for m in recent]
