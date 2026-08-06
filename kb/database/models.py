"""SQLAlchemy ORM models for the knowledge base."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True)
    name = Column(String(256), nullable=False)
    path = Column(String(1024), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_scanned = Column(DateTime, nullable=True)
    file_count = Column(Integer, default=0)
    language_stats = Column(Text, default="{}")  # JSON: {"py": 42, "ts": 18}

    files = relationship("File", back_populates="project", cascade="all, delete-orphan")
    chats = relationship("Chat", back_populates="project", cascade="all, delete-orphan")
    architecture_notes = relationship("ArchitectureNote", back_populates="project", cascade="all, delete-orphan")
    git_history = relationship("GitHistory", back_populates="project", cascade="all, delete-orphan")
    tasks = relationship("Task", back_populates="project", cascade="all, delete-orphan")
    bookmarks = relationship("Bookmark", back_populates="project", cascade="all, delete-orphan")

    def get_language_stats(self) -> dict:
        return json.loads(self.language_stats or "{}")

    def set_language_stats(self, stats: dict):
        self.language_stats = json.dumps(stats)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    path = Column(String(2048), nullable=False)
    relative_path = Column(String(2048), nullable=False)
    language = Column(String(64), nullable=True)
    file_hash = Column(String(64), nullable=True)
    size_bytes = Column(Integer, default=0)
    last_indexed = Column(DateTime, nullable=True)
    last_modified = Column(DateTime, nullable=True)
    # Parsed metadata stored as JSON
    imports_json = Column(Text, default="[]")
    exports_json = Column(Text, default="[]")
    classes_json = Column(Text, default="[]")
    functions_json = Column(Text, default="[]")
    interfaces_json = Column(Text, default="[]")
    enums_json = Column(Text, default="[]")
    todos_json = Column(Text, default="[]")
    dependencies_json = Column(Text, default="[]")
    summary = Column(Text, default="")

    __table_args__ = (UniqueConstraint("project_id", "path"),)

    project = relationship("Project", back_populates="files")
    chunks = relationship("Chunk", back_populates="file", cascade="all, delete-orphan")
    bookmarks = relationship("Bookmark", back_populates="file", cascade="all, delete-orphan")

    def get_imports(self) -> list:
        return json.loads(self.imports_json or "[]")

    def get_functions(self) -> list:
        return json.loads(self.functions_json or "[]")

    def get_classes(self) -> list:
        return json.loads(self.classes_json or "[]")

    def get_todos(self) -> list:
        return json.loads(self.todos_json or "[]")


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=False)
    content = Column(Text, nullable=False)
    start_line = Column(Integer, default=0)
    end_line = Column(Integer, default=0)
    chunk_index = Column(Integer, default=0)
    token_count = Column(Integer, default=0)
    # FAISS vector id (int64)
    vector_id = Column(Integer, nullable=True)

    file = relationship("File", back_populates="chunks")


# ---------------------------------------------------------------------------
# Chats & Messages
# ---------------------------------------------------------------------------

class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    title = Column(String(512), default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    git_state_json = Column(Text, default="{}")  # snapshot of git state at chat start

    project = relationship("Project", back_populates="chats")
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan",
                            order_by="Message.timestamp")

    def get_git_state(self) -> dict:
        return json.loads(self.git_state_json or "{}")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    chat_id = Column(Integer, ForeignKey("chats.id"), nullable=False)
    role = Column(String(32), nullable=False)   # "user" | "assistant" | "system"
    content = Column(Text, nullable=False)
    thinking = Column(Text, default="")         # reasoning tokens
    timestamp = Column(DateTime, default=datetime.utcnow)
    retrieved_chunks_json = Column(Text, default="[]")   # list of chunk IDs used
    token_count = Column(Integer, default=0)

    chat = relationship("Chat", back_populates="messages")

    def get_retrieved_chunks(self) -> list:
        return json.loads(self.retrieved_chunks_json or "[]")


# ---------------------------------------------------------------------------
# Architecture Notes
# ---------------------------------------------------------------------------

class ArchitectureNote(Base):
    __tablename__ = "architecture_notes"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    category = Column(String(128), default="general")
    # e.g. "decision" | "preference" | "convention" | "bug" | "idea" | "task"
    title = Column(String(512), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    tags = Column(Text, default="[]")   # JSON list of tag strings

    project = relationship("Project", back_populates="architecture_notes")

    def get_tags(self) -> list:
        return json.loads(self.tags or "[]")


# ---------------------------------------------------------------------------
# Git History
# ---------------------------------------------------------------------------

class GitHistory(Base):
    __tablename__ = "git_history"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    commit_hash = Column(String(64), nullable=True)
    branch = Column(String(256), nullable=True)
    message = Column(Text, nullable=True)
    author = Column(String(256), nullable=True)
    timestamp = Column(DateTime, nullable=True)
    staged_files_json = Column(Text, default="[]")
    modified_files_json = Column(Text, default="[]")
    untracked_files_json = Column(Text, default="[]")
    diff_summary = Column(Text, default="")
    recorded_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="git_history")


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    status = Column(String(32), default="pending")
    # "pending" | "in_progress" | "done" | "cancelled"
    title = Column(String(512), nullable=False)
    description = Column(Text, default="")
    priority = Column(String(16), default="medium")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project", back_populates="tasks")


# ---------------------------------------------------------------------------
# Bookmarks
# ---------------------------------------------------------------------------

class Bookmark(Base):
    __tablename__ = "bookmarks"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=True)
    note = Column(Text, default="")
    line_number = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="bookmarks")
    file = relationship("File", back_populates="bookmarks")


# ---------------------------------------------------------------------------
# User Preferences
# ---------------------------------------------------------------------------

class UserPreference(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True)
    key = Column(String(256), nullable=False, unique=True)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
