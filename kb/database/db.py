"""Database session factory and schema management."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from kb.config.settings import get_settings
from kb.database.models import Base

_engine = None
_SessionFactory = None


def _get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        db_path = settings.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        _engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            echo=False,
        )

        # Enable WAL mode and foreign keys for SQLite
        @event.listens_for(_engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.close()

    return _engine


def _get_session_factory():
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=_get_engine(), expire_on_commit=False)
    return _SessionFactory


def init_db():
    """Create all tables and FTS5 virtual table if they don't exist."""
    engine = _get_engine()
    Base.metadata.create_all(engine)
    _create_fts5_table(engine)


def _create_fts5_table(engine):
    """Create SQLite FTS5 virtual table for full-text search on chunks."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(
                content,
                file_path UNINDEXED,
                chunk_id UNINDEXED,
                content='chunks',
                content_rowid='id'
            )
        """))

        # Triggers to keep FTS in sync
        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_insert
            AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content, chunk_id)
                VALUES (new.id, new.content, new.id);
            END
        """))

        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_delete
            AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, chunk_id)
                VALUES ('delete', old.id, old.content, old.id);
            END
        """))

        conn.execute(text("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_update
            AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, chunk_id)
                VALUES ('delete', old.id, old.content, old.id);
                INSERT INTO chunks_fts(rowid, content, chunk_id)
                VALUES (new.id, new.content, new.id);
            END
        """))
        conn.commit()


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager providing a SQLAlchemy session with auto-commit/rollback."""
    factory = _get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_raw_session() -> Session:
    """Return a raw session (caller is responsible for commit/close)."""
    return _get_session_factory()()
