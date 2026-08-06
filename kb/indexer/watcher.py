"""Live file watcher using watchdog for real-time incremental indexing."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

from kb.utils.display import console


class _KBEventHandler(FileSystemEventHandler):
    """Debounced file system event handler that queues changes for indexing."""

    def __init__(self, project_id: int, root: Path, debounce_seconds: float = 2.0):
        super().__init__()
        self._project_id = project_id
        self._root = root
        self._debounce = debounce_seconds
        self._pending: dict[str, str] = {}   # path -> event type
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

    def _schedule_flush(self):
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self._debounce, self._flush)
        self._timer.daemon = True
        self._timer.start()

    def _flush(self):
        with self._lock:
            pending = dict(self._pending)
            self._pending.clear()

        if not pending:
            return

        from kb.config.settings import get_settings
        from kb.database.db import get_session
        from kb.database.models import File
        from kb.indexer.worker import delete_files_from_index, run_indexing

        settings = get_settings()
        to_index: list[Path] = []
        to_delete: list[str] = []

        for path_str, event_type in pending.items():
            path = Path(path_str)
            if event_type == "deleted":
                try:
                    rel = str(path.relative_to(self._root))
                    to_delete.append(rel)
                except ValueError:
                    pass
            else:
                if path.exists() and path.is_file():
                    to_index.append(path)

        if to_delete:
            delete_files_from_index(to_delete, self._project_id)
            console.print(f"[dim][del] Removed {len(to_delete)} deleted file(s) from index[/dim]")

        if to_index:
            run_indexing(to_index, self._root, self._project_id,
                         workers=settings.indexer.workers)
            console.print(f"[dim][~] Re-indexed {len(to_index)} changed file(s)[/dim]")

    def _record(self, event: FileSystemEvent, event_type: str):
        from kb.config.settings import get_settings
        settings = get_settings()

        path = Path(event.src_path)
        if path.is_dir():
            return

        # Check if file should be indexed
        from kb.indexer.scanner import _should_include
        supported = set(settings.supported_extensions)
        ignored = set(settings.ignored_extensions)
        if not _should_include(path, supported, ignored):
            return

        with self._lock:
            self._pending[str(path)] = event_type
        self._schedule_flush()

    def on_created(self, event):
        self._record(event, "created")

    def on_modified(self, event):
        self._record(event, "modified")

    def on_deleted(self, event):
        self._record(event, "deleted")

    def on_moved(self, event):
        # Treat as delete + create
        self._record(event, "deleted")
        from watchdog.events import FileMovedEvent
        if isinstance(event, FileMovedEvent):
            class _FakeEvent:
                src_path = event.dest_path
                is_directory = event.is_directory
            self._record(_FakeEvent(), "created")


class FileWatcher:
    """Wraps watchdog Observer for a project directory."""

    def __init__(self, project_id: int, root: Path):
        self._project_id = project_id
        self._root = root
        self._observer: Optional[Observer] = None

    def start(self):
        handler = _KBEventHandler(self._project_id, self._root)
        self._observer = Observer()
        self._observer.schedule(handler, str(self._root), recursive=True)
        self._observer.start()
        console.print(f"[success][>] Watching {self._root} for changes...[/success]")

    def stop(self):
        if self._observer:
            self._observer.stop()
            self._observer.join()
            console.print("[dim]Watcher stopped.[/dim]")

    def run_forever(self):
        """Block until Ctrl+C."""
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
