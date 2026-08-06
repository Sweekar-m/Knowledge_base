"""Abstract base class for file parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ParsedFile:
    """Structured representation of a parsed source file."""
    path: str = ""
    language: str = ""
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    classes: List[dict] = field(default_factory=list)
    functions: List[dict] = field(default_factory=list)
    interfaces: List[dict] = field(default_factory=list)
    enums: List[dict] = field(default_factory=list)
    todos: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    summary: str = ""
    raw_content: str = ""

    def to_index_text(self) -> str:
        """Return a text representation for embedding/indexing."""
        parts = [f"File: {self.path}"]
        if self.imports:
            parts.append("Imports: " + ", ".join(self.imports[:20]))
        if self.classes:
            class_names = [c.get("name", "") for c in self.classes]
            parts.append("Classes: " + ", ".join(class_names[:20]))
        if self.functions:
            fn_names = [f.get("name", "") for f in self.functions]
            parts.append("Functions: " + ", ".join(fn_names[:30]))
        if self.todos:
            parts.append("TODOs: " + "; ".join(self.todos[:5]))
        if self.summary:
            parts.append(self.summary)
        return "\n".join(parts)


class BaseParser(ABC):
    """Abstract parser interface."""

    @abstractmethod
    def can_parse(self, path: Path) -> bool:
        """Return True if this parser handles the given file."""
        ...

    @abstractmethod
    def parse(self, path: Path, content: str) -> ParsedFile:
        """Parse file content and return structured metadata."""
        ...
