"""Generic regex-based parser for Go, Rust, Java, C/C++, and other languages."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from kb.parser.base import BaseParser, ParsedFile

# Language detection
_LANG_MAP = {
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".txt": "text",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".dockerfile": "dockerfile",
}

# Per-language patterns: (import_pattern, function_pattern, class_pattern)
_PATTERNS = {
    "go": {
        "import": re.compile(r'import\s+"([^"]+)"', re.MULTILINE),
        "function": re.compile(r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(", re.MULTILINE),
        "class": re.compile(r"^type\s+(\w+)\s+struct\b", re.MULTILINE),
        "todo": re.compile(r"//\s*(TODO|FIXME|HACK|BUG)[:\s]+(.*)", re.IGNORECASE),
    },
    "rust": {
        "import": re.compile(r"^use\s+([\w::{}, ]+);", re.MULTILINE),
        "function": re.compile(r"^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*\(", re.MULTILINE),
        "class": re.compile(r"^(?:pub\s+)?struct\s+(\w+)", re.MULTILINE),
        "todo": re.compile(r"//\s*(TODO|FIXME|HACK|BUG)[:\s]+(.*)", re.IGNORECASE),
    },
    "java": {
        "import": re.compile(r"^import\s+([\w.]+);", re.MULTILINE),
        "function": re.compile(r"(?:public|private|protected|static|\s)+[\w<>[\]]+\s+(\w+)\s*\([^)]*\)\s*(?:throws[\w,\s]+)?\s*\{", re.MULTILINE),
        "class": re.compile(r"(?:public|private|abstract|final|\s)*class\s+(\w+)", re.MULTILINE),
        "todo": re.compile(r"//\s*(TODO|FIXME|HACK|BUG)[:\s]+(.*)", re.IGNORECASE),
    },
    "cpp": {
        "import": re.compile(r"^#include\s+[<\"]([^>\"]+)[>\"]", re.MULTILINE),
        "function": re.compile(r"^[\w:*&<>]+\s+(\w+)\s*\([^)]*\)\s*(?:const)?\s*\{", re.MULTILINE),
        "class": re.compile(r"^(?:class|struct)\s+(\w+)", re.MULTILINE),
        "todo": re.compile(r"//\s*(TODO|FIXME|HACK|BUG)[:\s]+(.*)", re.IGNORECASE),
    },
    "c": {
        "import": re.compile(r"^#include\s+[<\"]([^>\"]+)[>\"]", re.MULTILINE),
        "function": re.compile(r"^[\w*]+\s+(\w+)\s*\([^)]*\)\s*\{", re.MULTILINE),
        "class": re.compile(r"^struct\s+(\w+)", re.MULTILINE),
        "todo": re.compile(r"//\s*(TODO|FIXME|HACK|BUG)[:\s]+(.*)", re.IGNORECASE),
    },
}

_DEFAULT_TODO = re.compile(r"(?:#|//)\s*(TODO|FIXME|HACK|BUG)[:\s]+(.*)", re.IGNORECASE)


class GenericParser(BaseParser):

    def can_parse(self, path: Path) -> bool:
        return True  # Fallback: always can parse

    def parse(self, path: Path, content: str) -> ParsedFile:
        lang = _LANG_MAP.get(path.suffix.lower(), "text")
        pf = ParsedFile(path=str(path), language=lang, raw_content=content)

        patterns = _PATTERNS.get(lang, {})

        if "import" in patterns:
            pf.imports = list(dict.fromkeys(patterns["import"].findall(content)))[:60]
        if "function" in patterns:
            pf.functions = [
                {"name": m.group(1), "lineno": content[:m.start()].count("\n") + 1}
                for m in patterns["function"].finditer(content)
            ]
        if "class" in patterns:
            pf.classes = [
                {"name": m.group(1), "lineno": content[:m.start()].count("\n") + 1}
                for m in patterns["class"].finditer(content)
            ]

        todo_pat = patterns.get("todo", _DEFAULT_TODO)
        pf.todos = [f"{m.group(1)}: {m.group(2).strip()}" for m in todo_pat.finditer(content)][:20]
        pf.summary = self._build_summary(pf)
        return pf

    def _build_summary(self, pf: ParsedFile) -> str:
        parts = []
        if pf.classes:
            parts.append("Types/Structs: " + ", ".join(c["name"] for c in pf.classes[:5]))
        if pf.functions:
            parts.append("Functions: " + ", ".join(f["name"] for f in pf.functions[:8]))
        return ". ".join(parts)
