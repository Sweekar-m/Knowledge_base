"""TypeScript / JavaScript parser using tree-sitter with regex fallback."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from kb.parser.base import BaseParser, ParsedFile

_TS_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"}

# Regex patterns for JS/TS parsing
_IMPORT_RE = re.compile(
    r"""(?:import\s+(?:(?:\{[^}]*\}|\*\s+as\s+\w+|\w+)(?:\s*,\s*(?:\{[^}]*\}|\w+))*)\s+from\s+|require\s*\(\s*)['"]([^'"]+)['"]""",
    re.MULTILINE,
)
_EXPORT_RE = re.compile(
    r"export\s+(?:default\s+)?(?:class|function|const|let|var|interface|enum|type)\s+(\w+)",
)
_FUNCTION_RE = re.compile(
    r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)",
    re.MULTILINE,
)
_ARROW_FUNCTION_RE = re.compile(
    r"(?:export\s+)?(?:const|let)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*(?::\s*\S+\s*)?=>",
    re.MULTILINE,
)
_CLASS_RE = re.compile(
    r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?",
    re.MULTILINE,
)
_INTERFACE_RE = re.compile(r"(?:export\s+)?interface\s+(\w+)", re.MULTILINE)
_ENUM_RE = re.compile(r"(?:export\s+)?(?:const\s+)?enum\s+(\w+)", re.MULTILINE)
_TODO_RE = re.compile(r"//\s*(TODO|FIXME|HACK|XXX|BUG)[:\s]+(.*)", re.IGNORECASE)


class TypeScriptParser(BaseParser):

    def can_parse(self, path: Path) -> bool:
        return path.suffix in _TS_EXTENSIONS

    def parse(self, path: Path, content: str) -> ParsedFile:
        ext = path.suffix
        if ext in (".ts", ".tsx"):
            lang = "typescript"
        elif ext in (".jsx", ".tsx"):
            lang = "react"
        else:
            lang = "javascript"

        pf = ParsedFile(path=str(path), language=lang, raw_content=content)

        pf.imports = self._extract_imports(content)
        pf.exports = self._extract_exports(content)
        pf.functions = self._extract_functions(content)
        pf.classes = self._extract_classes(content)
        pf.interfaces = self._extract_interfaces(content)
        pf.enums = self._extract_enums(content)
        pf.todos = self._extract_todos(content)
        pf.summary = self._build_summary(pf)
        return pf

    def _extract_imports(self, content: str) -> List[str]:
        return list(dict.fromkeys(_IMPORT_RE.findall(content)))[:60]

    def _extract_exports(self, content: str) -> List[str]:
        return list(dict.fromkeys(_EXPORT_RE.findall(content)))[:30]

    def _extract_functions(self, content: str) -> List[dict]:
        fns = {}
        for m in _FUNCTION_RE.finditer(content):
            name = m.group(1)
            fns[name] = {"name": name, "args": m.group(2).split(","), "lineno": content[:m.start()].count("\n") + 1}
        for m in _ARROW_FUNCTION_RE.finditer(content):
            name = m.group(1)
            if name not in fns:
                fns[name] = {"name": name, "args": [], "lineno": content[:m.start()].count("\n") + 1}
        return list(fns.values())

    def _extract_classes(self, content: str) -> List[dict]:
        classes = []
        for m in _CLASS_RE.finditer(content):
            classes.append({
                "name": m.group(1),
                "extends": m.group(2) or "",
                "implements": [i.strip() for i in (m.group(3) or "").split(",") if i.strip()],
                "lineno": content[:m.start()].count("\n") + 1,
            })
        return classes

    def _extract_interfaces(self, content: str) -> List[dict]:
        return [{"name": m.group(1), "lineno": content[:m.start()].count("\n") + 1}
                for m in _INTERFACE_RE.finditer(content)]

    def _extract_enums(self, content: str) -> List[dict]:
        return [{"name": m.group(1)} for m in _ENUM_RE.finditer(content)]

    def _extract_todos(self, content: str) -> List[str]:
        return [f"{m.group(1)}: {m.group(2).strip()}" for m in _TODO_RE.finditer(content)][:20]

    def _build_summary(self, pf: ParsedFile) -> str:
        parts = []
        if pf.classes:
            parts.append("Classes: " + ", ".join(c["name"] for c in pf.classes[:5]))
        if pf.interfaces:
            parts.append("Interfaces: " + ", ".join(i["name"] for i in pf.interfaces[:5]))
        if pf.enums:
            parts.append("Enums: " + ", ".join(e["name"] for e in pf.enums[:5]))
        if pf.functions:
            parts.append("Functions: " + ", ".join(f["name"] for f in pf.functions[:8]))
        if pf.imports:
            parts.append(f"Imports {len(pf.imports)} modules")
        return ". ".join(parts)
