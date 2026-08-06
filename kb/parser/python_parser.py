"""Python AST-based file parser."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List

from kb.parser.base import BaseParser, ParsedFile

_TODO_RE = re.compile(r"#\s*(TODO|FIXME|HACK|XXX|BUG)[:\s]+(.*)", re.IGNORECASE)


class PythonParser(BaseParser):

    def can_parse(self, path: Path) -> bool:
        return path.suffix == ".py"

    def parse(self, path: Path, content: str) -> ParsedFile:
        pf = ParsedFile(path=str(path), language="python", raw_content=content)

        try:
            tree = ast.parse(content)
        except SyntaxError:
            pf.summary = f"[SyntaxError — could not parse {path.name}]"
            return pf

        pf.imports = self._extract_imports(tree)
        pf.functions = self._extract_functions(tree)
        pf.classes = self._extract_classes(tree)
        pf.todos = self._extract_todos(content)
        pf.summary = self._build_summary(pf)
        return pf

    # ------------------------------------------------------------------

    def _extract_imports(self, tree: ast.Module) -> List[str]:
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
        return imports[:60]

    def _extract_functions(self, tree: ast.Module) -> List[dict]:
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a.arg for a in node.args.args]
                decorators = []
                for d in node.decorator_list:
                    if isinstance(d, ast.Name):
                        decorators.append(d.id)
                    elif isinstance(d, ast.Attribute):
                        decorators.append(f"{d.value.id}.{d.attr}" if isinstance(d.value, ast.Name) else d.attr)

                docstring = ast.get_docstring(node) or ""
                functions.append({
                    "name": node.name,
                    "args": args,
                    "decorators": decorators,
                    "lineno": node.lineno,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "docstring": docstring[:200],
                })
        return functions

    def _extract_classes(self, tree: ast.Module) -> List[dict]:
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = []
                for b in node.bases:
                    if isinstance(b, ast.Name):
                        bases.append(b.id)
                    elif isinstance(b, ast.Attribute):
                        bases.append(b.attr)

                methods = [
                    n.name for n in ast.walk(node)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                docstring = ast.get_docstring(node) or ""
                classes.append({
                    "name": node.name,
                    "bases": bases,
                    "methods": methods,
                    "lineno": node.lineno,
                    "docstring": docstring[:200],
                })
        return classes

    def _extract_todos(self, content: str) -> List[str]:
        return [f"{m.group(1)}: {m.group(2).strip()}" for m in _TODO_RE.finditer(content)][:20]

    def _build_summary(self, pf: ParsedFile) -> str:
        parts = []
        if pf.classes:
            names = ", ".join(c["name"] for c in pf.classes[:5])
            parts.append(f"Classes: {names}")
        if pf.functions:
            names = ", ".join(f["name"] for f in pf.functions[:8])
            parts.append(f"Functions: {names}")
        if pf.imports:
            parts.append(f"Imports {len(pf.imports)} modules")
        return ". ".join(parts)
