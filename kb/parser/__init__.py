"""kb.parser package — parser registry."""

from __future__ import annotations

from pathlib import Path
from typing import List

from kb.parser.base import BaseParser, ParsedFile
from kb.parser.generic_parser import GenericParser
from kb.parser.python_parser import PythonParser
from kb.parser.ts_parser import TypeScriptParser

_PARSERS: List[BaseParser] = [
    PythonParser(),
    TypeScriptParser(),
    GenericParser(),   # fallback — always last
]


def get_parser(path: Path) -> BaseParser:
    """Return the first parser that can handle this file."""
    for parser in _PARSERS:
        if parser.can_parse(path):
            return parser
    return _PARSERS[-1]


def parse_file(path: Path, content: str) -> ParsedFile:
    """Parse a file and return structured metadata."""
    return get_parser(path).parse(path, content)
