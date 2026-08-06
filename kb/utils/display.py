import sys
from rich.console import Console
from rich.panel import Panel

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.theme import Theme

KB_THEME = Theme({
    "info": "bold cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "dim": "dim white",
    "thinking": "italic dim magenta",
    "answer": "white",
    "header": "bold blue",
    "accent": "bold cyan",
})

console = Console(theme=KB_THEME)


def info(msg: str):
    console.print(f"[info][i] {msg}[/info]")


def success(msg: str):
    console.print(f"[success][+] {msg}[/success]")


def warning(msg: str):
    console.print(f"[warning][!] {msg}[/warning]")


def error(msg: str):
    console.print(f"[error][x] {msg}[/error]")


def header(title: str):
    console.rule(f"[header]{title}[/header]")


def make_progress() -> Progress:
    """Return a styled Rich Progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def print_table(title: str, columns: list[str], rows: list[list]) -> None:
    """Print a styled table."""
    table = Table(title=title, show_header=True, header_style="bold cyan")
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*[str(c) for c in row])
    console.print(table)


def print_panel(content: str, title: str = "", style: str = "cyan"):
    console.print(Panel(content, title=title, border_style=style))
