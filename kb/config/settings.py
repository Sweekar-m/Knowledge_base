"""Configuration management using Pydantic Settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class NvidiaConfig(BaseModel):
    api_key: str = "<YOUR_NVIDIA_API_KEY>"
    base_url: str = "https://integrate.api.nvidia.com/v1"
    model: str = "nvidia/nemotron-3-ultra-550b-a55b"
    temperature: float = 0.3
    top_p: float = 0.95
    max_tokens: int = 16384
    reasoning_budget: int = 16384
    enable_thinking: bool = True


class ProjectConfig(BaseModel):
    default_path: str = "."
    name: str = "MyProject"


class EmbeddingsConfig(BaseModel):
    provider: str = "local"
    model: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    dimension: int = 384


class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64


class RetrievalConfig(BaseModel):
    top_k: int = 10
    max_context_tokens: int = 12000


class DatabaseConfig(BaseModel):
    path: str = "~/.kb/kb.sqlite"
    faiss_index_path: str = "~/.kb/faiss.index"


class IndexerConfig(BaseModel):
    workers: int = 4
    max_file_size_mb: int = 5
    auto_scan: bool = True


class MemoryConfig(BaseModel):
    recent_chats: int = 3
    max_past_messages: int = 10


class GitConfig(BaseModel):
    recent_commits: int = 10
    diff_max_lines: int = 200


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------

class Settings(BaseModel):
    nvidia: NvidiaConfig = Field(default_factory=NvidiaConfig)
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    ignored_dirs: List[str] = Field(default_factory=lambda: [
        "node_modules", ".git", "build", "dist", "target",
        ".venv", "venv", "__pycache__", ".mypy_cache",
        ".pytest_cache", ".ruff_cache", "coverage",
    ])
    ignored_extensions: List[str] = Field(default_factory=lambda: [
        ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
        ".mp4", ".avi", ".mov", ".mp3", ".wav", ".pdf",
        ".zip", ".tar", ".gz", ".exe", ".dll", ".so", ".dylib", ".bin",
    ])
    supported_extensions: List[str] = Field(default_factory=lambda: [
        ".ts", ".tsx", ".js", ".jsx", ".py", ".java", ".go", ".rs",
        ".cpp", ".c", ".h", ".hpp", ".json", ".yaml", ".yml",
        ".md", ".txt", ".toml",
    ])
    supported_filenames: List[str] = Field(default_factory=lambda: [
        "package.json", "README.md", "Dockerfile",
        "docker-compose.yml", ".env.example", "Makefile",
    ])
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    indexer: IndexerConfig = Field(default_factory=IndexerConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    git: GitConfig = Field(default_factory=GitConfig)

    @property
    def db_path(self) -> Path:
        return Path(self.database.path).expanduser()

    @property
    def faiss_path(self) -> Path:
        return Path(self.database.faiss_index_path).expanduser()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_PKG_CONFIG = Path(__file__).resolve().parent.parent.parent / "kb_config.yaml"

_CONFIG_SEARCH_PATHS = [
    Path.cwd() / "kb_config.yaml",
    Path.cwd() / "kb_config.yml",
    Path.home() / ".kb" / "kb_config.yaml",
    Path.home() / ".kb" / "kb_config.yml",
    _PKG_CONFIG,
]

_settings: Optional[Settings] = None


def _load_dotenv_files():
    """Load environment variables from .env files in CWD, ~/.kb/.env, or pkg root."""
    env_paths = [
        Path.cwd() / ".env",
        Path.home() / ".kb" / ".env",
        _PKG_CONFIG.parent / ".env",
    ]
    try:
        from dotenv import load_dotenv
        for p in env_paths:
            if p.exists():
                load_dotenv(p, override=False)
    except ImportError:
        for p in env_paths:
            if p.exists():
                try:
                    for line in p.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
                except Exception:
                    pass


def load_settings(config_path: Optional[Path] = None) -> Settings:
    """Load settings from YAML config file, falling back to defaults."""
    global _settings

    _load_dotenv_files()

    # Search for config file
    search_paths = [config_path] if config_path else _CONFIG_SEARCH_PATHS
    raw: dict = {}

    for p in search_paths:
        if p and p.exists():
            with open(p, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            break

    # Override from environment variables (.env or os.environ)
    env_api_key = os.environ.get("KB_NVIDIA_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    if env_api_key:
        raw.setdefault("nvidia", {})["api_key"] = env_api_key

    _settings = Settings(**raw)

    # Ensure DB directory exists
    _settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    _settings.faiss_path.parent.mkdir(parents=True, exist_ok=True)

    return _settings


def get_settings() -> Settings:
    """Return cached settings, loading from default paths if needed."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings
