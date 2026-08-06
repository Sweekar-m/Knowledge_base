"""Git integration using gitpython — reads repository state for context."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class CommitInfo:
    hash: str
    message: str
    author: str
    timestamp: str


@dataclass
class GitState:
    branch: str = "unknown"
    staged: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    untracked: List[str] = field(default_factory=list)
    recent_commits: List[CommitInfo] = field(default_factory=list)
    diff_summary: str = ""
    is_git_repo: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    def to_context_string(self, max_commits: int = 5, max_diff_lines: int = 100) -> str:
        """Format git state as a readable context block for the LLM."""
        if not self.is_git_repo:
            return "Not a git repository."

        lines = [f"Branch: {self.branch}"]

        if self.staged:
            lines.append(f"Staged ({len(self.staged)}): {', '.join(self.staged[:10])}")
        if self.modified:
            lines.append(f"Modified ({len(self.modified)}): {', '.join(self.modified[:10])}")
        if self.untracked:
            lines.append(f"Untracked ({len(self.untracked)}): {', '.join(self.untracked[:10])}")

        if self.recent_commits:
            lines.append(f"\nRecent commits (last {min(max_commits, len(self.recent_commits))}):")
            for c in self.recent_commits[:max_commits]:
                msg = c.message.split("\n")[0][:80]
                lines.append(f"  {c.hash[:8]} — {msg} ({c.author}, {c.timestamp[:10]})")

        if self.diff_summary:
            diff_lines = self.diff_summary.splitlines()[:max_diff_lines]
            lines.append(f"\nDiff summary:\n" + "\n".join(diff_lines))

        return "\n".join(lines)


def read_git_state(
    repo_path: Path,
    max_commits: int = 10,
    max_diff_lines: int = 200,
) -> GitState:
    """Read current git state from repo_path."""
    try:
        import git
        repo = git.Repo(repo_path, search_parent_directories=True)
    except Exception:
        return GitState(is_git_repo=False)

    state = GitState(is_git_repo=True)

    try:
        state.branch = repo.active_branch.name
    except TypeError:
        state.branch = "HEAD (detached)"

    # Status
    try:
        state.staged = [item.a_path for item in repo.index.diff("HEAD")]
    except Exception:
        state.staged = []

    try:
        state.modified = [item.a_path for item in repo.index.diff(None)]
    except Exception:
        state.modified = []

    try:
        state.untracked = repo.untracked_files[:30]
    except Exception:
        state.untracked = []

    # Recent commits
    try:
        commits = []
        for commit in repo.iter_commits(max_count=max_commits):
            commits.append(CommitInfo(
                hash=commit.hexsha,
                message=commit.message.strip(),
                author=str(commit.author),
                timestamp=str(datetime.fromtimestamp(commit.committed_date)),
            ))
        state.recent_commits = commits
    except Exception:
        state.recent_commits = []

    # Diff summary (staged + modified)
    try:
        diff_parts = []
        if state.staged:
            diff = repo.git.diff("--cached", "--stat")
            if diff:
                diff_parts.append("=== Staged diff ===\n" + diff)
        if state.modified:
            diff = repo.git.diff("--stat")
            if diff:
                diff_parts.append("=== Working tree diff ===\n" + diff)
        full_diff = "\n".join(diff_parts)
        lines = full_diff.splitlines()[:max_diff_lines]
        state.diff_summary = "\n".join(lines)
    except Exception:
        state.diff_summary = ""

    return state


def save_git_state(state: GitState, project_id: int):
    """Persist git state snapshot to the database."""
    from kb.database.db import get_session
    from kb.database.models import GitHistory

    commits = state.recent_commits
    latest = commits[0] if commits else None

    with get_session() as session:
        record = GitHistory(
            project_id=project_id,
            commit_hash=latest.hash if latest else None,
            branch=state.branch,
            message=latest.message if latest else None,
            author=latest.author if latest else None,
            timestamp=datetime.fromisoformat(latest.timestamp) if latest else None,
            staged_files_json=json.dumps(state.staged),
            modified_files_json=json.dumps(state.modified),
            untracked_files_json=json.dumps(state.untracked),
            diff_summary=state.diff_summary,
        )
        session.add(record)
