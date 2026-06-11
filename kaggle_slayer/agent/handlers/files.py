"""File-side tool handlers — read/write within the workspace.

All handlers take a `ctx` as their first positional argument. The ctx
exposes the Workspace and Journal; concrete construction happens in the
Solver. These functions don't import the Solver to avoid circular deps —
the contract is structural ("ctx must have .workspace and .journal").

Path safety: write_file and read_file resolve paths relative to the
workspace root and reject anything outside it. A small set of paths
(run_log.jsonl, notes.jsonl, context.md) is additionally protected from
agent writes — only the harness writes those.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from kaggle_slayer.agent.tools import ToolError
from kaggle_slayer.harness.journal import NOTE_CATEGORIES

_PROTECTED_BASENAMES: frozenset[str] = frozenset({"run_log.jsonl", "notes.jsonl", "context.md"})
"""Lowercased protected basenames. write_file compares `target.name.lower()`
against this set so case-insensitive filesystems (macOS default, Windows)
can't sneak in `Run_Log.jsonl` (F3)."""


def _resolve_under(workspace_root: Path, rel_path: str) -> Path:
    """Resolve `rel_path` under workspace_root and reject path traversal."""
    base = workspace_root.resolve()
    target = (base / rel_path).resolve()
    try:
        target.relative_to(base)
    except ValueError as e:
        raise ToolError(f"path {rel_path!r} resolves outside workspace") from e
    return target


def read_context(ctx: Any) -> str:
    """Read the workspace's context.md (the agent's brief)."""
    p: Path = ctx.workspace.context_path
    if not p.exists():
        raise ToolError(f"context.md not found at {p}")
    return p.read_text()


def read_file(ctx: Any, *, path: str) -> str:
    """Read a file from inside the workspace."""
    target = _resolve_under(ctx.workspace.root, path)
    if not target.exists():
        raise ToolError(f"file not found: {path}")
    if not target.is_file():
        raise ToolError(f"path is not a file: {path}")
    return target.read_text()


def write_file(ctx: Any, *, path: str, content: str) -> str:
    """Write a file inside the workspace. Protected harness files are rejected.

    F3: protected-basename comparison is case-insensitive (macOS HFS+/APFS
    and Windows NTFS resolve `Run_Log.jsonl` and `run_log.jsonl` to the
    same file by default).

    F17: an empty or directory-resolving path raises a typed ToolError
    rather than a generic IsADirectoryError from pathlib's write_text.
    """
    target = _resolve_under(ctx.workspace.root, path)
    raw_dir = ctx.workspace.raw_dir.resolve()
    if target == raw_dir or target.is_relative_to(raw_dir):
        raise ToolError(
            f"path {path!r} is under raw/ — competition data is read-only"
        )
    if (
        target.name.lower() in _PROTECTED_BASENAMES
        and target.parent == ctx.workspace.root.resolve()
    ) or (
        # leaderboard.jsonl is the submit gate's evidence (first-submit and
        # regression classification read it); the agent must not forge it.
        target.name.lower() == "leaderboard.jsonl"
        and target.parent == ctx.workspace.submissions_dir.resolve()
    ):
        raise ToolError(f"path {path!r} is protected (harness writes it)")
    if target == ctx.workspace.root.resolve() or target.is_dir():
        raise ToolError(
            f"path {path!r} resolves to a directory; provide a file path"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"wrote {len(content)} bytes to {path}"


def sample_rows(ctx: Any, *, table: str, n: int = 10, random: bool = False) -> str:
    """Return a sample of n rows from raw/<table>.csv as a string.

    The `table` arg is treated as a basename — '..' segments or absolute
    paths are rejected outright (F2: path traversal via the table arg).
    """
    if table.startswith("/") or table.startswith("\\") or ".." in Path(table).parts:
        raise ToolError(
            f"table {table!r} resolves outside raw/ (no path traversal allowed)"
        )
    target = _resolve_under(ctx.workspace.raw_dir, f"{table}.csv")
    raw_resolved = ctx.workspace.raw_dir.resolve()
    if not target.is_relative_to(raw_resolved):
        raise ToolError(f"table {table!r} resolves outside raw/")
    if not target.exists():
        raise ToolError(f"{target.name} not found in raw/")
    df = pd.read_csv(target)
    df = df.sample(n=n, random_state=42) if random and n < len(df) else df.head(n)
    return str(df.to_string(max_cols=20, max_colwidth=40))


def take_note(ctx: Any, *, category: str, content: str) -> str:
    """Append a structured note to notes.jsonl."""
    if category not in NOTE_CATEGORIES:
        raise ToolError(
            f"unknown category {category!r}; allowed: {sorted(NOTE_CATEGORIES)}"
        )
    ctx.journal.take_note(category=category, content=content)
    return f"noted ({category})"
