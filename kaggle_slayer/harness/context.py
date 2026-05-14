"""Build context.md for a per-competition workspace.

The agent reads context.md as its system message. The file is regenerated
from scratch on each call to build_context (overwrites stale content).

For Week 2 the structure is intentionally simple — a markdown file with
named sections. Future weeks can add: parsed competition rules, public-LB
calibration history, learned patterns from prior comps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd  # type: ignore[import-untyped]

from kaggle_slayer.harness.kaggle_client import CompetitionFile, CompetitionInfo, LBEntry
from kaggle_slayer.harness.workspace import Workspace


class _KaggleClientLike(Protocol):
    """Minimal interface build_context needs — easy to fake in tests."""

    def view_competition(self, name: str) -> CompetitionInfo: ...
    def list_files(self, name: str) -> list[CompetitionFile]: ...
    def get_leaderboard(self, name: str, *, top_n: int = 10) -> list[LBEntry]: ...


# Common target column patterns. Match is case-insensitive against the column
# name lowercased, plus suffix/prefix variants for compound names.
_TARGET_EXACT: frozenset[str] = frozenset({
    "target", "label", "y", "outcome", "class",
    "survived", "saleprice",  # Kaggle staples
})
_TARGET_SUFFIXES: tuple[str, ...] = ("_target", "_label", "_y", "_outcome", "_class")
_TARGET_PREFIXES: tuple[str, ...] = ("target_", "label_", "y_")


def _looks_like_target(column_name: str) -> bool:
    cl = column_name.lower()
    if cl in _TARGET_EXACT:
        return True
    if any(cl.endswith(s) for s in _TARGET_SUFFIXES):
        return True
    return any(cl.startswith(p) for p in _TARGET_PREFIXES)


def build_context(
    *,
    workspace: Workspace,
    kaggle_client: _KaggleClientLike,
    leaderboard_top_n: int = 5,
) -> Path:
    name = workspace.name
    info = kaggle_client.view_competition(name)
    files = kaggle_client.list_files(name)
    leaderboard = kaggle_client.get_leaderboard(name, top_n=leaderboard_top_n)

    sections: list[str] = [
        f"# Competition: {info.title or name}",
        "",
        "## Description",
        info.description.strip() or "(no description available)",
        "",
        "## Evaluation metric",
        f"`{info.metric}`" if info.metric else "(metric not provided by Kaggle API)",
        "",
        "## Data files",
        _files_section(files),
        "",
        "## Data profile (train.csv)",
        _data_profile(workspace),
        "",
        "## Public leaderboard (top scores for reference)",
        _lb_section(leaderboard),
    ]

    body = "\n".join(sections) + "\n"
    workspace.context_path.write_text(body)
    return workspace.context_path


def _files_section(files: list[CompetitionFile]) -> str:
    if not files:
        return "(no files listed)"
    lines = []
    for f in files:
        size_mb = f.size / 1024 / 1024
        lines.append(f"- `{f.name}` ({size_mb:.1f} MB)")
    return "\n".join(lines)


def _lb_section(lb: list[LBEntry]) -> str:
    if not lb:
        return "(no leaderboard data available)"
    lines = ["| Rank | Team | Score |", "|---|---|---|"]
    for i, entry in enumerate(lb, start=1):
        lines.append(f"| {i} | {entry.team_name or '(redacted)'} | {entry.score} |")
    return "\n".join(lines)


def _data_profile(workspace: Workspace) -> str:
    train_csv = workspace.raw_dir / "train.csv"
    if not train_csv.exists():
        return "*Train data not yet downloaded (no train.csv in raw/).*"

    try:
        df = pd.read_csv(train_csv, nrows=5000)
    except Exception as e:  # noqa: BLE001
        return f"*Could not read train.csv: {e!r}*"

    target_candidates = [c for c in df.columns if _looks_like_target(c)]

    lines = [
        f"- **Rows (sampled, first 5000):** {len(df)}",
        f"- **Columns:** {len(df.columns)}",
    ]
    if target_candidates:
        lines.append(f"- **Likely target column(s):** {', '.join(f'`{c}`' for c in target_candidates)}")
    else:
        lines.append("- **Target column:** none of the standard names matched; the agent should infer.")
    lines.append("")
    lines.append("### Column schema")
    lines.append("| Column | dtype | non-null | unique |")
    lines.append("|---|---|---|---|")
    for col in df.columns:
        nn = df[col].notna().sum()
        nu = df[col].nunique()
        lines.append(f"| `{col}` | {df[col].dtype} | {nn}/{len(df)} | {nu} |")
    return "\n".join(lines)
