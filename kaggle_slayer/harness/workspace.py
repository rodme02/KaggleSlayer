"""Per-competition workspace.

Spec §10 layout:

    competitions/<name>/
        raw/                  Kaggle download (gitignored)
        context.md            auto-generated brief
        agent/                ALL agent-written code
            fe.py
            model.py
            versions/         fe_v01.py, model_v01.py, ...
            scratch/          one-off scripts via run_python
        artifacts/            pipeline.pkl, oof_preds.npy, ...
        submissions/          dated CSVs + leaderboard.jsonl
        notes.jsonl           agent's scratchpad
        run_log.jsonl         tool-call audit log

A `Workspace` is just typed paths plus a couple of create/load helpers.
Journalling and resume live in separate modules (journal.py, resume.py)
so this file stays small and focused.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    """Typed view of a per-competition workspace directory."""

    root: Path

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def agent_dir(self) -> Path:
        return self.root / "agent"

    @property
    def versions_dir(self) -> Path:
        return self.agent_dir / "versions"

    @property
    def scratch_dir(self) -> Path:
        return self.agent_dir / "scratch"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def submissions_dir(self) -> Path:
        return self.root / "submissions"

    @property
    def context_path(self) -> Path:
        return self.root / "context.md"

    @property
    def fe_path(self) -> Path:
        return self.agent_dir / "fe.py"

    @property
    def model_path(self) -> Path:
        return self.agent_dir / "model.py"

    @property
    def run_log_path(self) -> Path:
        return self.root / "run_log.jsonl"

    @property
    def notes_path(self) -> Path:
        return self.root / "notes.jsonl"

    def next_version_path(self, kind: str) -> Path:
        """Return the next free path under versions/ for the given kind.

        kind: "fe" or "model". Scans versions/ for files matching
        `{kind}_v\\d+.py` and returns `versions/{kind}_v{N+1:02d}.py`.
        """
        if kind not in ("fe", "model"):
            raise ValueError(f"kind must be 'fe' or 'model', got {kind!r}")
        import re

        pattern = re.compile(rf"^{kind}_v(\d+)\.py$")
        existing: list[int] = []
        if self.versions_dir.is_dir():
            for f in self.versions_dir.iterdir():
                m = pattern.match(f.name)
                if m:
                    existing.append(int(m.group(1)))
        next_n = (max(existing) + 1) if existing else 1
        return self.versions_dir / f"{kind}_v{next_n:02d}.py"

    @classmethod
    def create(cls, root: Path) -> Workspace:
        """Create the workspace directory structure (idempotent)."""
        root = Path(root)
        for sub in (
            root,
            root / "raw",
            root / "agent",
            root / "agent" / "versions",
            root / "agent" / "scratch",
            root / "artifacts",
            root / "submissions",
        ):
            sub.mkdir(parents=True, exist_ok=True)
        return cls(root=root)

    @classmethod
    def load(cls, root: Path) -> Workspace:
        """Open an existing workspace; raises if not present."""
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"no workspace at {root}")
        return cls(root=root)
