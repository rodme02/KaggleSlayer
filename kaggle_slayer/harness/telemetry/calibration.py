"""CV-LB calibration tracker.

Every successful `submit_kaggle` writes one row here:

  {"ts": "...", "competition": "titanic", "cv_score": 0.82,
   "lb_score": null, "problem_type": "classification",
   "metric": "accuracy", "cv_strategy": "stratified_kfold"}

`lb_score` is `null` at write time (we don't have the real LB number
until Kaggle scores the submission). Week 6 will add a periodic
backfill task that polls `kaggle_client.get_leaderboard` and updates
the matching row.

The file is `~/.kaggle_slayer/calibration.jsonl` by default — a global
log across all competitions, so the dashboard's cross-comp calibration
chart has one place to read from.

This module is single-process / single-threaded; concurrent writers
would interleave records. Solver runs are serial, so this is fine.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path.home() / ".kaggle_slayer" / "calibration.jsonl"


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def record(
    *,
    competition: str,
    cv_score: float,
    lb_score: float | None,
    problem_type: str,
    metric: str,
    cv_strategy: str,
    path: Path | None = None,
) -> None:
    """Append one calibration row."""
    p = Path(path) if path is not None else DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": _now_iso(),
        "competition": competition,
        "cv_score": cv_score,
        "lb_score": lb_score,
        "problem_type": problem_type,
        "metric": metric,
        "cv_strategy": cv_strategy,
    }
    with p.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_history(
    *,
    competition: str | None = None,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Read all calibration rows; optionally filter by competition."""
    p = Path(path) if path is not None else DEFAULT_PATH
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    with p.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if competition is None or rec.get("competition") == competition:
                out.append(rec)
    return out
