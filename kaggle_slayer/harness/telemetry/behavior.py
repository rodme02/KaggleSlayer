"""Agent behavior metrics — derived from the journal.

turns_per_run, tool_counts, error_count, turns_to_first_submission,
stuck-loop detection. Pure functions over Journal.iter_records();
no side effects, no I/O beyond the journal read.

The dashboard reads these on demand; resume.summarize() delegates its
stuck-loop computation here.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace

_SUBMIT_TOOLS = frozenset({"submit_kaggle", "submit_local"})
# Same regex shape as dashboard.portfolio._MEAN_RE — parse the
# `mean=<float>` substring train_cv writes into result_summary.
_MEAN_RE = re.compile(r"mean=([0-9.]+)")


@dataclass
class BehaviorMetrics:
    turns_per_run: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    turns_to_first_submission: int | None = None
    turns_to_best_score: int | None = None
    tool_call_failure_rate: float = 0.0


def compute_metrics(workspace: Workspace) -> BehaviorMetrics:
    """Walk the journal once and compute all metrics."""
    records = list(Journal(workspace).iter_records())
    counts: Counter[str] = Counter()
    errors = 0
    first_submission_turn: int | None = None
    best_mean: float | None = None
    turns_to_best: int | None = None

    for i, rec in enumerate(records, start=1):
        kind = rec.get("kind")
        if kind not in ("tool_call", "tool_error"):
            continue
        tool = rec.get("tool", "")
        counts[tool] += 1
        if kind == "tool_error":
            errors += 1
        if first_submission_turn is None and tool in _SUBMIT_TOOLS:
            first_submission_turn = i
        if kind == "tool_call" and tool == "train_cv":
            summary = rec.get("result_summary", "")
            m = _MEAN_RE.search(summary)
            if m:
                try:
                    mean = float(m.group(1))
                except ValueError:
                    mean = None
                if mean is not None and (best_mean is None or mean > best_mean):
                    best_mean = mean
                    turns_to_best = i

    turns_total = sum(counts.values())
    failure_rate = errors / max(turns_total, 1)
    return BehaviorMetrics(
        turns_per_run=turns_total,
        tool_counts=dict(counts),
        error_count=errors,
        turns_to_first_submission=first_submission_turn,
        turns_to_best_score=turns_to_best,
        tool_call_failure_rate=failure_rate,
    )


def detect_stuck_loop(
    workspace: Workspace,
    *,
    window: int = 10,
    threshold: int = 5,
) -> dict[str, Any] | None:
    """If the same (tool, args) appears `threshold` times in the trailing
    window of `window` records, return a description dict; else None."""
    records = list(Journal(workspace).iter_records())
    window_records = records[-window:]
    sigs: Counter[tuple[str, str]] = Counter()
    for rec in window_records:
        kind = rec.get("kind")
        if kind not in ("tool_call", "tool_error"):
            continue
        sig = (rec.get("tool", ""), json.dumps(rec.get("args", {}), sort_keys=True))
        sigs[sig] += 1
    if not sigs:
        return None
    (tool, args_repr), count = sigs.most_common(1)[0]
    if count < threshold:
        return None
    return {
        "tool": tool,
        "args": json.loads(args_repr),
        "repeats": count,
        "window": window,
    }
