"""Workspace resume / inspection helpers.

For Week 2 this is a read-only summary: count tool calls, detect stuck
loops, return the last call. Week 4 will extend it to rebuild the
Solver's conversation history so an aborted run can pick up where it
left off.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


@dataclass
class ResumeSummary:
    total_calls: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    last_call: dict[str, Any] | None = None
    stuck_loop: dict[str, Any] | None = None


def summarize(workspace: Workspace, *, stuck_window: int = 10, stuck_threshold: int = 5) -> ResumeSummary:
    """Read run_log.jsonl and return a high-level summary.

    stuck_loop detection: if the same (tool, args) tuple appears
    `stuck_threshold` times within the last `stuck_window` calls, flag it.
    """
    j = Journal(workspace)
    records = list(j.iter_records())

    summary = ResumeSummary(total_calls=len(records))
    if not records:
        return summary

    counts: Counter[str] = Counter()
    for r in records:
        counts[r["tool"]] += 1
        if r["kind"] == "tool_error":
            summary.error_count += 1
    summary.tool_counts = dict(counts)
    summary.last_call = records[-1]

    # Stuck loop: tally (tool, hash(args)) over the trailing window
    window = records[-stuck_window:]
    sigs: Counter[tuple[str, str]] = Counter()
    for r in window:
        sig = (r["tool"], json.dumps(r.get("args", {}), sort_keys=True))
        sigs[sig] += 1
    for (tool, args_repr), count in sigs.most_common(1):
        if count >= stuck_threshold:
            summary.stuck_loop = {
                "tool": tool,
                "args": json.loads(args_repr),
                "repeats": count,
                "window": stuck_window,
            }
    return summary
