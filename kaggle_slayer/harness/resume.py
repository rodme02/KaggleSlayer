"""Workspace resume / inspection helpers.

Week 2 added a read-only summary: count tool calls, detect stuck loops,
return the last call. Week 4 extended this with `rebuild_conversation`,
which replays `run_log.jsonl` as the Message history the Solver
originally sent — so an aborted run can be resumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from kaggle_slayer.agent.llm_client import Message, ToolCall
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
    # Counting lives in telemetry.behavior (single source for journal
    # walking); checkpoint records carry no 'tool' key and are excluded.
    from kaggle_slayer.harness.telemetry import behavior  # noqa: PLC0415

    metrics = behavior.compute_metrics(workspace)
    summary = ResumeSummary(
        total_calls=metrics.turns_per_run,
        tool_counts=metrics.tool_counts,
        error_count=metrics.error_count,
    )

    records = list(Journal(workspace).iter_records())
    summary.last_call = next(
        (r for r in reversed(records) if r.get("kind") in ("tool_call", "tool_error")),
        None,
    )
    if records:
        summary.stuck_loop = behavior.detect_stuck_loop(
            workspace, window=stuck_window, threshold=stuck_threshold,
        )
    return summary


class ResumeError(Exception):
    """Raised when the journal is in a state from which we cannot resume."""


def rebuild_conversation(workspace: Workspace) -> list[Message]:
    """Replay run_log.jsonl as the Message history the Solver originally sent.

    For each tool_call / tool_error record:
      - emit a model(role) Message with that call's tool_calls=[ToolCall(...)]
      - emit a tool(role) Message with the result (or the error string)

    checkpoint records are ignored (they're not part of the LLM conversation).
    Raises ResumeError if the last tool_call was 'done' (workspace finished).
    """
    j = Journal(workspace)
    records = list(j.iter_records())
    if not records:
        return []

    # Look at the last tool_call (skipping checkpoint records) — if it's `done`,
    # the run is finished and there's nothing to resume.
    last_tool_record = next(
        (r for r in reversed(records) if r.get("kind") in ("tool_call", "tool_error")),
        None,
    )
    if last_tool_record is not None and last_tool_record.get("tool") == "done":
        # F15: 'done' could appear as either tool_call or tool_error, so the
        # wording is "last tool reference" rather than "last tool call".
        raise ResumeError(
            "workspace already finished (last tool reference was 'done'); "
            "delete run_log.jsonl to start fresh"
        )

    messages: list[Message] = []
    for rec in records:
        kind = rec.get("kind")
        if kind not in ("tool_call", "tool_error"):
            continue  # checkpoint records aren't part of the LLM conversation
        tool_name = rec.get("tool", "unknown")
        # F8: prefer the original LLM-issued id when the journal stored one;
        # fall back to the synthetic resume_<n> for back-compat with journals
        # written before tool_call_id was journalled.
        tc_id = rec.get("tool_call_id") or f"resume_{len(messages)}"
        tc = ToolCall(id=tc_id, name=tool_name, args=rec.get("args", {}))
        messages.append(Message(role="model", content="", tool_calls=[tc]))
        result = (
            rec.get("result_summary", "") if kind == "tool_call" else rec.get("error", "")
        )
        payload = json.dumps({"tool": tool_name, "result": result})
        messages.append(Message(role="tool", content=payload))
    return messages
