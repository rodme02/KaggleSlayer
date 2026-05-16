"""Append-only journaller for per-competition workspaces.

run_log.jsonl  — tool-call audit log, machine-readable
notes.jsonl    — agent's scratchpad, structured by category

Every append flushes and fsyncs before returning, so a crash immediately
after a `log_tool_call(...)` call does not lose the entry. This is the
durability property the spec §12 resume mechanism depends on.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from kaggle_slayer.harness.workspace import Workspace

NOTE_CATEGORIES: frozenset[str] = frozenset(
    {"observation", "decision", "hypothesis", "todo"}
)

# Cap per-arg string size when journalling tool calls. A 5KB write_file content
# is useful in the LLM transcript but only bloats the audit log. Anything
# longer is replaced by a marker referencing the original length.
_JOURNAL_ARG_CAP_CHARS: int = 2048


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


class Journal:
    """Append-only journaller bound to a Workspace."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    # --- run_log.jsonl ---

    def log_tool_call(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        result_summary: str,
        tool_call_id: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "ts": _now_iso(),
            "kind": "tool_call",
            "tool": tool,
            "args": self._summarize_args(args),
            "result_summary": result_summary,
        }
        if tool_call_id is not None:
            # F8: preserve the LLM-issued id so resume can replay function_call
            # / function_response pairs against stricter providers that won't
            # accept fabricated ids.
            record["tool_call_id"] = tool_call_id
        self._append(self.workspace.run_log_path, record)

    def log_tool_error(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        error: str,
        tool_call_id: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "ts": _now_iso(),
            "kind": "tool_error",
            "tool": tool,
            "args": self._summarize_args(args),
            "error": error,
        }
        if tool_call_id is not None:
            record["tool_call_id"] = tool_call_id
        self._append(self.workspace.run_log_path, record)

    @staticmethod
    def _summarize_args(args: dict[str, Any]) -> dict[str, Any]:
        """Return a shallow copy of args with oversized string values replaced
        by a short truncation marker. Non-string values are passed through
        unchanged — caps apply only to text payloads that explode the log."""
        summarized: dict[str, Any] = {}
        for k, v in args.items():
            if isinstance(v, str) and len(v) > _JOURNAL_ARG_CAP_CHARS:
                summarized[k] = f"<truncated, {len(v)} chars>"
            else:
                summarized[k] = v
        return summarized

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Yield every record from run_log.jsonl, in order.

        A trailing partial line (from a crash mid-write) is silently skipped —
        the spec §12 resume mechanism depends on this resilience.
        """
        path = self.workspace.run_log_path
        if not path.exists():
            return
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Partial trailing write from a crash; skip it.
                    continue

    # --- notes.jsonl ---

    def take_note(self, *, category: str, content: str) -> None:
        if category not in NOTE_CATEGORIES:
            raise ValueError(
                f"unknown category '{category}'; allowed: {sorted(NOTE_CATEGORIES)}"
            )
        self._append(
            self.workspace.notes_path,
            {"ts": _now_iso(), "category": category, "content": content},
        )

    def list_notes(self, *, category: str | None = None) -> list[dict[str, Any]]:
        path = self.workspace.notes_path
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if category is None or rec.get("category") == category:
                    records.append(rec)
        return records

    # --- internal: durable append ---

    @staticmethod
    def _append(path: Path, record: dict[str, Any]) -> None:
        """Append a JSON line and flush+fsync before returning."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
