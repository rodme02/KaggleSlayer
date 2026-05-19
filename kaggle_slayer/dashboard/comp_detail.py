"""Competition-detail page: timeline, cost, calibration, notes, submissions.

Pure helpers are unit-tested. `render` calls Streamlit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggle_slayer.agent.cost_ledger import DEFAULT_LEDGER_PATH, CostLedger
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.telemetry import behavior, calibration
from kaggle_slayer.harness.workspace import Workspace


def journal_timeline(workspace: Workspace) -> list[dict[str, Any]]:
    """Every journal record in order."""
    return list(Journal(workspace).iter_records())


def read_notes(
    workspace: Workspace, *, category: str | None = None
) -> list[dict[str, Any]]:
    """All notes from notes.jsonl, optionally filtered by category."""
    return Journal(workspace).list_notes(category=category)


def list_submissions(workspace: Workspace) -> list[Path]:
    """All submission CSVs (excludes leaderboard.jsonl etc)."""
    if not workspace.submissions_dir.is_dir():
        return []
    return sorted(workspace.submissions_dir.glob("*.csv"))


def calibration_for(workspace: Workspace) -> list[dict[str, Any]]:
    """Calibration history filtered by competition name."""
    return calibration.read_history(competition=workspace.name)


def render(workspace_root: Path) -> None:
    """Streamlit page: per-comp detail."""
    import streamlit as st  # type: ignore[import-untyped]

    if not workspace_root.is_dir():
        st.error(f"path does not exist: `{workspace_root}`")
        return
    workspace = Workspace(root=workspace_root)
    st.title(f"Competition · `{workspace.name}`")

    # Top metrics
    ledger = CostLedger(path=DEFAULT_LEDGER_PATH)
    cost = ledger.total_for(competition=workspace.name)
    timeline = journal_timeline(workspace)
    cal_rows = calibration_for(workspace)
    subs = list_submissions(workspace)
    cols = st.columns(4)
    cols[0].metric("Tool calls", len(timeline))
    cols[1].metric("Cost (USD)", f"${cost:.4f}")
    cols[2].metric("Submissions", len(cal_rows))
    cols[3].metric("Submissions on disk", len(subs))

    # Behavior metrics (spec §11.2)
    metrics = behavior.compute_metrics(workspace)
    behav_cols = st.columns(4)
    behav_cols[0].metric("Turns / run", metrics.turns_per_run)
    behav_cols[1].metric(
        "Turns to first submission",
        metrics.turns_to_first_submission
        if metrics.turns_to_first_submission is not None
        else "—",
    )
    behav_cols[2].metric(
        "Turns to best CV",
        metrics.turns_to_best_score
        if metrics.turns_to_best_score is not None
        else "—",
    )
    behav_cols[3].metric(
        "Tool-call failure rate", f"{metrics.tool_call_failure_rate:.1%}"
    )

    # Timeline (table)
    st.subheader("Tool-call timeline")
    if timeline:
        import pandas as pd  # type: ignore[import-untyped]

        df = pd.DataFrame(
            [
                {
                    "ts": r.get("ts", ""),
                    "kind": r.get("kind", ""),
                    "tool": r.get("tool", ""),
                    "args_keys": ", ".join(r.get("args", {}).keys()),
                    "summary": (r.get("result_summary") or r.get("error") or "")[
                        :120
                    ],
                }
                for r in timeline
            ]
        )
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No journal records yet.")

    # Calibration table
    st.subheader("CV ↔ LB calibration")
    if cal_rows:
        st.dataframe(cal_rows, use_container_width=True)
    else:
        st.info("No submissions recorded yet.")

    # Notes
    st.subheader("Agent notes")
    notes = read_notes(workspace)
    if notes:
        st.dataframe(notes, use_container_width=True)
    else:
        st.info("Agent hasn't taken any notes yet.")

    # Submission downloads
    st.subheader("Submissions on disk")
    if subs:
        for p in subs:
            with p.open("rb") as f:
                st.download_button(
                    label=f"Download {p.name}",
                    data=f,
                    file_name=p.name,
                    mime="text/csv",
                )
    else:
        st.info("No submission CSVs yet.")
