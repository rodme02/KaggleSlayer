"""Portfolio page — list of competitions with summary metrics.

Pure helpers (list_competitions, best_cv_for) are unit-tested. The
`render` function calls Streamlit and is exercised manually + via the
slow-tier integration test (Task 14).
"""

from __future__ import annotations

import re
from pathlib import Path

from kaggle_slayer.agent.cost_ledger import DEFAULT_LEDGER_PATH, CostLedger
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace

_MEAN_RE = re.compile(r"mean=([0-9.]+)")


def list_competitions(comps_root: Path) -> list[str]:
    """Return sorted names of competition workspaces under `comps_root`."""
    comps_root = Path(comps_root)
    if not comps_root.is_dir():
        return []
    names: list[str] = []
    for child in comps_root.iterdir():
        if not child.is_dir():
            continue
        if (child / "agent").is_dir():
            names.append(child.name)
    return sorted(names)


def best_cv_for(workspace: Workspace) -> float | None:
    """Walk the journal, extract `mean=<float>` from train_cv result_summary
    lines, return the max."""
    j = Journal(workspace)
    best: float | None = None
    for rec in j.iter_records():
        if rec.get("tool") != "train_cv" or rec.get("kind") != "tool_call":
            continue
        summary = rec.get("result_summary", "")
        m = _MEAN_RE.search(summary)
        if not m:
            continue
        try:
            mean = float(m.group(1))
        except ValueError:
            continue
        if best is None or mean > best:
            best = mean
    return best


def render(comps_root: Path) -> None:
    """Streamlit page: card per competition."""
    import streamlit as st  # type: ignore[import-untyped]

    st.title("Portfolio")
    names = list_competitions(comps_root)
    if not names:
        st.info(
            f"No competitions found under `{comps_root}`. Run "
            f"`kaggle-slayer <path>` to create one."
        )
        return

    ledger = CostLedger(path=DEFAULT_LEDGER_PATH)
    for name in names:
        ws = Workspace(root=comps_root / name)
        cv = best_cv_for(ws)
        cost = ledger.total_for(competition=name)
        run_log = ws.run_log_path

        with st.container(border=True):
            cols = st.columns([2, 1, 1, 1])
            cols[0].markdown(f"### `{name}`")
            cols[1].metric("Best CV", f"{cv:.4f}" if cv is not None else "—")
            cols[2].metric("Cost (USD)", f"${cost:.4f}")
            tool_count = (
                str(len(list(Journal(ws).iter_records())))
                if run_log.exists() else "0"
            )
            cols[3].metric("Tool calls", tool_count)
