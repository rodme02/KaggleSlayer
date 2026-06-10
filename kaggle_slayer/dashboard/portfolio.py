"""Portfolio page — list of competitions with summary metrics.

Pure helpers (list_competitions, best_cv_for) are unit-tested. The
`render` function calls Streamlit and is exercised manually + via the
slow-tier integration test (Task 14).
"""

from __future__ import annotations

from pathlib import Path

from kaggle_slayer.agent.cost_ledger import DEFAULT_LEDGER_PATH, CostLedger
from kaggle_slayer.harness.telemetry import behavior
from kaggle_slayer.harness.workspace import Workspace


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
    """Best train_cv mean in the journal, metric-direction aware (min for
    rmse/mae/logloss, max otherwise). Delegates to telemetry.behavior so the
    summary parsing and direction logic live in exactly one place."""
    return behavior.compute_metrics(workspace).best_cv_mean


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
        metrics = behavior.compute_metrics(ws)
        cv = metrics.best_cv_mean
        cost = ledger.total_for(competition=name)

        with st.container(border=True):
            cols = st.columns([2, 1, 1, 1])
            cols[0].markdown(f"### `{name}`")
            cols[1].metric("Best CV", f"{cv:.4f}" if cv is not None else "—")
            cols[2].metric("Cost (USD)", f"${cost:.4f}")
            cols[3].metric("Tool calls", str(metrics.turns_per_run))
