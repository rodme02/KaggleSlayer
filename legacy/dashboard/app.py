"""KaggleSlayer dashboard.

Reads MLflow runs from the local file store (or MLFLOW_TRACKING_URI) and
renders a leaderboard, per-run drilldown, and submission preview. Run with:

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import pandas as pd
import plotly.express as px
import streamlit as st

EXPERIMENT_PREFIX = "kaggle-slayer/"


def _configure() -> None:
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return
    repo_root = Path(__file__).resolve().parent.parent
    mlflow.set_tracking_uri(f"file:{repo_root / 'mlruns'}")


@st.cache_data(ttl=30)
def load_runs() -> pd.DataFrame:
    """Pull every kaggle-slayer experiment into a single tidy frame."""
    client = mlflow.tracking.MlflowClient()
    rows: list[dict] = []
    for exp in client.search_experiments():
        if not exp.name.startswith(EXPERIMENT_PREFIX):
            continue
        competition = exp.name.removeprefix(EXPERIMENT_PREFIX)
        for run in client.search_runs(exp.experiment_id, max_results=500):
            rows.append(
                {
                    "competition": competition,
                    "run_id": run.info.run_id,
                    "run_name": run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
                    "status": run.data.tags.get("status", run.info.status),
                    "best_model": run.data.tags.get("best_model", "?"),
                    "problem_type": run.data.tags.get("problem_type", "?"),
                    "cv_score": run.data.metrics.get("cv_score"),
                    "duration_seconds": run.data.metrics.get("duration_seconds"),
                    "start_time": pd.to_datetime(run.info.start_time, unit="ms"),
                }
            )
    return pd.DataFrame(rows)


def render_leaderboard(df: pd.DataFrame) -> None:
    st.subheader("Leaderboard")
    if df.empty:
        st.info("No runs yet. Run `kaggle-slayer <competition> --data-path ...` to generate one.")
        return

    cols = st.columns(3)
    cols[0].metric("Total runs", len(df))
    cols[1].metric("Competitions", df["competition"].nunique())
    cols[2].metric("Best CV", f"{df['cv_score'].max():.4f}" if df["cv_score"].notna().any() else "—")

    chart = px.bar(
        df.sort_values("cv_score", ascending=False).head(20),
        x="run_name",
        y="cv_score",
        color="competition",
        hover_data=["best_model", "problem_type", "duration_seconds"],
        title="Top 20 runs by CV score",
    )
    chart.update_layout(xaxis_tickangle=-30, height=420)
    st.plotly_chart(chart, use_container_width=True)

    st.dataframe(
        df.sort_values("start_time", ascending=False)[
            ["competition", "run_name", "best_model", "problem_type", "cv_score", "duration_seconds", "status", "start_time"]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_run_detail(df: pd.DataFrame) -> None:
    st.subheader("Run drilldown")
    if df.empty:
        return
    pick = st.selectbox(
        "Select a run",
        df.sort_values("start_time", ascending=False)["run_id"].tolist(),
        format_func=lambda rid: _label_for(df, rid),
    )
    if not pick:
        return

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(pick)

    a, b = st.columns(2)
    with a:
        st.markdown("**Tags**")
        st.json(run.data.tags)
    with b:
        st.markdown("**Metrics**")
        st.json(run.data.metrics)

    st.markdown("**Params**")
    st.json(run.data.params)

    artifact_root = Path(run.info.artifact_uri.removeprefix("file://"))
    submission = artifact_root / "submission.csv"
    if submission.exists():
        st.markdown("**Submission preview**")
        st.dataframe(pd.read_csv(submission).head(50), use_container_width=True, hide_index=True)
        st.download_button("Download submission.csv", submission.read_bytes(), file_name="submission.csv")


def _label_for(df: pd.DataFrame, run_id: str) -> str:
    row = df[df["run_id"] == run_id].iloc[0]
    score = f"{row['cv_score']:.4f}" if pd.notna(row["cv_score"]) else "—"
    return f"{row['competition']} · {row['run_name']} · {row['best_model']} · CV={score}"


def main() -> None:
    st.set_page_config(page_title="KaggleSlayer", page_icon="🏆", layout="wide")
    _configure()
    st.title("KaggleSlayer")
    st.caption("AutoML for tabular Kaggle competitions — leak-free CV, MLflow-tracked.")

    df = load_runs()
    tab1, tab2 = st.tabs(["Leaderboard", "Run detail"])
    with tab1:
        render_leaderboard(df)
    with tab2:
        render_run_detail(df)


if __name__ == "__main__":
    main()
