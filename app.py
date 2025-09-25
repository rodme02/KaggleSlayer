# app.py
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
import sys
import json
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional

st.set_page_config(page_title="Kaggle Slayer", layout="wide")

# ---------- Paths / Config ----------
DATA_DIR = Path(".")
COMP_CSV = DATA_DIR / "competition_data/competitions.csv"
BL_CSV = DATA_DIR / "competition_data/blacklist.csv"
HARVESTER = Path("agents/get_competitions.py")

DISPLAY_COLS = [
    "ref", "title", "category", "deadline", "reward", "has_access", "last_checked_at"
]

# ---------- Helpers ----------
@st.cache_data(ttl=10)
def load_competitions() -> pd.DataFrame:
    if not COMP_CSV.exists():
        return pd.DataFrame(columns=DISPLAY_COLS)
    df = pd.read_csv(COMP_CSV)
    for col in DISPLAY_COLS:
        if col not in df.columns:
            df[col] = ""
    # normalize booleans
    df["has_access"] = df["has_access"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    # ensure url
    if "url" not in df.columns:
        df["url"] = "https://www.kaggle.com/competitions/" + df["ref"].astype(str)
    else:
        df["url"] = df["url"].fillna("https://www.kaggle.com/competitions/" + df["ref"].astype(str))
    return df

def write_competitions(df: pd.DataFrame):
    # keep whatever other columns exist; just persist to CSV
    df.to_csv(COMP_CSV, index=False)

def append_blacklist(ref: str, reason: str = "manual_blacklist"):
    BL_CSV.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    row = {"ref": ref, "reason": reason, "timestamp": ts}
    if BL_CSV.exists():
        bl = pd.read_csv(BL_CSV)
        bl = pd.concat([bl, pd.DataFrame([row])], ignore_index=True)
    else:
        bl = pd.DataFrame([row])
    bl.to_csv(BL_CSV, index=False)

def run_harvester():
    cmd = [sys.executable, str(HARVESTER),
           "--out-csv", str(COMP_CSV),
           "--blacklist", str(BL_CSV),
           "--out-dir", "downloaded_datasets",
           "--max-pages", "-1"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return res.returncode, res.stdout, res.stderr
    except Exception as e:
        return 1, "", f"{e}"

def link_button(label: str, url: str, key: str):
    # Streamlit's native link_button works in newer versions; fallback to markdown link if needed
    try:
        st.link_button(label, url, key=key)
    except Exception:
        st.markdown(f"[{label}]({url})")

# ---------- Pipeline Helpers ----------
def get_downloaded_competitions():
    """Get list of downloaded competitions with their analysis status"""
    datasets_dir = Path("downloaded_datasets")
    if not datasets_dir.exists():
        return []

    competitions = []
    for comp_dir in datasets_dir.iterdir():
        if comp_dir.is_dir() and (comp_dir / "train.csv").exists():
            # Check analysis status
            scout_done = (comp_dir / "scout_output").exists()
            model_done = (comp_dir / "baseline_model").exists()
            submissions_exist = (comp_dir / "submissions").exists()

            competitions.append({
                "name": comp_dir.name,
                "path": comp_dir,
                "scout_done": scout_done,
                "model_done": model_done,
                "submissions_exist": submissions_exist
            })

    return competitions

def load_dataset_info(comp_path: Path) -> Optional[Dict]:
    """Load dataset info from scout analysis"""
    info_path = comp_path / "scout_output" / "dataset_info.json"
    if info_path.exists():
        with open(info_path, 'r') as f:
            return json.load(f)
    return None

def load_baseline_results(comp_path: Path) -> Optional[Dict]:
    """Load baseline model results"""
    results_path = comp_path / "baseline_model" / "baseline_results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    return None

def load_submission_history(comp_path: Path) -> List[Dict]:
    """Load submission history"""
    log_path = comp_path / "submissions" / "submission_log.jsonl"
    if not log_path.exists():
        return []

    history = []
    with open(log_path, 'r') as f:
        for line in f:
            if line.strip():
                history.append(json.loads(line.strip()))
    return history

def run_pipeline_step(comp_name: str, step: str, **kwargs):
    """Run a specific pipeline step"""
    comp_path = Path("downloaded_datasets") / comp_name

    if step == "scout":
        cmd = [sys.executable, "agents/data_scout.py", str(comp_path)]
    elif step == "model":
        cmd = [sys.executable, "agents/baseline_model.py", str(comp_path)]
    elif step == "submit":
        cmd = [sys.executable, "agents/submitter.py", str(comp_path), "--dry-run"]
        if kwargs.get("message"):
            cmd.extend(["--message", kwargs["message"]])
    elif step == "full_pipeline":
        cmd = [sys.executable, "run_pipeline.py", str(comp_path), "--dry-run"]
        if kwargs.get("message"):
            cmd.extend(["--message", kwargs["message"]])
    else:
        return 1, "", f"Unknown step: {step}"

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return res.returncode, res.stdout, res.stderr
    except Exception as e:
        return 1, "", f"{e}"

# ---------- UI ----------
st.title("Kaggle Slayer — Dashboard")

tab_comp, tab_pipeline, tab_analytics = st.tabs(["Competitions", "Pipeline Management", "Analytics & Results"])

with tab_comp:
    hdr_col1, hdr_col2 = st.columns([1, 3])
    with hdr_col1:
        st.subheader("Competitions")
    with hdr_col2:
        # Refresh runs your harvester script to update competitions.csv
        if st.button("🔄 Refresh table (run harvester)", type="primary", use_container_width=True):
            with st.spinner("Running harvester..."):
                code, out, err = run_harvester()
            if code == 0:
                st.success("Harvester finished.")
                st.cache_data.clear()  # clear cached CSV reads
            else:
                st.error("Harvester returned a non-zero exit code.")
                if err:
                    st.exception(err)
                elif out:
                    st.write(out)

    df = load_competitions()

    # Quick filters
    filt_col1, filt_col2, filt_col3 = st.columns([2, 1, 1])
    with filt_col1:
        q = st.text_input("Search (ref/title)", "")
    with filt_col2:
        cat_vals = ["(All)"] + sorted([c for c in df["category"].dropna().unique() if c != ""])
        pick_cat = st.selectbox("Category", cat_vals, index=0)
    with filt_col3:
        only_access = st.checkbox("Only hasAccess", value=False)

    # Apply filters
    show = df.copy()
    if q:
        ql = q.lower()
        show = show[show["ref"].astype(str).str.lower().str.contains(ql) |
                    show["title"].astype(str).str.lower().str.contains(ql)]
    if pick_cat != "(All)":
        show = show[show["category"] == pick_cat]
    if only_access:
        show = show[show["has_access"] == True]

    # Show table with actions
    if show.empty:
        st.info("No competitions to display.")
    else:
        # Headers
        cols = st.columns([1.1, 2.0, 1.2, 1.1, 0.8, 0.9, 1.2, 1.0, 1.0])
        headers = ["ref", "title", "category", "deadline", "reward", "hasAccess", "last checked at", "open/join", "blacklist"]
        for c, h in zip(cols, headers):
            c.markdown(f"**{h}**")

        # Rows
        for i, row in show.iterrows():
            c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns([1.1, 2.0, 1.2, 1.1, 0.8, 0.9, 1.2, 1.0, 1.0])
            with c1:
                st.write(row["ref"])
            with c2:
                st.write(row["title"])
            with c3:
                st.write(row["category"])
            with c4:
                st.write(row["deadline"])
            with c5:
                st.write(row["reward"])
            with c6:
                st.write("✅" if bool(row["has_access"]) else "—")
            with c7:
                st.write(row.get("last_checked_at", ""))

            # Open/Join link button
            url = row.get("url", f"https://www.kaggle.com/competitions/{row['ref']}")
            with c8:
                link_button("Open / Join", url, key=f"open_{row['ref']}")

            # Blacklist action
            with c9:
                if st.button("Blacklist", key=f"bl_{row['ref']}"):
                    # 1) Append to blacklist.csv
                    append_blacklist(row["ref"], reason="not_tabular")
                    # 2) Remove from competitions.csv and persist
                    new_df = df[df["ref"] != row["ref"]].copy()
                    write_competitions(new_df)
                    st.success(f"Blacklisted '{row['ref']}' and removed from table.")
                    st.cache_data.clear()
                    st.rerun()

# ---------- PIPELINE MANAGEMENT TAB ----------
with tab_pipeline:
    st.subheader("Pipeline Management")

    # Get downloaded competitions
    competitions = get_downloaded_competitions()

    if not competitions:
        st.info("No downloaded competitions found. Use the Competitions tab to download datasets first.")
    else:
        st.write(f"Found {len(competitions)} downloaded competitions:")

        # Pipeline status overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            scout_done = sum(1 for c in competitions if c["scout_done"])
            st.metric("Data Scout Complete", f"{scout_done}/{len(competitions)}")
        with col2:
            model_done = sum(1 for c in competitions if c["model_done"])
            st.metric("Baseline Models", f"{model_done}/{len(competitions)}")
        with col3:
            submissions = sum(1 for c in competitions if c["submissions_exist"])
            st.metric("With Submissions", f"{submissions}/{len(competitions)}")
        with col4:
            fully_done = sum(1 for c in competitions if c["scout_done"] and c["model_done"] and c["submissions_exist"])
            st.metric("Complete Pipelines", f"{fully_done}/{len(competitions)}")

        st.divider()

        # Individual competition management
        for comp in competitions:
            with st.expander(f"🏆 {comp['name'].replace('-', ' ').title()}", expanded=False):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write("**Status:**")
                    st.write(f"{'✅' if comp['scout_done'] else '❌'} Data Scout")
                    st.write(f"{'✅' if comp['model_done'] else '❌'} Baseline Model")
                    st.write(f"{'✅' if comp['submissions_exist'] else '❌'} Submissions")

                with col2:
                    st.write("**Quick Actions:**")

                    if st.button(f"🔍 Run Scout", key=f"scout_{comp['name']}", disabled=comp['scout_done']):
                        with st.spinner("Running Data Scout..."):
                            code, out, err = run_pipeline_step(comp["name"], "scout")
                        if code == 0:
                            st.success("Data Scout completed!")
                            st.rerun()
                        else:
                            st.error(f"Error: {err}")

                    if st.button(f"🤖 Train Model", key=f"model_{comp['name']}", disabled=not comp['scout_done']):
                        with st.spinner("Training baseline model..."):
                            code, out, err = run_pipeline_step(comp["name"], "model")
                        if code == 0:
                            st.success("Baseline model trained!")
                            st.rerun()
                        else:
                            st.error(f"Error: {err}")

                    if st.button(f"📤 Create Submission", key=f"submit_{comp['name']}", disabled=not comp['model_done']):
                        message = st.text_input(f"Submission message for {comp['name']}", "Baseline submission", key=f"msg_{comp['name']}")
                        with st.spinner("Creating submission..."):
                            code, out, err = run_pipeline_step(comp["name"], "submit", message=message)
                        if code == 0:
                            st.success("Submission created!")
                            st.rerun()
                        else:
                            st.error(f"Error: {err}")

                with col3:
                    st.write("**Full Pipeline:**")
                    pipeline_message = st.text_input(f"Pipeline message", "Complete pipeline run", key=f"pipeline_msg_{comp['name']}")

                    if st.button(f"🚀 Run Full Pipeline", key=f"full_{comp['name']}", type="primary"):
                        with st.spinner("Running complete pipeline..."):
                            code, out, err = run_pipeline_step(comp["name"], "full_pipeline", message=pipeline_message)
                        if code == 0:
                            st.success("Complete pipeline finished!")
                            st.rerun()
                        else:
                            st.error(f"Error: {err}")

                # Show recent activity
                if comp['scout_done'] or comp['model_done'] or comp['submissions_exist']:
                    st.write("**Recent Results:**")

                    # Dataset info preview
                    if comp['scout_done']:
                        dataset_info = load_dataset_info(comp["path"])
                        if dataset_info:
                            st.write(f"📊 Dataset: {dataset_info['total_rows']:,} rows × {dataset_info['total_columns']} columns")
                            st.write(f"🎯 Target: {dataset_info.get('target_column', 'Unknown')} ({dataset_info.get('target_type', 'Unknown')})")

                    # Model results preview
                    if comp['model_done']:
                        baseline_results = load_baseline_results(comp["path"])
                        if baseline_results:
                            st.write(f"🤖 Model: {baseline_results['model_type']}")
                            st.write(f"📈 CV Score: {baseline_results['cv_mean']:.4f} (±{baseline_results['cv_std']*2:.4f})")

                    # Submission count
                    if comp['submissions_exist']:
                        submission_history = load_submission_history(comp["path"])
                        st.write(f"📤 Submissions: {len(submission_history)}")

# ---------- ANALYTICS TAB ----------
with tab_analytics:
    st.subheader("Analytics & Results")

    competitions = get_downloaded_competitions()
    analyzed_comps = [c for c in competitions if c["scout_done"]]

    if not analyzed_comps:
        st.info("No analyzed competitions found. Run Data Scout on some competitions first.")
    else:
        # Competition selector
        comp_names = [c["name"] for c in analyzed_comps]
        selected_comp = st.selectbox("Select Competition", comp_names)

        if selected_comp:
            comp_path = Path("downloaded_datasets") / selected_comp

            # Load all available data
            dataset_info = load_dataset_info(comp_path)
            baseline_results = load_baseline_results(comp_path)
            submission_history = load_submission_history(comp_path)

            # Competition Overview
            st.write(f"## {selected_comp.replace('-', ' ').title()}")

            if dataset_info:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", f"{dataset_info['total_rows']:,}")
                with col2:
                    st.metric("Features", dataset_info['total_columns'])
                with col3:
                    st.metric("Target Column", dataset_info.get('target_column', 'Unknown'))
                with col4:
                    st.metric("Problem Type", dataset_info.get('target_type', 'Unknown').title())

            # Dataset Analysis
            if dataset_info:
                st.write("### 📊 Dataset Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    # Feature types pie chart
                    feature_types = dataset_info['feature_types']
                    type_counts = {}
                    for ftype in feature_types.values():
                        type_counts[ftype] = type_counts.get(ftype, 0) + 1

                    fig_pie = px.pie(
                        values=list(type_counts.values()),
                        names=list(type_counts.keys()),
                        title="Feature Types Distribution"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col2:
                    # Missing values bar chart
                    missing_data = [(k, v) for k, v in dataset_info['missing_percentages'].items() if v > 0]
                    if missing_data:
                        missing_data.sort(key=lambda x: x[1], reverse=True)

                        fig_bar = px.bar(
                            x=[x[1] for x in missing_data[:10]],
                            y=[x[0] for x in missing_data[:10]],
                            orientation='h',
                            title="Top 10 Features with Missing Values (%)",
                            labels={'x': 'Missing Percentage', 'y': 'Feature'}
                        )
                        fig_bar.update_layout(height=400)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.info("No missing values detected in the dataset!")

            # Model Performance
            if baseline_results:
                st.write("### 🤖 Baseline Model Performance")

                col1, col2 = st.columns(2)

                with col1:
                    # CV scores visualization
                    cv_scores = baseline_results['cv_scores']
                    fig_cv = go.Figure()
                    fig_cv.add_trace(go.Scatter(
                        x=list(range(1, len(cv_scores) + 1)),
                        y=cv_scores,
                        mode='lines+markers',
                        name='CV Score',
                        line=dict(color='blue', width=3),
                        marker=dict(size=8)
                    ))
                    fig_cv.add_hline(y=baseline_results['cv_mean'], line_dash="dash",
                                   annotation_text=f"Mean: {baseline_results['cv_mean']:.4f}")
                    fig_cv.update_layout(
                        title="Cross-Validation Scores",
                        xaxis_title="Fold",
                        yaxis_title="Score",
                        showlegend=False
                    )
                    st.plotly_chart(fig_cv, use_container_width=True)

                with col2:
                    # Feature importance
                    if baseline_results.get('feature_importance'):
                        importance_data = list(baseline_results['feature_importance'].items())
                        importance_data.sort(key=lambda x: abs(x[1]), reverse=True)

                        top_features = importance_data[:10]
                        fig_imp = px.bar(
                            x=[abs(x[1]) for x in top_features],
                            y=[x[0] for x in top_features],
                            orientation='h',
                            title="Top 10 Feature Importance",
                            labels={'x': 'Importance', 'y': 'Feature'}
                        )
                        fig_imp.update_layout(height=400)
                        st.plotly_chart(fig_imp, use_container_width=True)

                # Model details
                st.write("**Model Details:**")
                st.write(f"- **Algorithm:** {baseline_results['model_type']}")
                st.write(f"- **Features Used:** {len(baseline_results['features_used'])}")
                st.write(f"- **CV Mean:** {baseline_results['cv_mean']:.4f}")
                st.write(f"- **CV Std:** {baseline_results['cv_std']:.4f}")
                st.write(f"- **Training Time:** {baseline_results['training_timestamp']}")

            # Submission History
            if submission_history:
                st.write("### 📤 Submission History")

                # Create submission dataframe
                sub_df = pd.DataFrame(submission_history)
                sub_df['timestamp'] = pd.to_datetime(sub_df['submission_timestamp'])
                sub_df = sub_df.sort_values('timestamp', ascending=False)

                # Display submission table
                display_cols = ['timestamp', 'submission_status', 'submission_message', 'public_score']
                if 'public_score' in sub_df.columns:
                    sub_display = sub_df[display_cols].copy()
                    sub_display['timestamp'] = sub_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
                    st.dataframe(sub_display, use_container_width=True)

                # Submission timeline if we have scores
                if 'public_score' in sub_df.columns and sub_df['public_score'].notna().any():
                    scores_df = sub_df[sub_df['public_score'].notna()].copy()
                    if len(scores_df) > 0:
                        fig_timeline = px.line(
                            scores_df,
                            x='timestamp',
                            y='public_score',
                            title='Public Score Timeline',
                            markers=True
                        )
                        st.plotly_chart(fig_timeline, use_container_width=True)

            # Raw data download
            st.write("### 💾 Download Results")
            col1, col2, col3 = st.columns(3)

            with col1:
                if dataset_info and (comp_path / "scout_output" / "train_cleaned.csv").exists():
                    with open(comp_path / "scout_output" / "train_cleaned.csv", "rb") as f:
                        st.download_button(
                            "📊 Download Cleaned Data",
                            data=f.read(),
                            file_name=f"{selected_comp}_cleaned.csv",
                            mime="text/csv"
                        )

            with col2:
                if baseline_results and (comp_path / "baseline_model" / "predictions.csv").exists():
                    with open(comp_path / "baseline_model" / "predictions.csv", "rb") as f:
                        st.download_button(
                            "🤖 Download Predictions",
                            data=f.read(),
                            file_name=f"{selected_comp}_predictions.csv",
                            mime="text/csv"
                        )

            with col3:
                if (comp_path / "submissions" / "latest_submission.csv").exists():
                    with open(comp_path / "submissions" / "latest_submission.csv", "rb") as f:
                        st.download_button(
                            "📤 Download Latest Submission",
                            data=f.read(),
                            file_name=f"{selected_comp}_submission.csv",
                            mime="text/csv"
                        )
