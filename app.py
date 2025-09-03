# app.py
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
import sys

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

# ---------- UI ----------
st.title("Kaggle Slayer — Dashboard")

tab_comp, = st.tabs(["Competitions"])

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
