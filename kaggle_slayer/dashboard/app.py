"""Streamlit entry: portfolio + comp-detail routing.

The dashboard reads only on-disk artifacts — never touches Kaggle or
Gemini. Pages live in `kaggle_slayer/dashboard/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st  # type: ignore[import-untyped]


def _in_streamlit_runtime() -> bool:
    """True iff this module is being executed inside a running Streamlit script.

    We can't just check `"streamlit.runtime.scriptrunner" in sys.modules`,
    because importing streamlit alone loads that submodule. Instead, ask
    streamlit whether the current thread has an active ScriptRunContext.
    """
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import (  # type: ignore[import-untyped]
            get_script_run_ctx,
        )
    except ImportError:  # older/newer streamlit layouts
        return False
    return get_script_run_ctx() is not None


def main() -> None:
    """Entry point for `kaggle-slayer-dashboard`."""
    # When invoked as a console_script (not via `streamlit run`), re-exec under
    # streamlit so the user gets the browser UI.
    if not _in_streamlit_runtime():
        import streamlit.web.cli as stcli  # type: ignore[import-untyped]
        sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
        sys.exit(stcli.main())
    _run_pages()


def _run_pages() -> None:
    # Lazy imports so the module can be imported without streamlit page state.
    from kaggle_slayer.dashboard import comp_detail, portfolio

    st.set_page_config(page_title="KaggleSlayer", layout="wide")
    st.sidebar.title("KaggleSlayer")
    page = st.sidebar.radio("Page", ["Portfolio", "Competition detail"])
    comps_root = Path(
        st.sidebar.text_input("Competitions root", value="competitions")
    )
    if page == "Portfolio":
        portfolio.render(comps_root)
    else:
        portfolio_names = portfolio.list_competitions(comps_root)
        if not portfolio_names:
            st.warning(f"No competitions found under {comps_root}.")
            return
        chosen = st.sidebar.selectbox("Competition", portfolio_names)
        comp_detail.render(comps_root / chosen)


# When `streamlit run kaggle_slayer/dashboard/app.py` is used directly,
# the module is imported and we land here.
if _in_streamlit_runtime():
    _run_pages()
