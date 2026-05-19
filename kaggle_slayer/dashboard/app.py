"""Streamlit entry: portfolio + comp-detail routing.

The dashboard reads only on-disk artifacts — never touches Kaggle or
Gemini. Pages live in `kaggle_slayer/dashboard/`.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _in_streamlit_runtime() -> bool:
    """True iff this module is being executed inside a running Streamlit script.

    Uses the public `st.runtime.exists()` API (streamlit 1.18+) instead of
    poking at the internal `scriptrunner_utils.script_run_context` module
    path, which has moved between streamlit versions. The except-fallback
    is retained as a safety net in case `runtime` is missing entirely on
    very old releases.
    """
    try:
        import streamlit as st  # type: ignore[import-untyped]  # noqa: PLC0415

        return bool(st.runtime.exists())
    except (ImportError, AttributeError):
        return False


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
    import streamlit as st  # type: ignore[import-untyped]  # noqa: PLC0415

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
