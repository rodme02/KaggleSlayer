"""Tests for kaggle_slayer.dashboard.app's pure helpers.

`main` / `_run_pages` drive Streamlit and are exercised manually; the
argv construction for the re-exec is pure and tested here.
"""

from __future__ import annotations

from pathlib import Path

from kaggle_slayer.dashboard import app


def test_streamlit_argv_forwards_user_args():
    """`kaggle-slayer-dashboard --server.port 8502` must reach streamlit
    instead of being silently dropped on re-exec."""
    argv = app._streamlit_argv(Path("/x/app.py"), ["--server.port", "8502"])
    assert argv == ["streamlit", "run", "/x/app.py", "--server.port", "8502"]


def test_streamlit_argv_without_user_args():
    argv = app._streamlit_argv(Path("/x/app.py"), [])
    assert argv == ["streamlit", "run", "/x/app.py"]
