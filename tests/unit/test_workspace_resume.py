"""Tests for kaggle_slayer.harness.resume."""

from __future__ import annotations

import pytest

from kaggle_slayer.harness import journal as journal_mod
from kaggle_slayer.harness import resume as resume_mod
from kaggle_slayer.harness.workspace import Workspace


@pytest.fixture
def populated_workspace(tmp_path):
    w = Workspace.create(root=tmp_path / "comp")
    j = journal_mod.Journal(w)
    j.log_tool_call(tool="load_competition", args={"name": "titanic"}, result_summary="loaded")
    j.log_tool_call(tool="profile_data", args={}, result_summary="891 rows, 12 cols")
    j.log_tool_error(tool="submit_kaggle", args={}, error="403 Forbidden")
    j.log_tool_call(tool="train_cv", args={}, result_summary="cv=0.823")
    return w


def test_resume_summary_empty_workspace(tmp_path):
    w = Workspace.create(root=tmp_path / "empty")
    summary = resume_mod.summarize(w)
    assert summary.total_calls == 0
    assert summary.tool_counts == {}
    assert summary.last_call is None
    assert summary.error_count == 0


def test_resume_summary_counts_per_tool(populated_workspace):
    summary = resume_mod.summarize(populated_workspace)
    assert summary.total_calls == 4
    assert summary.tool_counts == {
        "load_competition": 1,
        "profile_data": 1,
        "submit_kaggle": 1,
        "train_cv": 1,
    }
    assert summary.error_count == 1


def test_resume_summary_last_call(populated_workspace):
    summary = resume_mod.summarize(populated_workspace)
    assert summary.last_call is not None
    assert summary.last_call["tool"] == "train_cv"
    assert summary.last_call["kind"] == "tool_call"


def test_resume_summary_detects_stuck_loop(tmp_path):
    """5+ identical (tool, args) in a 10-call window indicates a stuck loop."""
    w = Workspace.create(root=tmp_path / "stuck")
    j = journal_mod.Journal(w)
    for _ in range(6):
        j.log_tool_call(tool="train_cv", args={"fe": "agent/fe.py"}, result_summary="failed")
    summary = resume_mod.summarize(w)
    assert summary.stuck_loop is not None
    assert summary.stuck_loop["tool"] == "train_cv"
    assert summary.stuck_loop["repeats"] >= 5


def test_resume_summary_no_stuck_loop_when_args_vary(tmp_path):
    w = Workspace.create(root=tmp_path / "ok")
    j = journal_mod.Journal(w)
    for i in range(6):
        j.log_tool_call(tool="train_cv", args={"fe": f"agent/fe_v{i}.py"}, result_summary=f"cv={0.8 + i * 0.01}")
    summary = resume_mod.summarize(w)
    assert summary.stuck_loop is None
