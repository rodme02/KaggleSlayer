"""Tests for kaggle_slayer.harness.telemetry.behavior."""

from __future__ import annotations

import pytest

from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.telemetry import behavior
from kaggle_slayer.harness.workspace import Workspace


@pytest.fixture
def ws(tmp_path):
    return Workspace.create(root=tmp_path / "comp")


def test_compute_metrics_counts_turns(ws):
    j = Journal(ws)
    j.log_tool_call(tool="take_note", args={}, result_summary="ok")
    j.log_tool_call(tool="train_cv", args={}, result_summary="ok")
    j.log_tool_call(tool="done", args={"summary": "x"}, result_summary="ack")

    metrics = behavior.compute_metrics(ws)
    assert metrics.turns_per_run == 3
    assert metrics.tool_counts == {"take_note": 1, "train_cv": 1, "done": 1}


def test_compute_metrics_turns_to_first_submission(ws):
    """The turn index where the FIRST submit_kaggle (or submit_local) lands."""
    j = Journal(ws)
    j.log_tool_call(tool="take_note", args={}, result_summary="ok")
    j.log_tool_call(tool="write_file", args={}, result_summary="ok")
    j.log_tool_call(tool="train_cv", args={}, result_summary="ok")
    j.log_tool_call(tool="submit_local", args={"label": "v1"}, result_summary="ok")  # 4th turn
    j.log_tool_call(tool="done", args={"summary": "x"}, result_summary="ack")

    metrics = behavior.compute_metrics(ws)
    assert metrics.turns_to_first_submission == 4


def test_compute_metrics_no_submission_yet(ws):
    j = Journal(ws)
    j.log_tool_call(tool="take_note", args={}, result_summary="ok")
    metrics = behavior.compute_metrics(ws)
    assert metrics.turns_to_first_submission is None


def test_detect_stuck_loop_flags_same_call_repeated(ws):
    """Five identical (tool, args) calls in the last 10 records means stuck."""
    j = Journal(ws)
    for _ in range(5):
        j.log_tool_call(
            tool="train_cv", args={}, result_summary="mean=0.5",
        )
    stuck = behavior.detect_stuck_loop(ws, window=10, threshold=5)
    assert stuck is not None
    assert stuck["tool"] == "train_cv"
    assert stuck["repeats"] == 5


def test_detect_stuck_loop_no_repetition_returns_none(ws):
    j = Journal(ws)
    for i in range(5):
        j.log_tool_call(
            tool="take_note", args={"category": "observation", "content": f"#{i}"},
            result_summary="noted",
        )
    stuck = behavior.detect_stuck_loop(ws, window=10, threshold=5)
    assert stuck is None
