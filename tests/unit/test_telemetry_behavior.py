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


def test_compute_metrics_error_count_positive(ws):
    """A logged tool_error bumps error_count by one."""
    j = Journal(ws)
    j.log_tool_call(tool="take_note", args={}, result_summary="ok")
    j.log_tool_error(tool="train_cv", args={}, error="kaboom")
    j.log_tool_call(tool="done", args={"summary": "x"}, result_summary="ack")

    metrics = behavior.compute_metrics(ws)
    assert metrics.error_count == 1


def test_compute_metrics_error_count_zero_when_only_successes(ws):
    """Journals with no tool_error records report zero errors."""
    j = Journal(ws)
    j.log_tool_call(tool="take_note", args={}, result_summary="ok")
    j.log_tool_call(tool="train_cv", args={}, result_summary="mean=0.5")

    metrics = behavior.compute_metrics(ws)
    assert metrics.error_count == 0


def test_compute_metrics_turns_to_best_score_picks_first_max(ws):
    """When several train_cv calls vary in mean, the turn index of the
    first call that reached the running max wins."""
    j = Journal(ws)
    j.log_tool_call(tool="take_note", args={}, result_summary="ok")            # turn 1
    j.log_tool_call(tool="train_cv", args={}, result_summary="mean=0.50, std=0.01")  # turn 2
    j.log_tool_call(tool="train_cv", args={}, result_summary="mean=0.80, std=0.01")  # turn 3 -- best
    j.log_tool_call(tool="train_cv", args={}, result_summary="mean=0.70, std=0.01")  # turn 4

    metrics = behavior.compute_metrics(ws)
    assert metrics.turns_to_best_score == 3


def test_compute_metrics_turns_to_best_score_lower_is_better(ws):
    """For rmse (higher_is_better=False) the best train_cv is the LOWEST mean."""
    j = Journal(ws)
    j.log_tool_call(tool="train_cv", args={}, result_summary="metric=rmse, mean=0.50, std=0.01")  # turn 1
    j.log_tool_call(tool="train_cv", args={}, result_summary="metric=rmse, mean=0.30, std=0.01")  # turn 2 — best
    j.log_tool_call(tool="train_cv", args={}, result_summary="metric=rmse, mean=0.40, std=0.01")  # turn 3

    metrics = behavior.compute_metrics(ws)
    assert metrics.turns_to_best_score == 2
    assert metrics.best_cv_mean == pytest.approx(0.30)


def test_compute_metrics_parses_negative_mean(ws):
    """r2 can be negative; a mean=-0.10 summary must be tracked, not dropped."""
    j = Journal(ws)
    j.log_tool_call(tool="train_cv", args={}, result_summary="metric=r2, mean=-0.10, std=0.01")
    metrics = behavior.compute_metrics(ws)
    assert metrics.turns_to_best_score == 1
    assert metrics.best_cv_mean == pytest.approx(-0.10)


def test_compute_metrics_best_cv_defaults_to_higher_is_better(ws):
    """Summaries without a metric= segment keep the historical max semantics."""
    j = Journal(ws)
    j.log_tool_call(tool="train_cv", args={}, result_summary="mean=0.50, std=0.01")
    j.log_tool_call(tool="train_cv", args={}, result_summary="mean=0.80, std=0.01")
    metrics = behavior.compute_metrics(ws)
    assert metrics.best_cv_mean == pytest.approx(0.80)
    assert metrics.turns_to_best_score == 2


def test_compute_metrics_turns_to_best_score_none_when_no_train_cv(ws):
    j = Journal(ws)
    j.log_tool_call(tool="take_note", args={}, result_summary="ok")
    metrics = behavior.compute_metrics(ws)
    assert metrics.turns_to_best_score is None


def test_compute_metrics_tool_call_failure_rate(ws):
    """failure_rate = errors / total_turns, bounded [0, 1]."""
    j = Journal(ws)
    j.log_tool_call(tool="take_note", args={}, result_summary="ok")
    j.log_tool_error(tool="train_cv", args={}, error="kaboom")
    j.log_tool_call(tool="train_cv", args={}, result_summary="mean=0.5")

    metrics = behavior.compute_metrics(ws)
    assert metrics.turns_per_run == 3
    assert metrics.error_count == 1
    assert metrics.tool_call_failure_rate == pytest.approx(1 / 3)


def test_compute_metrics_tool_call_failure_rate_zero_when_no_turns(ws):
    """No journal => no division-by-zero, just 0.0."""
    metrics = behavior.compute_metrics(ws)
    assert metrics.turns_per_run == 0
    assert metrics.tool_call_failure_rate == 0.0


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
