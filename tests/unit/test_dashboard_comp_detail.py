"""Tests for the pure helpers in kaggle_slayer.dashboard.comp_detail."""

from __future__ import annotations

import pytest

from kaggle_slayer.dashboard import comp_detail
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


@pytest.fixture
def ws(tmp_path):
    return Workspace.create(root=tmp_path / "titanic")


def test_journal_timeline_returns_in_order(ws):
    j = Journal(ws)
    j.log_tool_call(
        tool="take_note",
        args={"category": "observation", "content": "x"},
        result_summary="noted",
    )
    j.log_tool_call(tool="train_cv", args={}, result_summary="mean=0.8")
    timeline = comp_detail.journal_timeline(ws)
    assert len(timeline) == 2
    assert timeline[0]["tool"] == "take_note"
    assert timeline[1]["tool"] == "train_cv"


def test_journal_timeline_empty_for_missing_log(ws):
    assert comp_detail.journal_timeline(ws) == []


def test_notes_browser_returns_filtered_categories(ws):
    j = Journal(ws)
    j.take_note(category="observation", content="A")
    j.take_note(category="decision", content="B")
    j.take_note(category="observation", content="C")
    notes = comp_detail.read_notes(ws)
    assert len(notes) == 3
    observations = comp_detail.read_notes(ws, category="observation")
    assert len(observations) == 2
    assert all(n["category"] == "observation" for n in observations)


def test_list_submissions_returns_csv_paths(ws):
    (ws.submissions_dir / "2026-05-17_v01.csv").write_text("id,target\n1,0\n")
    (ws.submissions_dir / "2026-05-17_v02.csv").write_text("id,target\n1,1\n")
    (ws.submissions_dir / "leaderboard.jsonl").write_text("")  # not a submission
    submissions = comp_detail.list_submissions(ws)
    assert len(submissions) == 2
    assert all(p.suffix == ".csv" for p in submissions)


def test_calibration_for_competition_filters_history(ws, tmp_path, monkeypatch):
    """The page reads from telemetry.calibration filtered by competition name."""
    from kaggle_slayer.harness.telemetry import calibration

    cal_path = tmp_path / "calibration.jsonl"
    monkeypatch.setattr(calibration, "DEFAULT_PATH", cal_path)
    calibration.record(
        competition="titanic",
        cv_score=0.82,
        lb_score=None,
        problem_type="classification",
        metric="accuracy",
        cv_strategy="stratified_kfold",
    )
    calibration.record(
        competition="other",
        cv_score=0.5,
        lb_score=None,
        problem_type="regression",
        metric="rmse",
        cv_strategy="kfold",
    )
    rows = comp_detail.calibration_for(ws)
    assert len(rows) == 1
    assert rows[0]["competition"] == "titanic"
