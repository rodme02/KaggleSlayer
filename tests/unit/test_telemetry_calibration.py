"""Tests for kaggle_slayer.harness.telemetry.calibration."""

from __future__ import annotations

import json

import pytest

from kaggle_slayer.harness.telemetry import calibration


@pytest.fixture
def isolated_calibration(tmp_path, monkeypatch):
    path = tmp_path / "calibration.jsonl"
    monkeypatch.setattr(calibration, "DEFAULT_PATH", path)
    return path


def test_record_calibration_appends_row(isolated_calibration):
    calibration.record(
        competition="titanic",
        cv_score=0.82,
        lb_score=None,
        problem_type="classification",
        metric="accuracy",
        cv_strategy="stratified_kfold",
    )
    lines = isolated_calibration.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["competition"] == "titanic"
    assert rec["cv_score"] == 0.82
    assert rec["lb_score"] is None
    assert rec["problem_type"] == "classification"
    assert rec["metric"] == "accuracy"
    assert rec["cv_strategy"] == "stratified_kfold"
    assert "ts" in rec


def test_read_history_returns_all_records_in_order(isolated_calibration):
    calibration.record(competition="a", cv_score=0.5, lb_score=None,
                       problem_type="regression", metric="rmse", cv_strategy="kfold")
    calibration.record(competition="b", cv_score=0.9, lb_score=0.88,
                       problem_type="classification", metric="auc", cv_strategy="stratified_kfold")
    history = calibration.read_history()
    assert len(history) == 2
    assert history[0]["competition"] == "a"
    assert history[1]["competition"] == "b"


def test_read_history_filters_by_competition(isolated_calibration):
    calibration.record(competition="a", cv_score=0.5, lb_score=None,
                       problem_type="regression", metric="rmse", cv_strategy="kfold")
    calibration.record(competition="b", cv_score=0.9, lb_score=0.88,
                       problem_type="classification", metric="auc", cv_strategy="stratified_kfold")
    only_a = calibration.read_history(competition="a")
    assert len(only_a) == 1
    assert only_a[0]["competition"] == "a"


def test_read_history_handles_missing_file(isolated_calibration):
    assert calibration.read_history() == []


def test_read_history_skips_malformed_lines(isolated_calibration):
    isolated_calibration.parent.mkdir(parents=True, exist_ok=True)
    with isolated_calibration.open("a") as f:
        f.write('{"competition": "a", "cv_score": 0.5}\n')
        f.write("not json at all\n")
        f.write('{"competition": "b", "cv_score": 0.8}\n')
    history = calibration.read_history()
    assert len(history) == 2
    assert history[0]["competition"] == "a"
    assert history[1]["competition"] == "b"
