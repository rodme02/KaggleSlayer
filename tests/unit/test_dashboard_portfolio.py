"""Tests for kaggle_slayer.dashboard.portfolio.

The Streamlit `render` function calls live `st.*` APIs; we test the pure
helpers (list_competitions + best_cv_for) directly.
"""

from __future__ import annotations

import pytest

from kaggle_slayer.dashboard import portfolio
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


@pytest.fixture
def comps_root(tmp_path):
    return tmp_path / "competitions"


def test_list_competitions_returns_empty_for_missing_root(comps_root):
    assert portfolio.list_competitions(comps_root) == []


def test_list_competitions_returns_workspace_dirs(comps_root):
    comps_root.mkdir(parents=True)
    Workspace.create(root=comps_root / "titanic")
    Workspace.create(root=comps_root / "house-prices")
    names = portfolio.list_competitions(comps_root)
    assert sorted(names) == ["house-prices", "titanic"]


def test_list_competitions_skips_non_workspaces(comps_root):
    """A directory missing `agent/` is not a workspace."""
    comps_root.mkdir(parents=True)
    Workspace.create(root=comps_root / "titanic")
    (comps_root / "not-a-workspace").mkdir()
    (comps_root / "not-a-workspace" / "random.txt").write_text("x")
    names = portfolio.list_competitions(comps_root)
    assert names == ["titanic"]


def test_best_cv_for_reads_from_journal(comps_root):
    comps_root.mkdir(parents=True)
    ws = Workspace.create(root=comps_root / "titanic")
    j = Journal(ws)
    j.log_tool_call(tool="train_cv", args={}, result_summary=(
        "train_cv complete: stratified_kfold (5 folds), metric=accuracy, "
        "mean=0.7800, std=0.02, fold_scores=[0.78, 0.76, 0.80, 0.79, 0.77], "
        "duration_s=0.50"
    ))
    j.log_tool_call(tool="train_cv", args={}, result_summary=(
        "train_cv complete: stratified_kfold (5 folds), metric=accuracy, "
        "mean=0.8500, std=0.01, fold_scores=[0.85, 0.84, 0.86, 0.85, 0.85], "
        "duration_s=0.50"
    ))
    best = portfolio.best_cv_for(ws)
    assert best is not None
    assert abs(best - 0.85) < 1e-6


def test_best_cv_for_lower_is_better_metric(comps_root):
    """For rmse the best CV is the minimum mean, not the maximum."""
    comps_root.mkdir(parents=True)
    ws = Workspace.create(root=comps_root / "house-prices")
    j = Journal(ws)
    for mean in ("0.7800", "0.4500", "0.6000"):
        j.log_tool_call(tool="train_cv", args={}, result_summary=(
            f"train_cv complete: kfold (5 folds), metric=rmse, "
            f"mean={mean}, std=0.02, fold_scores=[0.78], duration_s=0.50"
        ))
    best = portfolio.best_cv_for(ws)
    assert best == pytest.approx(0.45)


def test_best_cv_for_parses_negative_mean(comps_root):
    """r2 can go negative; such runs must not be silently excluded."""
    comps_root.mkdir(parents=True)
    ws = Workspace.create(root=comps_root / "r2-comp")
    Journal(ws).log_tool_call(tool="train_cv", args={}, result_summary=(
        "train_cv complete: kfold (5 folds), metric=r2, "
        "mean=-0.1234, std=0.02, fold_scores=[-0.12], duration_s=0.50"
    ))
    best = portfolio.best_cv_for(ws)
    assert best == pytest.approx(-0.1234)


def test_best_cv_for_returns_none_when_no_train_cv(comps_root):
    comps_root.mkdir(parents=True)
    ws = Workspace.create(root=comps_root / "titanic")
    Journal(ws).log_tool_call(tool="take_note", args={}, result_summary="ok")
    assert portfolio.best_cv_for(ws) is None
