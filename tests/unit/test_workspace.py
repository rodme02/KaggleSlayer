"""Tests for kaggle_slayer.harness.workspace.Workspace."""

from __future__ import annotations

import pytest

from kaggle_slayer.harness import workspace as ws_mod


def test_workspace_create_makes_all_directories(tmp_path):
    w = ws_mod.Workspace.create(root=tmp_path / "competitions" / "titanic")
    assert w.root.is_dir()
    assert w.raw_dir.is_dir()
    assert w.agent_dir.is_dir()
    assert w.versions_dir.is_dir()
    assert w.scratch_dir.is_dir()
    assert w.artifacts_dir.is_dir()
    assert w.submissions_dir.is_dir()
    assert w.run_log_path.parent == w.root
    assert w.notes_path.parent == w.root


def test_workspace_create_is_idempotent(tmp_path):
    root = tmp_path / "competitions" / "titanic"
    w1 = ws_mod.Workspace.create(root=root)
    w2 = ws_mod.Workspace.create(root=root)
    assert w1.root == w2.root
    assert root.is_dir()


def test_workspace_load_existing(tmp_path):
    root = tmp_path / "competitions" / "titanic"
    ws_mod.Workspace.create(root=root)
    loaded = ws_mod.Workspace.load(root=root)
    assert loaded.root == root


def test_workspace_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no workspace at"):
        ws_mod.Workspace.load(root=tmp_path / "does_not_exist")


def test_workspace_context_path(tmp_path):
    w = ws_mod.Workspace.create(root=tmp_path / "comp")
    assert w.context_path == w.root / "context.md"


def test_workspace_fe_and_model_paths(tmp_path):
    w = ws_mod.Workspace.create(root=tmp_path / "comp")
    assert w.fe_path == w.agent_dir / "fe.py"
    assert w.model_path == w.agent_dir / "model.py"


def test_workspace_competition_name_derived_from_dir(tmp_path):
    w = ws_mod.Workspace.create(root=tmp_path / "competitions" / "house-prices")
    assert w.name == "house-prices"
