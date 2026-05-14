"""Tests for Workspace.next_version_path()."""

from __future__ import annotations

import pytest

from kaggle_slayer.harness.workspace import Workspace


@pytest.fixture
def ws(tmp_path):
    return Workspace.create(root=tmp_path / "comp")


def test_next_version_path_first_call(ws):
    p = ws.next_version_path("fe")
    assert p == ws.versions_dir / "fe_v01.py"


def test_next_version_path_increments_on_existing_files(ws):
    (ws.versions_dir / "fe_v01.py").write_text("# v1")
    (ws.versions_dir / "fe_v02.py").write_text("# v2")
    p = ws.next_version_path("fe")
    assert p == ws.versions_dir / "fe_v03.py"


def test_next_version_path_handles_model_kind(ws):
    (ws.versions_dir / "model_v01.py").write_text("# v1")
    p = ws.next_version_path("model")
    assert p == ws.versions_dir / "model_v02.py"


def test_next_version_path_ignores_unrelated_files(ws):
    # An fe_v01.py and a model_v01.py — asking for fe should not consider model.
    (ws.versions_dir / "fe_v01.py").write_text("")
    (ws.versions_dir / "model_v05.py").write_text("")
    assert ws.next_version_path("fe") == ws.versions_dir / "fe_v02.py"
    assert ws.next_version_path("model") == ws.versions_dir / "model_v06.py"


def test_next_version_path_rejects_invalid_kind(ws):
    with pytest.raises(ValueError, match="kind"):
        ws.next_version_path("submission")  # not fe or model
