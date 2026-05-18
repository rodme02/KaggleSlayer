"""Tests for kaggle_slayer.harness.telemetry.errors."""

from __future__ import annotations

import json

import pytest

from kaggle_slayer.harness.telemetry import errors


@pytest.fixture
def isolated_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(errors, "DEFAULT_DIR", tmp_path / "errors")
    return tmp_path / "errors"


def test_capture_writes_json_file(isolated_errors):
    try:
        raise ValueError("kaboom")
    except ValueError as e:
        path = errors.capture(e, recent_calls=[{"tool": "x"}], env={"FOO": "bar"})
    assert path.exists()
    rec = json.loads(path.read_text())
    assert rec["exception"]["type"] == "ValueError"
    assert "kaboom" in rec["exception"]["message"]
    assert rec["recent_calls"] == [{"tool": "x"}]
    assert rec["env"]["FOO"] == "bar"
    assert "traceback" in rec


def test_capture_filename_has_iso_timestamp(isolated_errors):
    try:
        raise RuntimeError("boom")
    except RuntimeError as e:
        path = errors.capture(e, recent_calls=[], env={})
    import re
    assert re.match(r"\d{4}-\d{2}-\d{2}_\d{6}", path.stem)


def test_capture_rotation_keeps_last_100(isolated_errors):
    """When >100 error files exist, the oldest are pruned."""
    # Seed 105 fake error files with sortable names; the rotation must trim them.
    isolated_errors.mkdir(parents=True, exist_ok=True)
    for i in range(105):
        (isolated_errors / f"2026-05-17_{i:06d}_old.json").write_text("{}")
    try:
        raise ValueError("x")
    except ValueError as e:
        errors.capture(e, recent_calls=[], env={})
    files = sorted(isolated_errors.glob("*.json"))
    assert len(files) == 100


def test_capture_redacts_secrets_from_env(isolated_errors):
    """Keys whose UPPERCASE name contains KEY/TOKEN/SECRET/PASSWORD get redacted."""
    try:
        raise ValueError("x")
    except ValueError as e:
        path = errors.capture(e, recent_calls=[], env={
            "GEMINI_API_KEY": "sk-real-key",
            "KAGGLE_API_TOKEN": "KGAT_real",
            "MY_SECRET": "hush",
            "DB_PASSWORD": "p4ssw0rd",
            "PATH": "/usr/bin",
        })
    rec = json.loads(path.read_text())
    assert rec["env"]["GEMINI_API_KEY"] == "<redacted>"
    assert rec["env"]["KAGGLE_API_TOKEN"] == "<redacted>"
    assert rec["env"]["MY_SECRET"] == "<redacted>"
    assert rec["env"]["DB_PASSWORD"] == "<redacted>"
    assert rec["env"]["PATH"] == "/usr/bin"
