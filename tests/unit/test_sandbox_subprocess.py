"""Tests for kaggle_slayer.harness.sandbox.run_subprocess."""

from __future__ import annotations

import sys

import pytest

from kaggle_slayer.harness import sandbox
from kaggle_slayer.harness.workspace import Workspace


@pytest.fixture
def ws(tmp_path):
    return Workspace.create(root=tmp_path / "comp")


def test_run_subprocess_returns_stdout(ws):
    result = sandbox.run_subprocess(
        code="print('hello, world')",
        workspace=ws,
        timeout_s=10,
        memory_mb=256,
    )
    assert result.returncode == 0
    assert "hello, world" in result.stdout
    assert result.stderr == ""
    assert result.killed_reason is None


def test_run_subprocess_captures_stderr_and_nonzero_exit(ws):
    result = sandbox.run_subprocess(
        code="import sys; sys.stderr.write('boom\\n'); sys.exit(3)",
        workspace=ws,
        timeout_s=10,
        memory_mb=256,
    )
    assert result.returncode == 3
    assert "boom" in result.stderr


def test_run_subprocess_kills_on_timeout(ws):
    """Infinite loop must be killed by timeout_s; returncode is None or signal."""
    result = sandbox.run_subprocess(
        code="while True:\n    pass",
        workspace=ws,
        timeout_s=1,
        memory_mb=256,
    )
    assert result.killed_reason == "timeout"
    # Should record a returncode != 0 (signal or None depending on platform)
    assert result.returncode != 0


def test_run_subprocess_runs_with_workspace_cwd(ws):
    """The subprocess's cwd must be the workspace root, not the test cwd."""
    result = sandbox.run_subprocess(
        code="import os; print(os.getcwd())",
        workspace=ws,
        timeout_s=10,
        memory_mb=256,
    )
    assert result.returncode == 0
    assert str(ws.root.resolve()) in result.stdout


@pytest.mark.skipif(sys.platform == "darwin", reason="RLIMIT_AS not enforced reliably on macOS")
def test_run_subprocess_kills_on_memory_overflow(ws):
    """Allocating a huge bytearray triggers the memory limit on Linux."""
    code = (
        "data = bytearray(900 * 1024 * 1024)\n"  # 900 MB
        "print('alloc_done')\n"
    )
    result = sandbox.run_subprocess(
        code=code,
        workspace=ws,
        timeout_s=15,
        memory_mb=128,  # 128 MB cap — much less than 900 MB attempt
    )
    # On Linux RLIMIT_AS forces MemoryError or kill; alloc_done must not print
    assert "alloc_done" not in result.stdout


def test_run_subprocess_writes_script_to_scratch_dir(ws):
    """The temp script lives under workspace.scratch_dir for post-hoc debugging."""
    sandbox.run_subprocess(
        code="x = 1\nprint(x)",
        workspace=ws,
        timeout_s=10,
        memory_mb=256,
    )
    scripts = list(ws.scratch_dir.glob("run_*.py"))
    assert len(scripts) >= 1
    assert "x = 1" in scripts[0].read_text()
