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


def test_run_subprocess_accepts_existing_script_path(ws):
    """F7: when script_path is supplied, run_subprocess does NOT rewrite
    the file. It just executes it. The caller (run_python) owns the file
    so lint+run can share a single on-disk artifact instead of writing
    twice."""
    ws.scratch_dir.mkdir(parents=True, exist_ok=True)
    script = ws.scratch_dir / "run_caller_owned.py"
    script.write_text("print('owned-by-caller')")
    pre_mtime = script.stat().st_mtime
    result = sandbox.run_subprocess(
        script_path=script,
        workspace=ws,
        timeout_s=10,
        memory_mb=256,
    )
    assert result.returncode == 0
    assert "owned-by-caller" in result.stdout
    # The path returned matches what the caller passed in (no rename).
    assert result.script_path == script
    # And the file content has not been overwritten by run_subprocess.
    assert script.exists()
    assert script.read_text() == "print('owned-by-caller')"
    assert script.stat().st_mtime == pre_mtime


def test_run_subprocess_with_code_writes_run_ts_py(ws):
    """F7: when script_path is None (default), run_subprocess keeps the
    original behavior: write code to a fresh run_<ts>.py."""
    result = sandbox.run_subprocess(
        code="print('still works')",
        workspace=ws,
        timeout_s=10,
        memory_mb=256,
    )
    assert result.returncode == 0
    assert "still works" in result.stdout
    assert result.script_path.parent == ws.scratch_dir
    assert result.script_path.name.startswith("run_")
    assert result.script_path.suffix == ".py"
