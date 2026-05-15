"""Tests for kaggle_slayer.agent.handlers.python.run_python."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from kaggle_slayer.agent.handlers import python as ph
from kaggle_slayer.agent.tools import ToolError
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


@dataclass
class _Ctx:
    workspace: Workspace
    journal: Journal


@pytest.fixture
def ctx(tmp_path):
    ws = Workspace.create(root=tmp_path / "comp")
    return _Ctx(workspace=ws, journal=Journal(ws))


def test_run_python_executes_simple_code_and_returns_summary(ctx):
    result = ph.run_python(ctx, code="print(2 + 2)")
    assert "stdout=" in result and "4" in result
    assert "returncode=0" in result


def test_run_python_includes_stderr(ctx):
    result = ph.run_python(
        ctx,
        code="import sys; sys.stderr.write('warn\\n'); print('ok')",
    )
    assert "warn" in result and "ok" in result


def test_run_python_propagates_non_zero_exit(ctx):
    result = ph.run_python(ctx, code="import sys; sys.exit(7)")
    assert "returncode=7" in result


def test_run_python_rejects_lint_violations(ctx):
    """The code is AST-linted before exec; os.remove must be rejected."""
    with pytest.raises(ToolError, match="lint"):
        ph.run_python(ctx, code="import os; os.remove('train.csv')")


def test_run_python_rejects_subprocess_imports(ctx):
    with pytest.raises(ToolError, match="lint"):
        ph.run_python(
            ctx,
            code="import subprocess; subprocess.run(['ls'])",
        )


def test_run_python_handles_timeout(ctx):
    result = ph.run_python(ctx, code="while True:\n    pass", timeout_s=1)
    assert "killed=timeout" in result


def test_run_python_caps_output_size(ctx):
    """Tool result must not balloon the conversation history."""
    big = "for _ in range(100000):\n    print('x' * 80)\n"
    result = ph.run_python(ctx, code=big, timeout_s=5)
    # The result string we return to the LLM is capped to 8 KB.
    assert len(result) <= 9000  # 8 KB cap + small header/footer overhead


def test_run_python_writes_script_to_scratch(ctx):
    ph.run_python(ctx, code="print('persisted')")
    scripts = list(ctx.workspace.scratch_dir.glob("run_*.py"))
    assert len(scripts) >= 1
    assert "persisted" in scripts[0].read_text()
