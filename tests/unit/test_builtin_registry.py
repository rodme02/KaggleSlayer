"""Tests for handlers.make_builtin_registry — the wired-up tool registry."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from kaggle_slayer.agent.handlers import make_builtin_registry
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


@dataclass
class _Ctx:
    workspace: Workspace
    journal: Journal
    target_col: str = "target"
    problem_type: str = "classification"
    metric_name: str = "accuracy"
    cv_kind: str | None = None
    cv_params: dict = field(default_factory=dict)
    finished: bool = False
    final_summary: str = ""


@pytest.fixture
def ctx(tmp_path):
    ws = Workspace.create(root=tmp_path / "comp")
    return _Ctx(workspace=ws, journal=Journal(ws))


def test_builtin_registry_has_expected_tools(ctx):
    reg = make_builtin_registry()
    expected = {
        "read_context", "read_file", "write_file", "sample_rows",
        "take_note", "set_cv", "train_cv", "submit_local", "done",
        "run_python", "set_metric", "submit_kaggle", "request_human_approval",
    }
    assert set(reg.names()) == expected


def test_builtin_registry_invoke_write_file(ctx, tmp_path):
    reg = make_builtin_registry()
    reg.invoke("write_file", ctx=ctx, args={"path": "agent/fe.py", "content": "x = 1"})
    assert (ctx.workspace.agent_dir / "fe.py").read_text() == "x = 1"


def test_builtin_registry_invoke_done(ctx):
    reg = make_builtin_registry()
    reg.invoke("done", ctx=ctx, args={"summary": "all good"})
    assert ctx.finished is True
    assert ctx.final_summary == "all good"


def test_builtin_registry_function_declarations_format(ctx):
    """All declarations have name + description + parameters keys, JSON-schema shape."""
    reg = make_builtin_registry()
    decls = reg.to_function_declarations()
    assert len(decls) == 13
    for d in decls:
        assert d["name"]
        assert d["description"]
        assert d["parameters"]["type"] == "object"
