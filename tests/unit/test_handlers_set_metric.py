"""Tests for ml_h.set_metric — always checkpoint-gated."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from kaggle_slayer.agent.handlers import ml as ml_h
from kaggle_slayer.agent.tools import ToolError
from kaggle_slayer.harness import checkpoints as cp
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
    checkpoint_handler: cp.CheckpointHandler | None = None
    best_cv_mean: float | None = None


@pytest.fixture
def ctx(tmp_path):
    ws = Workspace.create(root=tmp_path / "comp")
    journal = Journal(ws)
    return _Ctx(
        workspace=ws,
        journal=journal,
        checkpoint_handler=cp.CheckpointHandler(
            mode=cp.HandlerMode.STUB,
            journal=journal,
            stub_decision=cp.Decision.APPROVE,
        ),
    )


def test_set_metric_changes_metric_on_approval(ctx):
    result = ml_h.set_metric(ctx, name="auc")
    assert ctx.metric_name == "auc"
    assert "auc" in result


def test_set_metric_rejected_on_deny(ctx, tmp_path):
    """If the checkpoint handler denies, metric must not change."""
    ws = Workspace.create(root=tmp_path / "comp2")
    journal = Journal(ws)
    deny_handler = cp.CheckpointHandler(
        mode=cp.HandlerMode.STUB, journal=journal, stub_decision=cp.Decision.DENY
    )
    ctx_deny = type(ctx)(workspace=ws, journal=journal, checkpoint_handler=deny_handler)
    original = ctx_deny.metric_name
    with pytest.raises(ToolError, match="denied"):
        ml_h.set_metric(ctx_deny, name="auc")
    assert ctx_deny.metric_name == original


def test_set_metric_validates_known_metric(ctx):
    """Unknown metric is rejected before the checkpoint runs."""
    with pytest.raises(ToolError, match="unknown metric"):
        ml_h.set_metric(ctx, name="bogus_metric")


def test_set_metric_journals_checkpoint_decision(ctx):
    ml_h.set_metric(ctx, name="auc")
    import json
    lines = ctx.workspace.run_log_path.read_text().splitlines()
    cp_records = [json.loads(line) for line in lines if json.loads(line).get("kind") == "checkpoint"]
    assert len(cp_records) == 1
    assert cp_records[0]["trigger"] == "set_metric"
    assert cp_records[0]["decision"] == "approve"
