"""Tests for ml_h.request_human_approval — agent-initiated checkpoint.

Unlike set_metric and submit_kaggle (which raise ToolError on DENY/ABORT),
this handler ALWAYS returns the decision as a string. The agent — not the
harness — decides what to do with the answer. F12 documents this asymmetry
and these tests pin the behavior down.
"""

from __future__ import annotations

import json
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


def _make_ctx(tmp_path, *, stub_decision: cp.Decision) -> _Ctx:
    ws = Workspace.create(root=tmp_path / "comp")
    journal = Journal(ws)
    handler = cp.CheckpointHandler(
        mode=cp.HandlerMode.STUB, journal=journal, stub_decision=stub_decision
    )
    return _Ctx(workspace=ws, journal=journal, checkpoint_handler=handler)


def test_request_human_approval_returns_decision_string_on_deny(tmp_path):
    """DENY must NOT raise. The agent gets back 'decision=deny' and decides."""
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.DENY)
    result = ml_h.request_human_approval(ctx, action="experiment with X")
    assert result == "decision=deny"


def test_request_human_approval_returns_decision_string_on_approve(tmp_path):
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.APPROVE)
    result = ml_h.request_human_approval(ctx, action="experiment with X")
    assert result == "decision=approve"


def test_request_human_approval_returns_decision_string_on_abort(tmp_path):
    """ABORT also returns — the agent (not the harness) decides what to do."""
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.ABORT)
    result = ml_h.request_human_approval(ctx, action="experiment with X")
    assert result == "decision=abort"


def test_request_human_approval_returns_decision_string_on_skip_check(tmp_path):
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.SKIP_CHECK)
    result = ml_h.request_human_approval(ctx, action="experiment with X")
    assert result == "decision=skip_check"


def test_request_human_approval_parses_evidence_json(tmp_path):
    """Valid evidence_json must be parsed and attached to the checkpoint record."""
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.APPROVE)
    ml_h.request_human_approval(
        ctx, action="weigh in", evidence_json='{"k": "v", "n": 3}'
    )
    cp_records = [
        json.loads(line) for line in ctx.workspace.run_log_path.read_text().splitlines()
        if json.loads(line).get("kind") == "checkpoint"
    ]
    assert len(cp_records) == 1
    assert cp_records[0]["evidence"] == {"k": "v", "n": 3}


def test_request_human_approval_handles_invalid_json(tmp_path):
    """A broken evidence_json must still produce a checkpoint, with the raw text."""
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.APPROVE)
    ml_h.request_human_approval(ctx, action="weigh in", evidence_json="not json")
    cp_records = [
        json.loads(line) for line in ctx.workspace.run_log_path.read_text().splitlines()
        if json.loads(line).get("kind") == "checkpoint"
    ]
    assert len(cp_records) == 1
    assert cp_records[0]["evidence"] == {"raw_evidence": "not json"}


def test_request_human_approval_requires_checkpoint_handler(tmp_path):
    """No checkpoint handler on the context → ToolError, since the agent can't
    actually solicit a decision without one."""
    ws = Workspace.create(root=tmp_path / "comp")
    journal = Journal(ws)
    ctx = _Ctx(workspace=ws, journal=journal, checkpoint_handler=None)
    with pytest.raises(ToolError, match="checkpoint handler"):
        ml_h.request_human_approval(ctx, action="x")
