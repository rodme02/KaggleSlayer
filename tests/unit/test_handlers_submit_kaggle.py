"""Tests for ml_h.submit_kaggle — checkpoint-gated, regression-aware."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

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
    kaggle_client: object | None = None
    competition: str = "test-comp"


def _make_ctx(tmp_path, *, stub_decision=cp.Decision.APPROVE):
    ws = Workspace.create(root=tmp_path / "comp")
    journal = Journal(ws)
    # Place a fake submission CSV so submit_kaggle has something to push
    (ws.submissions_dir / "2026-05-15_001_lr.csv").write_text("id,target\n1,0\n2,1\n")
    handler = cp.CheckpointHandler(
        mode=cp.HandlerMode.STUB, journal=journal, stub_decision=stub_decision
    )
    fake_kaggle = MagicMock()
    return _Ctx(
        workspace=ws, journal=journal, checkpoint_handler=handler,
        kaggle_client=fake_kaggle, competition="test-comp",
    )


def test_submit_kaggle_first_submission_gated_as_first(tmp_path):
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.APPROVE)
    result = ml_h.submit_kaggle(ctx, csv_path="submissions/2026-05-15_001_lr.csv", message="baseline")
    # Approved → kaggle.submit must have been called once
    ctx.kaggle_client.submit.assert_called_once()
    # And the trigger journalled was SUBMIT_KAGGLE_FIRST
    import json
    lines = ctx.workspace.run_log_path.read_text().splitlines()
    cp_records = [json.loads(line) for line in lines if json.loads(line).get("kind") == "checkpoint"]
    assert cp_records[0]["trigger"] == "submit_kaggle_first"
    assert "test-comp" in result


def test_submit_kaggle_denied_does_not_submit(tmp_path):
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.DENY)
    with pytest.raises(ToolError, match="denied"):
        ml_h.submit_kaggle(ctx, csv_path="submissions/2026-05-15_001_lr.csv", message="baseline")
    ctx.kaggle_client.submit.assert_not_called()


def test_submit_kaggle_subsequent_non_regression(tmp_path):
    """Second submission with same-or-better CV is gated as SUBMIT_KAGGLE_NON_REGRESSION."""
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.APPROVE)
    # Simulate a successful first submission already journalled
    ctx.journal._append(  # noqa: SLF001
        ctx.workspace.run_log_path,
        {"ts": "2026-05-15", "kind": "tool_call", "tool": "submit_kaggle",
         "args": {"csv_path": "..", "message": "v1"}, "result_summary": "submitted"},
    )
    ctx.best_cv_mean = 0.80
    # Pretend a v2 CSV exists; the test sets best_cv_mean to a higher value (improved)
    (ctx.workspace.submissions_dir / "v2.csv").write_text("id,target\n1,0\n")
    # The new CV that the LLM just ran is implicitly stored as best_cv_mean already.
    ml_h.submit_kaggle(ctx, csv_path="submissions/v2.csv", message="v2")
    import json
    lines = ctx.workspace.run_log_path.read_text().splitlines()
    cp_records = [json.loads(line) for line in lines if json.loads(line).get("kind") == "checkpoint"]
    # Most recent checkpoint must be a non-regression trigger
    assert cp_records[-1]["trigger"] == "submit_kaggle_non_regression"


def test_submit_kaggle_subsequent_regression(tmp_path):
    """Second submission with worse CV than best is gated as SUBMIT_KAGGLE_REGRESSION."""
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.APPROVE)
    ctx.journal._append(  # noqa: SLF001
        ctx.workspace.run_log_path,
        {"ts": "2026-05-15", "kind": "tool_call", "tool": "submit_kaggle",
         "args": {"csv_path": "..", "message": "v1"}, "result_summary": "submitted",
         "cv_at_submit": 0.85},  # we track the CV at each submission
    )
    ctx.best_cv_mean = 0.80  # current model is WORSE than 0.85 at the previous submit
    (ctx.workspace.submissions_dir / "v2.csv").write_text("id,target\n1,0\n")
    ml_h.submit_kaggle(ctx, csv_path="submissions/v2.csv", message="v2")
    import json
    lines = ctx.workspace.run_log_path.read_text().splitlines()
    cp_records = [json.loads(line) for line in lines if json.loads(line).get("kind") == "checkpoint"]
    assert cp_records[-1]["trigger"] == "submit_kaggle_regression"


def test_submit_kaggle_rejects_nonexistent_csv(tmp_path):
    ctx = _make_ctx(tmp_path)
    with pytest.raises(ToolError, match="not found"):
        ml_h.submit_kaggle(ctx, csv_path="submissions/missing.csv", message="x")


def test_submit_kaggle_rejects_path_traversal(tmp_path):
    ctx = _make_ctx(tmp_path)
    with pytest.raises(ToolError, match="outside"):
        ml_h.submit_kaggle(ctx, csv_path="../escape.csv", message="x")
