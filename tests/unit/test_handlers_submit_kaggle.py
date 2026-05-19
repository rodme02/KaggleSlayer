"""Tests for ml_h.submit_kaggle — checkpoint-gated, regression-aware."""

from __future__ import annotations

import json
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


def _seed_leaderboard(ws: Workspace, *, cv_at_submit: float | None, message: str = "prior") -> None:
    """Append a leaderboard.jsonl record to simulate a prior successful submit."""
    lb_path = ws.submissions_dir / "leaderboard.jsonl"
    lb_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": "2026-05-15T00:00:00+00:00",
        "csv": "prior.csv",
        "cv_at_submit": cv_at_submit,
        "message": message,
        "competition": ws.name,
    }
    with lb_path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _read_leaderboard(ws: Workspace) -> list[dict]:
    lb_path = ws.submissions_dir / "leaderboard.jsonl"
    if not lb_path.exists():
        return []
    return [json.loads(line) for line in lb_path.read_text().splitlines() if line.strip()]


def test_submit_kaggle_first_submission_gated_as_first(tmp_path):
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.APPROVE)
    result = ml_h.submit_kaggle(ctx, csv_path="submissions/2026-05-15_001_lr.csv", message="baseline")
    # Approved → kaggle.submit must have been called once
    ctx.kaggle_client.submit.assert_called_once()
    # And the trigger journalled was SUBMIT_KAGGLE_FIRST
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
    """Second submission with same-or-better CV is gated as SUBMIT_KAGGLE_NON_REGRESSION.

    Prior submits are tracked in submissions/leaderboard.jsonl (spec §10).
    """
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.APPROVE)
    # Simulate a successful first submission via leaderboard.jsonl (no CV recorded).
    _seed_leaderboard(ctx.workspace, cv_at_submit=None, message="v1")
    ctx.best_cv_mean = 0.80
    (ctx.workspace.submissions_dir / "v2.csv").write_text("id,target\n1,0\n")
    ml_h.submit_kaggle(ctx, csv_path="submissions/v2.csv", message="v2")
    lines = ctx.workspace.run_log_path.read_text().splitlines()
    cp_records = [json.loads(line) for line in lines if json.loads(line).get("kind") == "checkpoint"]
    # Most recent checkpoint must be a non-regression trigger
    assert cp_records[-1]["trigger"] == "submit_kaggle_non_regression"


def test_submit_kaggle_subsequent_regression(tmp_path):
    """Second submission with worse CV than best is gated as SUBMIT_KAGGLE_REGRESSION."""
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.APPROVE)
    _seed_leaderboard(ctx.workspace, cv_at_submit=0.85, message="v1")
    ctx.best_cv_mean = 0.80  # current model is WORSE than 0.85 at the previous submit
    (ctx.workspace.submissions_dir / "v2.csv").write_text("id,target\n1,0\n")
    ml_h.submit_kaggle(ctx, csv_path="submissions/v2.csv", message="v2")
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


def test_submit_kaggle_writes_single_run_log_record(tmp_path):
    """F1/F3: exactly one tool_call record per successful submit.

    Previously the handler wrote one record AND the Solver's _dispatch wrote a
    second one, producing two duplicate entries that ballooned the rebuilt
    conversation to 4 messages per submit. The handler must now write zero
    run_log records itself; the Solver's _dispatch handles journalling.
    """
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.APPROVE)
    # Simulate the Solver's _dispatch journalling by calling log_tool_call
    # exactly once around the handler invocation. The handler itself must add
    # zero extra tool_call records to run_log.jsonl.
    result = ml_h.submit_kaggle(
        ctx, csv_path="submissions/2026-05-15_001_lr.csv", message="baseline"
    )
    ctx.journal.log_tool_call(
        tool="submit_kaggle",
        args={"csv_path": "submissions/2026-05-15_001_lr.csv", "message": "baseline"},
        result_summary=result,
    )
    lines = ctx.workspace.run_log_path.read_text().splitlines()
    records = [json.loads(line) for line in lines]
    submit_calls = [r for r in records if r.get("tool") == "submit_kaggle" and r.get("kind") == "tool_call"]
    assert len(submit_calls) == 1, (
        f"expected exactly one submit_kaggle tool_call record, found {len(submit_calls)}"
    )


def test_submit_kaggle_appends_leaderboard_record(tmp_path):
    """F1: successful submit must append a single record to submissions/leaderboard.jsonl."""
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.APPROVE)
    ctx.best_cv_mean = 0.72
    ml_h.submit_kaggle(
        ctx, csv_path="submissions/2026-05-15_001_lr.csv", message="baseline"
    )
    records = _read_leaderboard(ctx.workspace)
    assert len(records) == 1
    rec = records[0]
    assert rec["csv"] == "2026-05-15_001_lr.csv"
    assert rec["message"] == "baseline"
    assert rec["competition"] == "test-comp"
    assert rec["cv_at_submit"] == pytest.approx(0.72)
    assert "ts" in rec


def test_submit_kaggle_leaderboard_record_not_written_on_denial(tmp_path):
    """Denied submits must NOT pollute leaderboard.jsonl."""
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.DENY)
    with pytest.raises(ToolError, match="denied"):
        ml_h.submit_kaggle(
            ctx, csv_path="submissions/2026-05-15_001_lr.csv", message="baseline"
        )
    assert _read_leaderboard(ctx.workspace) == []


def test_submit_kaggle_regression_for_rmse_metric(tmp_path):
    """F14: with rmse (lower is better), a HIGHER current CV vs prior best is
    a regression. The previous classifier inverted this and would auto-approve
    a worse-logloss/RMSE/MAE model under AUTO_SAFE — silently shipping rubbish.
    """
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.APPROVE)
    ctx.metric_name = "rmse"
    _seed_leaderboard(ctx.workspace, cv_at_submit=0.5)  # prior best (lower=better)
    ctx.best_cv_mean = 0.7  # current model is WORSE (higher RMSE)
    (ctx.workspace.submissions_dir / "v2.csv").write_text("id,target\n1,0\n")
    ml_h.submit_kaggle(ctx, csv_path="submissions/v2.csv", message="v2")
    lines = ctx.workspace.run_log_path.read_text().splitlines()
    cp_records = [json.loads(line) for line in lines if json.loads(line).get("kind") == "checkpoint"]
    assert cp_records[-1]["trigger"] == "submit_kaggle_regression"


def test_submit_kaggle_non_regression_for_rmse_metric(tmp_path):
    """F14: with rmse, a LOWER current CV vs prior best is an improvement."""
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.APPROVE)
    ctx.metric_name = "rmse"
    _seed_leaderboard(ctx.workspace, cv_at_submit=0.7)  # prior best
    ctx.best_cv_mean = 0.5  # current model is BETTER (lower RMSE)
    (ctx.workspace.submissions_dir / "v2.csv").write_text("id,target\n1,0\n")
    ml_h.submit_kaggle(ctx, csv_path="submissions/v2.csv", message="v2")
    lines = ctx.workspace.run_log_path.read_text().splitlines()
    cp_records = [json.loads(line) for line in lines if json.loads(line).get("kind") == "checkpoint"]
    assert cp_records[-1]["trigger"] == "submit_kaggle_non_regression"


def test_submit_kaggle_regression_for_logloss_metric(tmp_path):
    """F14: with logloss (lower is better), a higher score is a regression."""
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.APPROVE)
    ctx.metric_name = "logloss"
    _seed_leaderboard(ctx.workspace, cv_at_submit=0.30)
    ctx.best_cv_mean = 0.45  # worse logloss
    (ctx.workspace.submissions_dir / "v2.csv").write_text("id,target\n1,0\n")
    ml_h.submit_kaggle(ctx, csv_path="submissions/v2.csv", message="v2")
    lines = ctx.workspace.run_log_path.read_text().splitlines()
    cp_records = [json.loads(line) for line in lines if json.loads(line).get("kind") == "checkpoint"]
    assert cp_records[-1]["trigger"] == "submit_kaggle_regression"


def test_classify_submit_trigger_tolerates_partial_trailing_line(tmp_path):
    """A crashed mid-write leaderboard.jsonl must not break classification."""
    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.APPROVE)
    _seed_leaderboard(ctx.workspace, cv_at_submit=0.70)
    # Append a partial JSON line, as if a crash interrupted a write.
    lb_path = ctx.workspace.submissions_dir / "leaderboard.jsonl"
    with lb_path.open("a") as f:
        f.write('{"ts": "2026-05-16", "csv": "partial.cs')  # no newline, no closing brace
    ctx.best_cv_mean = 0.80
    (ctx.workspace.submissions_dir / "v2.csv").write_text("id,target\n1,0\n")
    # Should not raise — partial trailing line is silently skipped.
    ml_h.submit_kaggle(ctx, csv_path="submissions/v2.csv", message="v2")
    lines = ctx.workspace.run_log_path.read_text().splitlines()
    cp_records = [json.loads(line) for line in lines if json.loads(line).get("kind") == "checkpoint"]
    assert cp_records[-1]["trigger"] == "submit_kaggle_non_regression"


def test_submit_kaggle_appends_calibration_row(tmp_path, monkeypatch):
    """Successful submit_kaggle writes a row to the calibration log."""
    from kaggle_slayer.harness.telemetry import calibration

    cal_path = tmp_path / "calibration.jsonl"
    monkeypatch.setattr(calibration, "DEFAULT_PATH", cal_path)

    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.APPROVE)
    ctx.best_cv_mean = 0.82
    ctx.last_cv_mean = 0.82
    ml_h.submit_kaggle(
        ctx, csv_path="submissions/2026-05-15_001_lr.csv", message="baseline"
    )

    lines = cal_path.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["competition"] == ctx.competition
    assert row["cv_score"] == 0.82
    assert row["lb_score"] is None
    assert row["metric"] == "accuracy"
    assert row["problem_type"] == "classification"
    # Default classification cv_strategy when ctx.cv_kind is None.
    assert row["cv_strategy"] == "stratified_kfold"


def test_submit_kaggle_skips_calibration_on_denial(tmp_path, monkeypatch):
    """A denied submit must not write a calibration row."""
    from kaggle_slayer.harness.telemetry import calibration

    cal_path = tmp_path / "calibration.jsonl"
    monkeypatch.setattr(calibration, "DEFAULT_PATH", cal_path)

    ctx = _make_ctx(tmp_path, stub_decision=cp.Decision.DENY)
    ctx.best_cv_mean = 0.82
    with pytest.raises(ToolError):
        ml_h.submit_kaggle(
            ctx, csv_path="submissions/2026-05-15_001_lr.csv", message="x"
        )

    assert not cal_path.exists() or cal_path.read_text().strip() == ""
