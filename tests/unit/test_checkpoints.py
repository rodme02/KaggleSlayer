"""Tests for kaggle_slayer.harness.checkpoints."""

from __future__ import annotations

import json

import pytest

from kaggle_slayer.harness import checkpoints as cp
from kaggle_slayer.harness.journal import Journal
from kaggle_slayer.harness.workspace import Workspace


@pytest.fixture
def journal(tmp_path):
    ws = Workspace.create(root=tmp_path / "comp")
    return Journal(ws)


def test_decision_enum_has_four_outcomes():
    assert {cp.Decision.APPROVE, cp.Decision.DENY, cp.Decision.ABORT, cp.Decision.SKIP_CHECK}
    assert cp.Decision.APPROVE.value == "approve"


def test_checkpoint_request_dataclass():
    req = cp.CheckpointRequest(
        trigger=cp.CheckpointTrigger.SUBMIT_KAGGLE_FIRST,
        action="submit submission 'lr_v1' to kaggle competition titanic",
        evidence={"csv_rows": 418, "cv_mean": 0.81},
    )
    assert req.trigger == cp.CheckpointTrigger.SUBMIT_KAGGLE_FIRST
    assert req.evidence["cv_rows" if False else "csv_rows"] == 418


def test_handler_auto_mode_approves_safe_triggers(journal):
    """Auto-mode 'safe' approves auto-approve cases (per spec §9)."""
    handler = cp.CheckpointHandler(mode=cp.HandlerMode.AUTO_SAFE, journal=journal)
    req = cp.CheckpointRequest(
        trigger=cp.CheckpointTrigger.SUBMIT_KAGGLE_NON_REGRESSION,
        action="resubmit (CV improved)",
        evidence={"cv_mean": 0.85, "prev_best": 0.83},
    )
    assert handler.request(req) == cp.Decision.APPROVE


def test_handler_auto_mode_denies_unsafe_triggers(journal):
    """Auto-mode 'safe' denies the always-block triggers (per spec §9)."""
    handler = cp.CheckpointHandler(mode=cp.HandlerMode.AUTO_SAFE, journal=journal)
    req = cp.CheckpointRequest(
        trigger=cp.CheckpointTrigger.SUBMIT_KAGGLE_FIRST,
        action="first submission",
        evidence={},
    )
    assert handler.request(req) == cp.Decision.DENY


def test_handler_stub_mode_uses_injected_decision(journal):
    """Stub mode is used by tests: a fixed Decision."""
    handler = cp.CheckpointHandler(
        mode=cp.HandlerMode.STUB, journal=journal, stub_decision=cp.Decision.APPROVE
    )
    req = cp.CheckpointRequest(
        trigger=cp.CheckpointTrigger.SET_METRIC,
        action="change metric to auc",
        evidence={"current": "accuracy", "proposed": "auc"},
    )
    assert handler.request(req) == cp.Decision.APPROVE


def test_handler_journals_every_decision(journal):
    handler = cp.CheckpointHandler(
        mode=cp.HandlerMode.STUB, journal=journal, stub_decision=cp.Decision.DENY
    )
    handler.request(cp.CheckpointRequest(
        trigger=cp.CheckpointTrigger.WALL_CLOCK_BUDGET,
        action="extend wall-clock budget",
        evidence={"elapsed_s": 5400, "budget_s": 5400},
    ))
    lines = journal.workspace.run_log_path.read_text().splitlines()
    records = [json.loads(line) for line in lines]
    cp_records = [r for r in records if r.get("kind") == "checkpoint"]
    assert len(cp_records) == 1
    assert cp_records[0]["decision"] == "deny"
    assert cp_records[0]["trigger"] == "wall_clock_budget"
    assert cp_records[0]["evidence"]["elapsed_s"] == 5400


def test_callable_mode_invokes_provided_function(journal):
    """The CLI-side prompt uses a function injected via HandlerMode.CALLABLE."""
    seen: list[cp.CheckpointRequest] = []

    def prompt(req: cp.CheckpointRequest) -> cp.Decision:
        seen.append(req)
        return cp.Decision.ABORT

    handler = cp.CheckpointHandler(
        mode=cp.HandlerMode.CALLABLE, journal=journal, prompt_fn=prompt
    )
    req = cp.CheckpointRequest(
        trigger=cp.CheckpointTrigger.COST_BUDGET,
        action="raise cost budget",
        evidence={"spent_usd": 5.20, "budget_usd": 5.0},
    )
    assert handler.request(req) == cp.Decision.ABORT
    assert len(seen) == 1
    assert seen[0].trigger == cp.CheckpointTrigger.COST_BUDGET
