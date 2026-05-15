"""Integration: Solver loop drives set_metric and submit_kaggle checkpoints."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from kaggle_slayer.agent.llm_client import Response, ToolCall, Usage
from kaggle_slayer.agent.solver import Solver
from kaggle_slayer.harness import checkpoints as cp
from kaggle_slayer.harness.journal import Journal
from tests.fixtures.synthetic_comp import make_synthetic_comp

pytestmark = pytest.mark.integration


_FE_CODE = '''
import pandas as pd

class _PT:
    def __init__(self, cols, means):
        self.cols, self.means = cols, means
    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        for c in self.cols:
            if c in df.columns:
                out[c] = df[c].fillna(self.means.get(c, 0.0))
        if "id" in out.columns:
            out = out.drop(columns=["id"])
        return out

def fit_feature_transformer(train_df, target_col):
    cols = [c for c in train_df.columns if c not in (target_col, "id") and train_df[c].dtype.kind in "fiub"]
    means = {c: float(train_df[c].mean()) for c in cols}
    return _PT(cols, means)
'''

_MODEL_CODE = '''
from sklearn.linear_model import LogisticRegression
def fit_model(X_train, y_train, problem_type, metric_name):
    m = LogisticRegression(max_iter=500, random_state=42)
    m.fit(X_train, y_train)
    return m
'''


class _Scripted:
    def __init__(self, responses):
        self._r, self._i = list(responses), 0
        self.captured = []

    def call(self, messages, *, tools=None, model=None):
        self.captured.append(list(messages))
        r = self._r[self._i]
        self._i += 1
        return r


def _build_solver(workspace, client, handler):
    fake_kaggle = MagicMock()
    return Solver(
        workspace=workspace,
        llm_client=client,
        target_col="Survived",
        problem_type="classification",
        metric_name="accuracy",
        max_iterations=15,
        checkpoint_handler=handler,
        kaggle_client=fake_kaggle,
    ), fake_kaggle


def test_set_metric_approved_then_submit_kaggle_approved(tmp_path):
    workspace = make_synthetic_comp(tmp_path / "synthetic")
    journal = Journal(workspace)
    handler = cp.CheckpointHandler(mode=cp.HandlerMode.STUB, journal=journal, stub_decision=cp.Decision.APPROVE)
    client = _Scripted([
        Response(text="", tool_calls=[ToolCall(id="t1", name="set_metric", args={"name": "auc"})], usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t2", name="write_file", args={"path": "agent/fe.py", "content": _FE_CODE})], usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t3", name="write_file", args={"path": "agent/model.py", "content": _MODEL_CODE})], usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t4", name="train_cv", args={})], usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t5", name="submit_local", args={"label": "v1"})], usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t6", name="submit_kaggle", args={"csv_path": None, "message": "v1"})], usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t7", name="done", args={"summary": "fin"})], usage=Usage(0, 0, 0)),
    ])
    solver, fake_kaggle = _build_solver(workspace, client, handler)

    # Find the submit_local output to feed into submit_kaggle's csv_path.
    # Easier: patch the scripted sequence after the solver writes submissions/v1.csv.
    # Trick: pre-write the submission file so submit_kaggle's path resolves.
    (workspace.submissions_dir / "manual.csv").write_text("id,target\n1,0\n")
    client._r[5] = Response(text="", tool_calls=[ToolCall(
        id="t6", name="submit_kaggle",
        args={"csv_path": "submissions/manual.csv", "message": "v1"},
    )], usage=Usage(0, 0, 0))

    result = solver.solve()
    assert result.status == "done"
    fake_kaggle.submit.assert_called_once()

    cp_records = [
        json.loads(line) for line in workspace.run_log_path.read_text().splitlines()
        if json.loads(line).get("kind") == "checkpoint"
    ]
    triggers = [r["trigger"] for r in cp_records]
    assert "set_metric" in triggers
    assert any(t.startswith("submit_kaggle") for t in triggers)
    # Every checkpoint was approved
    assert all(r["decision"] == "approve" for r in cp_records)


def test_set_metric_denied_keeps_original_metric(tmp_path):
    workspace = make_synthetic_comp(tmp_path / "synthetic")
    journal = Journal(workspace)
    handler = cp.CheckpointHandler(mode=cp.HandlerMode.STUB, journal=journal, stub_decision=cp.Decision.DENY)
    client = _Scripted([
        Response(text="", tool_calls=[ToolCall(id="t1", name="set_metric", args={"name": "auc"})], usage=Usage(0, 0, 0)),
        # After being denied, the agent gives up and calls done
        Response(text="", tool_calls=[ToolCall(id="t2", name="done", args={"summary": "blocked"})], usage=Usage(0, 0, 0)),
    ])
    solver, fake_kaggle = _build_solver(workspace, client, handler)
    result = solver.solve()

    # Metric stays as the original 'accuracy'
    assert solver.ctx.metric_name == "accuracy"
    # The tool dispatch fed a ToolError back to the model
    tool_results = [m for m in client.captured[1] if m.role == "tool"]
    assert any("denied" in m.content.lower() for m in tool_results)
    # Kaggle was never touched
    fake_kaggle.submit.assert_not_called()
    assert result.status == "done"
