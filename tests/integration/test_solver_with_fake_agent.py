"""Integration test: Solver runs against a synthetic competition using a
scripted FakeLLMClient (no real Gemini calls)."""

from __future__ import annotations

import pytest

from kaggle_slayer.agent.llm_client import Response, ToolCall, Usage
from kaggle_slayer.agent.solver import Solver
from tests.fixtures.synthetic_comp import make_synthetic_comp

pytestmark = pytest.mark.integration


_FE_CODE = '''
import pandas as pd

class _PT:
    def __init__(self, cols, means):
        self.cols = cols
        self.means = means
    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        for c in self.cols:
            if c in df.columns:
                out[c] = df[c].fillna(self.means.get(c, 0.0))
        # Drop the id column if present
        if "id" in out.columns:
            out = out.drop(columns=["id"])
        return out

def fit_feature_transformer(train_df, target_col):
    cols = [c for c in train_df.columns
            if c not in (target_col, "id") and train_df[c].dtype.kind in "fiub"]
    means = {c: float(train_df[c].mean()) for c in cols}
    return _PT(cols, means)
'''

_MODEL_CODE = '''
from sklearn.linear_model import LogisticRegression, Ridge

def fit_model(X_train, y_train, problem_type, metric_name):
    if problem_type == "classification":
        m = LogisticRegression(max_iter=500, random_state=42)
    else:
        m = Ridge(alpha=1.0, random_state=42)
    m.fit(X_train, y_train)
    return m
'''


class _ScriptedClient:
    """LLMClient that returns a fixed sequence of Responses with tool calls."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0
        self.captured = []

    def call(self, messages, *, tools=None, model=None):
        self.captured.append(list(messages))
        r = self._responses[self._i]
        self._i += 1
        return r


def test_solver_end_to_end_with_scripted_tools(tmp_path):
    """The scripted agent: write fe.py, write model.py, train_cv, submit_local, done."""
    workspace = make_synthetic_comp(tmp_path / "synthetic")

    responses = [
        # 1. Write fe.py
        Response(text="", tool_calls=[ToolCall(
            id="t1", name="write_file",
            args={"path": "agent/fe.py", "content": _FE_CODE},
        )], usage=Usage(0, 0, 0)),
        # 2. Write model.py
        Response(text="", tool_calls=[ToolCall(
            id="t2", name="write_file",
            args={"path": "agent/model.py", "content": _MODEL_CODE},
        )], usage=Usage(0, 0, 0)),
        # 3. Train CV
        Response(text="", tool_calls=[ToolCall(
            id="t3", name="train_cv", args={},
        )], usage=Usage(0, 0, 0)),
        # 4. Submit local
        Response(text="", tool_calls=[ToolCall(
            id="t4", name="submit_local", args={"label": "scripted"},
        )], usage=Usage(0, 0, 0)),
        # 5. Done
        Response(text="", tool_calls=[ToolCall(
            id="t5", name="done", args={"summary": "scripted run complete"},
        )], usage=Usage(0, 0, 0)),
    ]
    client = _ScriptedClient(responses)
    solver = Solver(
        workspace=workspace,
        llm_client=client,
        target_col="Survived",
        problem_type="classification",
        metric_name="accuracy",
        max_iterations=10,
    )

    result = solver.solve()
    assert result.status == "done"
    assert "scripted" in result.summary

    # Verify the files the agent wrote
    assert workspace.fe_path.exists()
    assert workspace.model_path.exists()
    # Versions archive exists (one fe + one model)
    assert (workspace.versions_dir / "fe_v01.py").exists()
    assert (workspace.versions_dir / "model_v01.py").exists()
    # Submission written
    submissions = list(workspace.submissions_dir.glob("*scripted*.csv"))
    assert len(submissions) == 1
    # Run log has all 5 tool calls
    log_lines = workspace.run_log_path.read_text().splitlines()
    assert len(log_lines) >= 5
