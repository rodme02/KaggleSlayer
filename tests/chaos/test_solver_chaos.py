"""Chaos test: scripted Solver run with 5% injected LLM failures.

The Solver's retry semantics: GeminiClient has its own retry around
TransientLLMError; the scripted client used here does NOT (it's a thin
fake). So under chaos the Solver loop may raise the transient out.

That's allowed — the success criterion is journal integrity (no corrupt
records), not loop completion. A future Solver-side retry layer would
let us tighten this to require completion; until then, the test asserts
the weaker invariant.
"""

from __future__ import annotations

import json

import pytest

from kaggle_slayer.agent.llm_client import (
    Response,
    ToolCall,
    TransientLLMError,
    Usage,
)
from kaggle_slayer.agent.solver import Solver
from tests.chaos.conftest import FailureInjectingLLMClient
from tests.fixtures.synthetic_comp import make_synthetic_comp

pytestmark = [pytest.mark.chaos, pytest.mark.integration]


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
    """Minimal scripted client; iterates a fixed list of Responses."""

    def __init__(self, responses):
        self._r = list(responses)
        self._i = 0

    def __call__(self, messages, *, tools=None, model=None):
        if self._i >= len(self._r):
            # If chaos consumes more calls than scripted, return a benign "done"
            # response so the loop ends gracefully rather than IndexError-ing.
            return Response(
                text="",
                tool_calls=[ToolCall(id="t_end", name="done", args={"summary": "scripted-end"})],
                usage=Usage(0, 0, 0),
            )
        r = self._r[self._i]
        self._i += 1
        return r


def test_solver_survives_5_percent_chaos(tmp_path, chaos_seed):
    workspace = make_synthetic_comp(tmp_path / "synthetic")
    scripted = _Scripted([
        Response(text="", tool_calls=[ToolCall(id="t1", name="write_file",
                 args={"path": "agent/fe.py", "content": _FE_CODE})],
                 usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t2", name="write_file",
                 args={"path": "agent/model.py", "content": _MODEL_CODE})],
                 usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t3", name="train_cv", args={})],
                 usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t4", name="submit_local",
                 args={"label": "chaos"})], usage=Usage(0, 0, 0)),
        Response(text="", tool_calls=[ToolCall(id="t5", name="done",
                 args={"summary": "chaos-tier complete"})], usage=Usage(0, 0, 0)),
    ])
    chaos_client = FailureInjectingLLMClient(
        inner_call=scripted, rate=0.05, seed=chaos_seed,
    )

    solver = Solver(
        workspace=workspace,
        llm_client=chaos_client,
        target_col="Survived",
        problem_type="classification",
        metric_name="accuracy",
        max_iterations=20,
    )

    try:
        result = solver.solve()
        assert result.status in ("done", "max_iterations")
    except TransientLLMError:
        # A transient bubbled up — accepted. Journal still must be parseable.
        pass

    # The journal must be parseable line-by-line (no corruption).
    if workspace.run_log_path.exists():
        for line in workspace.run_log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            json.loads(line)  # raises if corrupted

    # We expect at least SOME successes occurred (otherwise nothing happened).
    assert chaos_client.successes >= 1


def test_chaos_client_failure_rate_is_seeded(chaos_seed):
    """Same seed -> same failure pattern across two runs."""
    rs = []
    for _ in range(2):
        scripted = _Scripted([
            Response(text="", tool_calls=[], usage=Usage(0, 0, 0))
            for _ in range(1000)
        ])
        c = FailureInjectingLLMClient(inner_call=scripted, rate=0.05, seed=chaos_seed)
        for _ in range(1000):
            try:
                c.call(messages=[])
            except TransientLLMError:
                pass
        rs.append(c.failures)
    assert rs[0] == rs[1]
    # ~5% of 1000 = 50; allow a wide band for RNG slop
    assert 25 <= rs[0] <= 100
