"""Chaos test: scripted Solver run with 5% injected LLM failures.

Verifies spec §11.3 / §13: "pipeline must survive without corrupting
state under transient LLM errors". The chaos client (which does NOT
have its own retry) is wrapped in :class:`RetryingLLMClient` — the
same retry shape GeminiClient uses internally. The Solver therefore
sees a resilient transport and the run must reach ``done``
deterministically, with the journal still parseable. We also assert
that failures were actually injected so a near-no-op pass is impossible.
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
from kaggle_slayer.agent.retrying_client import RetryingLLMClient
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
    # Wrap the chaos client in RetryingLLMClient so the Solver sees a resilient
    # transport — this is the spec §11.3 / §13 invariant under test. sleep is
    # a no-op so the test stays fast; retry_max=5 keeps us well clear of the
    # ~25-100 failures we'd expect across the scripted call budget.
    resilient_client = RetryingLLMClient(
        chaos_client,
        retry_max=5,
        retry_base_delay_s=0.0,
        sleep=lambda _: None,
    )

    solver = Solver(
        workspace=workspace,
        llm_client=resilient_client,
        target_col="Survived",
        problem_type="classification",
        metric_name="accuracy",
        max_iterations=20,
    )

    result = solver.solve()

    # Spec §11.3 / §13: under chaos with a retry wrapper, the run must reach
    # done deterministically.
    assert result.status == "done", (
        f"expected status=done, got {result.status!r} "
        f"(failures={chaos_client.failures}, successes={chaos_client.successes})"
    )

    # The journal must be parseable line-by-line (no corruption).
    assert workspace.run_log_path.exists(), "expected run_log.jsonl after a completed run"
    for line in workspace.run_log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        json.loads(line)  # raises if corrupted

    # We must have actually exercised the chaos path — at least one injected
    # failure (otherwise we'd be passing on the no-chaos happy path) AND at
    # least the 5 scripted-needed successes for the full FE/model/CV/submit/done
    # arc to complete.
    assert chaos_client.failures >= 1, (
        f"no failures injected — chaos client may not be wired correctly "
        f"(successes={chaos_client.successes})"
    )
    assert chaos_client.successes >= 5, (
        f"too few successes ({chaos_client.successes}) — the scripted "
        f"FE/model/train_cv/submit_local/done sequence didn't complete"
    )


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
