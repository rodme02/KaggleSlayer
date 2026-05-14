"""Real-Gemini E2E acceptance — slow tier, opt-in.

Runs the full Solver loop against a synthetic comp with real Gemini.
Costs ~$0.01-0.05 per run depending on iteration count. Skipped when
GEMINI_API_KEY is missing.

Run with: pytest -m slow tests/integration/test_solver_real_gemini.py -v
"""

from __future__ import annotations

import os

import pandas as pd
import pytest
from dotenv import load_dotenv

from kaggle_slayer.agent.cost_ledger import CostLedger
from kaggle_slayer.agent.llm_client import GeminiClient
from kaggle_slayer.agent.solver import Solver
from tests.fixtures.synthetic_comp import make_synthetic_comp

load_dotenv()

pytestmark = pytest.mark.slow


@pytest.fixture
def gemini_key():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        pytest.skip("no GEMINI_API_KEY / GOOGLE_API_KEY in env or .env")
    return key


def test_real_gemini_solves_synthetic_microcomp(tmp_path, gemini_key):
    """Acceptance gate: real Gemini reads context.md, writes fe.py + model.py,
    runs train_cv, calls submit_local, calls done. Submission CSV must exist
    with non-empty predictions and the right row count."""
    workspace = make_synthetic_comp(tmp_path / "synthetic")
    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    llm = GeminiClient(
        api_key=gemini_key,
        ledger=ledger,
        competition="synthetic-e2e",
        default_model="gemini-2.5-pro",
        retry_max=2,
    )
    solver = Solver(
        workspace=workspace,
        llm_client=llm,
        target_col="Survived",
        problem_type="classification",
        metric_name="accuracy",
        max_iterations=20,
        time_budget_s=600.0,
    )
    result = solver.solve()

    # Hard requirements for the acceptance:
    assert result.status == "done", (
        f"Solver did not finish cleanly: status={result.status}, "
        f"iterations={result.iterations}, summary={result.summary!r}"
    )
    # The agent must have written fe.py and model.py
    assert workspace.fe_path.exists(), "agent did not write fe.py"
    assert workspace.model_path.exists(), "agent did not write model.py"
    # At least one CV pass was archived
    assert any(workspace.versions_dir.glob("fe_v*.py"))
    assert any(workspace.versions_dir.glob("model_v*.py"))
    # At least one submission CSV exists
    submissions = list(workspace.submissions_dir.glob("*.csv"))
    assert submissions, "no submission CSV was written"
    sub = pd.read_csv(submissions[0])
    assert len(sub) == 100, f"submission row count wrong: {len(sub)}"
    # Predictions must be 0/1 (label) or floats in [0,1] (proba)
    pred_col = [c for c in sub.columns if c.lower() not in ("id",)][0]
    assert sub[pred_col].notna().all(), "predictions contain NaN"
    # Cost was tracked
    assert ledger.total_for(competition="synthetic-e2e") > 0
    print(
        f"\nDONE. iter={result.iterations}, "
        f"cost=${ledger.total_for(competition='synthetic-e2e'):.4f}, "
        f"summary={result.summary!r}"
    )
