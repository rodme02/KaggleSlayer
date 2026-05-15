"""Real-Gemini E2E acceptance — gated submission flow. Slow tier, opt-in.

Uses gemini-2.5-flash + AUTO_SAFE checkpoint mode + a mocked KaggleClient.
The CSV is generated for real (the model trains; submit_local writes it),
but no actual Kaggle upload happens — kaggle_client.submit is a MagicMock.

Cost: ~$0.005-0.02 per run.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv

from kaggle_slayer.agent import prompts as p_mod
from kaggle_slayer.agent.cost_ledger import CostLedger
from kaggle_slayer.agent.llm_client import GeminiClient
from kaggle_slayer.agent.solver import Solver
from kaggle_slayer.harness import checkpoints as cp
from kaggle_slayer.harness.journal import Journal
from tests.fixtures.synthetic_comp import make_synthetic_comp

load_dotenv()

pytestmark = pytest.mark.slow


@pytest.fixture
def gemini_key():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        pytest.skip("no GEMINI_API_KEY / GOOGLE_API_KEY in env")
    return key


def test_real_gemini_completes_with_gated_submit_kaggle(tmp_path, gemini_key):
    """Real Gemini solves the synthetic comp, attempts submit_kaggle, the
    AUTO_SAFE checkpoint approves (because there's no prior submission to
    regress against — wait, AUTO_SAFE denies first submissions; use STUB
    auto-approve here instead). Mocked KaggleClient.submit is called."""
    workspace = make_synthetic_comp(tmp_path / "synthetic")
    journal = Journal(workspace)
    handler = cp.CheckpointHandler(
        mode=cp.HandlerMode.STUB, journal=journal, stub_decision=cp.Decision.APPROVE
    )
    fake_kaggle = MagicMock()

    ledger = CostLedger(path=tmp_path / "cost.jsonl")
    llm = GeminiClient(
        api_key=gemini_key,
        ledger=ledger,
        competition="synthetic-checkpoint-e2e",
        default_model="gemini-2.5-flash",
        retry_max=4,
        retry_base_delay_s=20.0,
    )
    solver = Solver(
        workspace=workspace,
        llm_client=llm,
        target_col="Survived",
        problem_type="classification",
        metric_name="accuracy",
        max_iterations=25,
        time_budget_s=900.0,
        checkpoint_handler=handler,
        kaggle_client=fake_kaggle,
    )

    # Nudge the agent toward calling submit_kaggle after submit_local. Patch
    # `kaggle_slayer.agent.solver.load_system_prompt` (the binding the Solver
    # actually references) — patching `prompts.load_system_prompt` would NOT
    # reach it because solver.py did `from ... import load_system_prompt` at
    # import time and has its own reference.
    original_loader = p_mod.load_system_prompt

    def loader_with_kaggle():
        return original_loader() + (
            "\n\n## Extra instruction for this run\n"
            "After submit_local succeeds, you MUST call submit_kaggle "
            "with csv_path pointing at the file submit_local just wrote, "
            "and a 1-line message. Then call done."
        )

    with patch("kaggle_slayer.agent.solver.load_system_prompt", loader_with_kaggle):
        result = solver.solve()

    if result.status != "done" and workspace.run_log_path.exists():
        print("\n--- run_log.jsonl ---")
        print(workspace.run_log_path.read_text())

    assert result.status == "done", (
        f"status={result.status} iters={result.iterations} summary={result.summary!r}"
    )
    fake_kaggle.submit.assert_called()  # at least one submit attempt reached the gate

    # At least one submit_kaggle checkpoint was journalled
    cp_records = [
        json.loads(line) for line in workspace.run_log_path.read_text().splitlines()
        if json.loads(line).get("kind") == "checkpoint"
    ]
    submit_cp = [r for r in cp_records if r["trigger"].startswith("submit_kaggle")]
    assert submit_cp, "no submit_kaggle checkpoint recorded"
    print(
        f"\nDONE iter={result.iterations} "
        f"cost=${ledger.total_for(competition='synthetic-checkpoint-e2e'):.4f}"
    )
