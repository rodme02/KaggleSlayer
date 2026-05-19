"""Real-Gemini E2E with telemetry artifacts — slow tier, opt-in.

Runs the Solver end-to-end on a synthetic micro-comp with real Gemini.
Asserts: otel.jsonl has run / llm.call / tool:<name> spans, the cost
ledger has at least one row for the competition, and (if submit_kaggle
is reached via mock) a calibration row is appended.

Costs ~$0.005-0.02 per run. Skipped when GEMINI_API_KEY is missing.
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
from kaggle_slayer.harness.telemetry import calibration
from tests.fixtures.synthetic_comp import make_synthetic_comp

load_dotenv()

pytestmark = pytest.mark.slow


@pytest.fixture
def gemini_key():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        pytest.skip("no GEMINI_API_KEY / GOOGLE_API_KEY in env")
    return key


def test_real_gemini_writes_all_telemetry_artifacts(tmp_path, gemini_key, monkeypatch):
    """Acceptance: real Gemini writes otel.jsonl + cost ledger + (if submit_kaggle ran) calibration."""
    cal_path = tmp_path / "calibration.jsonl"
    monkeypatch.setattr(calibration, "DEFAULT_PATH", cal_path)
    cost_path = tmp_path / "cost.jsonl"

    workspace = make_synthetic_comp(tmp_path / "synthetic")
    journal = Journal(workspace)
    handler = cp.CheckpointHandler(
        mode=cp.HandlerMode.STUB, journal=journal, stub_decision=cp.Decision.APPROVE
    )
    fake_kaggle = MagicMock()

    ledger = CostLedger(path=cost_path)
    llm = GeminiClient(
        api_key=gemini_key,
        ledger=ledger,
        competition=workspace.name,
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
        max_iterations=20,
        time_budget_s=900.0,
        checkpoint_handler=handler,
        kaggle_client=fake_kaggle,
    )

    # Nudge the agent to call submit_kaggle so the calibration hook fires.
    # Patch `kaggle_slayer.agent.solver.load_system_prompt` (the binding the
    # Solver actually references) — patching `prompts.load_system_prompt`
    # would NOT reach it because solver.py did
    # `from ... import load_system_prompt` at import time.
    original_loader = p_mod.load_system_prompt

    def loader_with_kaggle() -> str:
        return original_loader() + (
            "\n\n## Extra instruction for this run\n"
            "After submit_local succeeds, you MUST call submit_kaggle with "
            "csv_path pointing at the file submit_local wrote, and a 1-line "
            "message. Then call done."
        )

    with patch("kaggle_slayer.agent.solver.load_system_prompt", loader_with_kaggle):
        result = solver.solve()

    if result.status != "done" and workspace.run_log_path.exists():
        print("\n--- run_log.jsonl ---")
        print(workspace.run_log_path.read_text())
        print("--- end run_log ---")

    assert result.status == "done", (
        f"status={result.status} iters={result.iterations} summary={result.summary!r}"
    )

    # OTel trace exists
    otel_path = workspace.root / "otel.jsonl"
    assert otel_path.exists(), "otel.jsonl was not written"
    spans = [json.loads(line) for line in otel_path.read_text().splitlines() if line.strip()]
    span_names = [s["name"] for s in spans]
    assert any(n.startswith("run:") for n in span_names), f"no run span; names={span_names}"
    assert any(n == "llm.call" for n in span_names), f"no llm.call span; names={span_names}"
    assert any(n.startswith("tool:") for n in span_names), f"no tool span; names={span_names}"

    assert ledger.total_for(competition=workspace.name) > 0, "cost ledger has no rows"

    if fake_kaggle.submit.called:
        rows = calibration.read_history(competition=workspace.name, path=cal_path)
        assert len(rows) >= 1, "submit_kaggle was called but no calibration row appended"
        assert rows[0]["metric"] == "accuracy"
        assert rows[0]["cv_score"] is not None
        assert rows[0]["lb_score"] is None

    print(
        f"\nDONE iter={result.iterations} "
        f"cost=${ledger.total_for(competition=workspace.name):.4f} "
        f"otel_spans={len(spans)} kaggle_submit_called={fake_kaggle.submit.called}"
    )
