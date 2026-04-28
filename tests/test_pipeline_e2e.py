"""End-to-end smoke test: coordinator runs on a synthetic competition and produces submission.csv.

Marked `slow` so it can be skipped locally (`pytest -m "not slow"`).
"""

from __future__ import annotations

import pandas as pd
import pytest

from agents.coordinator import PipelineCoordinator


@pytest.mark.slow
def test_coordinator_end_to_end(competition_dir):
    coord = PipelineCoordinator(competition_dir.name, competition_dir)
    results = coord.run(submit_to_kaggle=False)

    assert results["pipeline_status"] == "completed"
    assert "best_model" in results
    assert results["final_score"] is not None

    submission = competition_dir / "submission.csv"
    assert submission.exists(), "submission.csv was not created"

    df = pd.read_csv(submission)
    assert len(df) == 40  # 200 rows total, 160 train / 40 test
    assert list(df.columns) == ["PassengerId", "Survived"]
