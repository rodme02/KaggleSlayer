"""Week 1 acceptance: lint passes on stubs, leak-free CV runs end-to-end.

If this passes, Week 1 is done. Any future change that breaks this test
indicates the foundations have regressed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kaggle_slayer.harness import cv as cv_mod
from kaggle_slayer.harness import sandbox
from kaggle_slayer.harness.registry import cv_strategies, metrics

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FE_STUB = FIXTURES / "fe_stub.py"
MODEL_STUB = FIXTURES / "model_stub.py"


pytestmark = pytest.mark.integration


def test_stubs_pass_ast_lint():
    fe_result = sandbox.lint_module(FE_STUB)
    model_result = sandbox.lint_module(MODEL_STUB)
    assert fe_result.ok, fe_result.violations
    assert model_result.ok, model_result.violations


def test_leak_free_cv_e2e_binary_classification(synthetic_binary):
    train, target_col = synthetic_binary
    cv = cv_strategies.auto_select(
        problem_type="classification", train_df=train, target_col=target_col
    )
    result = cv_mod.train_cv(
        fe_path=FE_STUB,
        model_path=MODEL_STUB,
        train_df=train,
        target_col=target_col,
        cv=cv,
        metric=metrics.get("accuracy"),
    )
    assert result.mean > 0.70
    assert result.metadata["cv_strategy"] == "stratified_kfold"


def test_leak_free_cv_e2e_regression(synthetic_regression):
    train, target_col = synthetic_regression
    cv = cv_strategies.auto_select(
        problem_type="regression", train_df=train, target_col=target_col
    )
    result = cv_mod.train_cv(
        fe_path=FE_STUB,
        model_path=MODEL_STUB,
        train_df=train,
        target_col=target_col,
        cv=cv,
        metric=metrics.get("rmse"),
    )
    assert result.metadata["cv_strategy"] == "kfold"
    target_std = train[target_col].std()
    assert result.mean < 0.7 * target_std


def test_leak_free_cv_records_metadata():
    """Result.metadata must surface enough for downstream MLflow logging."""
    # Build a minimal in-memory dataset
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.normal(size=200),
        "target": rng.integers(0, 2, size=200),
    })
    cv = cv_strategies.get("stratified_kfold", n_splits=3)
    result = cv_mod.train_cv(
        fe_path=FE_STUB,
        model_path=MODEL_STUB,
        train_df=df,
        target_col="target",
        cv=cv,
        metric=metrics.get("accuracy"),
    )
    assert "cv_strategy" in result.metadata
    assert "n_splits" in result.metadata
    assert "metric" in result.metadata
    assert "problem_type" in result.metadata
    assert result.duration_s > 0
