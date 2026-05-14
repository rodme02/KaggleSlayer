"""Integration test: train_cv runs end-to-end on the stub fe/model pair.

This is the Week 1 acceptance test. It is the workhorse test that proves
the harness contracts are wired together: the leak-free CV contract calls
agent-style modules with train-fold data only, evaluates with the chosen
metric, and returns a CVResult that other layers can consume.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaggle_slayer.harness import cv as cv_mod
from kaggle_slayer.harness.registry import cv_strategies, metrics

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FE_STUB = FIXTURES / "fe_stub.py"
MODEL_STUB = FIXTURES / "model_stub.py"


pytestmark = pytest.mark.integration


def test_stub_classification_beats_chance(synthetic_binary):
    train, target_col = synthetic_binary
    cv = cv_strategies.get("stratified_kfold", n_splits=5, random_state=42)
    metric = metrics.get("accuracy")
    result = cv_mod.train_cv(
        fe_path=FE_STUB,
        model_path=MODEL_STUB,
        train_df=train,
        target_col=target_col,
        cv=cv,
        metric=metric,
    )
    assert result.mean > 0.70, f"stub LR scored {result.mean:.3f} (chance is 0.50)"
    assert len(result.fold_scores) == 5
    assert result.oof.shape == (len(train),)
    assert not np.isnan(result.oof).any()


def test_stub_classification_auc_with_proba(synthetic_binary):
    train, target_col = synthetic_binary
    cv = cv_strategies.get("stratified_kfold", n_splits=5)
    metric = metrics.get("auc")
    result = cv_mod.train_cv(
        fe_path=FE_STUB,
        model_path=MODEL_STUB,
        train_df=train,
        target_col=target_col,
        cv=cv,
        metric=metric,
    )
    assert result.mean > 0.75
    assert ((result.oof >= 0.0) & (result.oof <= 1.0)).all()


def test_stub_regression_rmse(synthetic_regression):
    train, target_col = synthetic_regression
    cv = cv_strategies.get("kfold", n_splits=5)
    metric = metrics.get("rmse")
    result = cv_mod.train_cv(
        fe_path=FE_STUB,
        model_path=MODEL_STUB,
        train_df=train,
        target_col=target_col,
        cv=cv,
        metric=metric,
    )
    # Synthetic regression has noise sd ~0.3, signal sd ~2 — Ridge should beat
    # the global-mean baseline by a lot.
    target_std = train[target_col].std()
    assert result.mean < 0.7 * target_std, f"stub Ridge rmse {result.mean:.3f}"


def test_stub_auto_select_picks_right_strategy(synthetic_binary, synthetic_regression):
    train_b, target_b = synthetic_binary
    cv_b = cv_strategies.auto_select(
        problem_type="classification", train_df=train_b, target_col=target_b
    )
    assert cv_b.name == "stratified_kfold"

    train_r, target_r = synthetic_regression
    cv_r = cv_strategies.auto_select(
        problem_type="regression", train_df=train_r, target_col=target_r
    )
    assert cv_r.name == "kfold"
