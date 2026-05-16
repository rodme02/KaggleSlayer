"""Tests for kaggle_slayer.harness.registry.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from kaggle_slayer.harness.registry import metrics


def test_get_known_metric_returns_metric_instance():
    m = metrics.get("accuracy")
    assert m.name == "accuracy"
    assert m.higher_is_better is True
    assert m.needs_proba is False


def test_get_unknown_metric_raises():
    with pytest.raises(KeyError, match="not_a_metric"):
        metrics.get("not_a_metric")


def test_list_metrics_includes_week1_set():
    names = set(metrics.list_metrics())
    assert {"accuracy", "auc", "logloss", "rmse", "mae", "r2"} <= names


def test_accuracy_perfect_predictions():
    m = metrics.get("accuracy")
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    assert m.score(y_true, y_pred) == 1.0


def test_accuracy_chance_predictions():
    m = metrics.get("accuracy")
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 0])
    assert m.score(y_true, y_pred) == 0.0


def test_auc_needs_proba_true():
    m = metrics.get("auc")
    assert m.needs_proba is True


def test_auc_perfect_separation():
    m = metrics.get("auc")
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    assert m.score(y_true, proba) == 1.0


def test_logloss_needs_proba_true():
    m = metrics.get("logloss")
    assert m.needs_proba is True
    assert m.higher_is_better is False


def test_logloss_perfect_predictions_close_to_zero():
    m = metrics.get("logloss")
    y_true = np.array([0, 1, 1, 0])
    proba = np.array([0.01, 0.99, 0.99, 0.01])
    assert m.score(y_true, proba) < 0.05


def test_rmse_zero_when_perfect():
    m = metrics.get("rmse")
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert m.score(y_true, y_pred) == 0.0
    assert m.higher_is_better is False


def test_rmse_known_value():
    m = metrics.get("rmse")
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    assert m.score(y_true, y_pred) == pytest.approx(0.5)


def test_mae_known_value():
    m = metrics.get("mae")
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    assert m.score(y_true, y_pred) == pytest.approx(0.5)


def test_r2_one_when_perfect():
    m = metrics.get("r2")
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    assert m.score(y_true, y_pred) == 1.0
    assert m.higher_is_better is True


def test_higher_is_better_pinned_for_every_week1_metric():
    """F14: regression detection in submit_kaggle keys off this field. A
    silent flip (e.g., from True to False for rmse) would let a worse model
    auto-submit under AUTO_SAFE. Pin every value so a future edit must
    update this test on purpose.
    """
    expected = {
        "accuracy": True,
        "auc": True,
        "r2": True,
        "rmse": False,
        "mae": False,
        "logloss": False,
    }
    for name, want in expected.items():
        assert metrics.get(name).higher_is_better is want, (
            f"metric {name!r}: higher_is_better expected {want}, "
            f"got {metrics.get(name).higher_is_better}"
        )
