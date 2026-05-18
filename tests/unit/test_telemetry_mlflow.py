"""Tests for kaggle_slayer.harness.telemetry.mlflow_logger."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kaggle_slayer.harness.telemetry import mlflow_logger
from kaggle_slayer.harness.workspace import Workspace


@pytest.fixture
def ws(tmp_path):
    return Workspace.create(root=tmp_path / "comp")


def test_log_train_cv_starts_and_ends_run(ws):
    """The context manager calls mlflow.start_run + log_params + log_metrics."""
    with patch("kaggle_slayer.harness.telemetry.mlflow_logger.mlflow") as mock_ml:
        mock_run = MagicMock()
        mock_ml.start_run.return_value.__enter__.return_value = mock_run
        mock_ml.start_run.return_value.__exit__.return_value = None

        with mlflow_logger.log_train_cv(
            competition="titanic",
            cv_strategy="stratified_kfold",
            metric="accuracy",
            fe_version="fe_v01",
            model_version="model_v01",
        ) as logger:
            logger.log_result(cv_mean=0.82, cv_std=0.03, fold_scores=[0.80, 0.83, 0.83])

    mock_ml.start_run.assert_called_once()
    mock_ml.log_params.assert_called_once()
    params = mock_ml.log_params.call_args[0][0]
    assert params["cv_strategy"] == "stratified_kfold"
    assert params["metric"] == "accuracy"
    assert params["fe_version"] == "fe_v01"
    assert params["model_version"] == "model_v01"

    mock_ml.log_metrics.assert_called()
    metrics = mock_ml.log_metrics.call_args[0][0]
    assert metrics["cv_mean"] == 0.82
    assert metrics["cv_std"] == 0.03
    assert metrics["fold_0"] == 0.80
    assert metrics["fold_1"] == 0.83
    assert metrics["fold_2"] == 0.83


def test_log_train_cv_sets_experiment_per_competition(ws):
    with patch("kaggle_slayer.harness.telemetry.mlflow_logger.mlflow") as mock_ml:
        mock_ml.start_run.return_value.__enter__.return_value = MagicMock()
        mock_ml.start_run.return_value.__exit__.return_value = None

        with mlflow_logger.log_train_cv(
            competition="titanic", cv_strategy="kfold", metric="accuracy",
            fe_version="fe_v01", model_version="model_v01",
        ):
            pass

    mock_ml.set_experiment.assert_called_once_with("kaggleslayer/titanic")


def test_log_train_cv_swallows_mlflow_failures(ws):
    """If mlflow.start_run raises, the user's code still runs — we don't break the agent."""
    with patch("kaggle_slayer.harness.telemetry.mlflow_logger.mlflow") as mock_ml:
        mock_ml.set_experiment.side_effect = RuntimeError("mlflow down")

        # Should NOT raise — failure is logged via the noop fallback.
        with mlflow_logger.log_train_cv(
            competition="x", cv_strategy="kfold", metric="rmse",
            fe_version="fe_v01", model_version="model_v01",
        ) as logger:
            logger.log_result(cv_mean=0.5, cv_std=0.1, fold_scores=[0.4, 0.6])
