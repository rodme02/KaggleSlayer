"""Tests for kaggle_slayer.harness.telemetry.mlflow_logger."""

from __future__ import annotations

from pathlib import Path
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
            problem_type="classification",
        ) as logger:
            logger.log_result(
                cv_mean=0.82,
                cv_std=0.03,
                fold_scores=[0.80, 0.83, 0.83],
                wall_clock_s=12.5,
            )

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
    assert metrics["wall_clock_s"] == 12.5


def test_log_train_cv_sets_experiment_per_competition(ws):
    with patch("kaggle_slayer.harness.telemetry.mlflow_logger.mlflow") as mock_ml:
        mock_ml.start_run.return_value.__enter__.return_value = MagicMock()
        mock_ml.start_run.return_value.__exit__.return_value = None

        with mlflow_logger.log_train_cv(
            competition="titanic", cv_strategy="kfold", metric="accuracy",
            fe_version="fe_v01", model_version="model_v01",
            problem_type="classification",
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
            problem_type="regression",
        ) as logger:
            logger.log_result(
                cv_mean=0.5,
                cv_std=0.1,
                fold_scores=[0.4, 0.6],
                wall_clock_s=3.2,
            )


def test_log_train_cv_sets_default_tracking_uri_when_env_unset(ws, monkeypatch):
    """F5: when MLFLOW_TRACKING_URI is unset, set_tracking_uri must land at
    file:~/.kaggle_slayer/mlruns so runs land in a consistent place."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with patch("kaggle_slayer.harness.telemetry.mlflow_logger.mlflow") as mock_ml:
        mock_ml.start_run.return_value.__enter__.return_value = MagicMock()
        mock_ml.start_run.return_value.__exit__.return_value = None

        with mlflow_logger.log_train_cv(
            competition="t", cv_strategy="kfold", metric="accuracy",
            fe_version="fe_v01", model_version="model_v01",
            problem_type="classification",
        ):
            pass

    expected = f"file:{Path.home() / '.kaggle_slayer' / 'mlruns'}"
    mock_ml.set_tracking_uri.assert_called_once_with(expected)


def test_log_train_cv_respects_env_tracking_uri(ws, monkeypatch):
    """F5: when MLFLOW_TRACKING_URI is set, we must NOT clobber it."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://my-mlflow.example.com")
    with patch("kaggle_slayer.harness.telemetry.mlflow_logger.mlflow") as mock_ml:
        mock_ml.start_run.return_value.__enter__.return_value = MagicMock()
        mock_ml.start_run.return_value.__exit__.return_value = None

        with mlflow_logger.log_train_cv(
            competition="t", cv_strategy="kfold", metric="accuracy",
            fe_version="fe_v01", model_version="model_v01",
            problem_type="classification",
        ):
            pass

    mock_ml.set_tracking_uri.assert_not_called()


def test_log_train_cv_sets_tags_with_competition_and_problem_type(ws):
    """F6: spec §11.1 calls for tags; we surface kaggle_competition and problem_type."""
    with patch("kaggle_slayer.harness.telemetry.mlflow_logger.mlflow") as mock_ml:
        mock_ml.start_run.return_value.__enter__.return_value = MagicMock()
        mock_ml.start_run.return_value.__exit__.return_value = None

        with mlflow_logger.log_train_cv(
            competition="titanic", cv_strategy="stratified_kfold", metric="accuracy",
            fe_version="fe_v01", model_version="model_v01",
            problem_type="classification",
        ):
            pass

    mock_ml.set_tags.assert_called_once()
    tags = mock_ml.set_tags.call_args[0][0]
    assert tags["kaggle_competition"] == "titanic"
    assert tags["problem_type"] == "classification"


def test_log_train_cv_routes_swallowed_errors_to_workspace_file(ws):
    """F4: when workspace is passed and mlflow blows up, a line lands in
    <workspace>/mlflow_errors.log so the operator can find it post-hoc."""
    with patch("kaggle_slayer.harness.telemetry.mlflow_logger.mlflow") as mock_ml:
        mock_ml.set_experiment.side_effect = RuntimeError("mlflow boom")

        with mlflow_logger.log_train_cv(
            competition="x", cv_strategy="kfold", metric="rmse",
            fe_version="fe_v01", model_version="model_v01",
            problem_type="regression",
            workspace=ws,
        ) as logger:
            logger.log_result(
                cv_mean=0.5, cv_std=0.1, fold_scores=[0.4, 0.6], wall_clock_s=1.0,
            )

    err_path = ws.root / "mlflow_errors.log"
    assert err_path.exists()
    body = err_path.read_text()
    assert "mlflow boom" in body or "RuntimeError" in body


def test_log_train_cv_inner_log_metrics_failure_routes_to_workspace_file(ws):
    """F4: when log_metrics raises, the error also lands in mlflow_errors.log
    if a workspace was provided."""
    with patch("kaggle_slayer.harness.telemetry.mlflow_logger.mlflow") as mock_ml:
        mock_ml.start_run.return_value.__enter__.return_value = MagicMock()
        mock_ml.start_run.return_value.__exit__.return_value = None
        mock_ml.log_metrics.side_effect = RuntimeError("metric write failed")

        with mlflow_logger.log_train_cv(
            competition="x", cv_strategy="kfold", metric="accuracy",
            fe_version="fe_v01", model_version="model_v01",
            problem_type="classification",
            workspace=ws,
        ) as logger:
            logger.log_result(
                cv_mean=0.5, cv_std=0.1, fold_scores=[0.4, 0.6], wall_clock_s=1.0,
            )

    err_path = ws.root / "mlflow_errors.log"
    assert err_path.exists()
    body = err_path.read_text()
    assert "metric write failed" in body or "RuntimeError" in body


def test_log_train_cv_no_workspace_falls_back_to_log_only(ws):
    """F4: when no workspace is provided, no mlflow_errors.log is written —
    we still rely on stdlib logging."""
    with patch("kaggle_slayer.harness.telemetry.mlflow_logger.mlflow") as mock_ml:
        mock_ml.set_experiment.side_effect = RuntimeError("nope")

        # No workspace kwarg → no file written, no crash.
        with mlflow_logger.log_train_cv(
            competition="x", cv_strategy="kfold", metric="accuracy",
            fe_version="fe_v01", model_version="model_v01",
            problem_type="classification",
        ) as logger:
            logger.log_result(
                cv_mean=0.5, cv_std=0.1, fold_scores=[0.4, 0.6], wall_clock_s=1.0,
            )

    # No file lookup needed — only confirm no crash. We DO confirm the
    # workspace-less branch wrote no file at the workspace path either.
    assert not (ws.root / "mlflow_errors.log").exists()

def test_log_train_cv_caller_exception_propagates_unmasked(ws):
    """A failing train_cv inside the block must surface ITS error — not
    RuntimeError("generator didn't stop after throw()"). The agent reads the
    error text to self-correct; masking it breaks the feedback loop."""
    with patch("kaggle_slayer.harness.telemetry.mlflow_logger.mlflow"):
        with pytest.raises(ValueError, match="boom"):
            with mlflow_logger.log_train_cv(
                competition="x", cv_strategy="kfold", metric="rmse",
                fe_version="fe_v01", model_version="model_v01",
                problem_type="regression", workspace=ws,
            ):
                raise ValueError("boom")
