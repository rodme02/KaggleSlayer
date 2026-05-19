"""MLflow logging for train_cv — one run per call.

Spec section 11.1: one experiment per competition (`kaggleslayer/<comp>`),
one run per `train_cv` invocation. Params: cv_strategy, metric, fe_version,
model_version. Tags: kaggle_competition, problem_type. Metrics: cv_mean,
cv_std, fold_0...fold_N, wall_clock_s.

Artifact logging (fe.py, model.py, oof_preds.npy) lands in Week 6.

`log_train_cv(...)` is a context manager. Errors from MLflow are
swallowed (the agent shouldn't crash if the tracking server is down) —
they're logged via stdlib logging so the operator can find them. If a
``workspace`` is provided, swallowed errors are also appended to
``<workspace>/mlflow_errors.log`` so the operator can find them post-hoc
without grepping stdlib logs.
"""

from __future__ import annotations

import contextlib
import logging
import os
import traceback
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import mlflow

from kaggle_slayer.harness.workspace import Workspace

_log = logging.getLogger(__name__)

_ERROR_LOG_FILENAME = "mlflow_errors.log"


def _record_error(workspace: Workspace | None, context: str, exc: BaseException) -> None:
    """Append an exception to the per-workspace mlflow_errors.log if available.

    The stdlib logging path is preserved by the caller (which uses
    ``_log.exception``). This helper is intentionally best-effort: if
    appending to the file itself fails, it's logged at ``warning`` and
    swallowed — the operator already has the upstream exception via stdlib.
    """
    if workspace is None:
        return
    err_path = workspace.root / _ERROR_LOG_FILENAME
    try:
        err_path.parent.mkdir(parents=True, exist_ok=True)
        formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        with err_path.open("a") as f:
            f.write(f"--- {context} ---\n")
            f.write(formatted)
            if not formatted.endswith("\n"):
                f.write("\n")
    except OSError:
        _log.warning("failed to append to %s", err_path, exc_info=True)


@dataclass
class _RunLogger:
    """Handed to the caller inside the `with` block; collects results."""

    workspace: Workspace | None = None
    _logged: bool = field(default=False, init=False)

    def log_result(
        self,
        *,
        cv_mean: float,
        cv_std: float,
        fold_scores: list[float],
        wall_clock_s: float,
    ) -> None:
        if self._logged:
            return  # idempotent
        try:
            metrics: dict[str, float] = {
                "cv_mean": float(cv_mean),
                "cv_std": float(cv_std),
                "wall_clock_s": float(wall_clock_s),
            }
            for i, score in enumerate(fold_scores):
                metrics[f"fold_{i}"] = float(score)
            mlflow.log_metrics(metrics)
        except Exception as exc:  # noqa: BLE001
            _log.exception("mlflow.log_metrics failed; continuing")
            _record_error(self.workspace, "mlflow.log_metrics failed", exc)
        self._logged = True


def _resolve_tracking_uri() -> str | None:
    """Return the tracking URI to set, or None if the env already controls it.

    Env precedence: if MLFLOW_TRACKING_URI is set we leave mlflow alone.
    Otherwise, point at file:~/.kaggle_slayer/mlruns so two kaggle-slayer
    invocations from different cwds share one experiment store.
    """
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return None
    return f"file:{Path.home() / '.kaggle_slayer' / 'mlruns'}"


@contextlib.contextmanager
def log_train_cv(
    *,
    competition: str,
    cv_strategy: str,
    metric: str,
    fe_version: str,
    model_version: str,
    problem_type: str,
    workspace: Workspace | None = None,
) -> Iterator[_RunLogger]:
    """Wrap one train_cv invocation in an MLflow run.

    Args:
        competition: The Kaggle competition slug — used as the experiment
            suffix and as the ``kaggle_competition`` tag.
        cv_strategy: Name of the CV strategy (e.g. ``stratified_kfold``).
        metric: Name of the scoring metric (e.g. ``accuracy``).
        fe_version: Version stem of the archived fe.py (e.g. ``fe_v03``).
        model_version: Version stem of the archived model.py.
        problem_type: ``classification`` / ``regression`` / ``ranking`` —
            surfaced as a tag so the dashboard can filter across comps.
        workspace: Optional. If provided, swallowed MLflow exceptions are
            appended to ``<workspace>/mlflow_errors.log`` in addition to
            stdlib logging. The agent loop never crashes on tracking
            errors either way.
    """
    logger = _RunLogger(workspace=workspace)
    tracking_uri = _resolve_tracking_uri()
    if tracking_uri is not None:
        try:
            mlflow.set_tracking_uri(tracking_uri)
        except Exception as exc:  # noqa: BLE001
            _log.exception("mlflow.set_tracking_uri failed; continuing")
            _record_error(workspace, "mlflow.set_tracking_uri failed", exc)
    try:
        mlflow.set_experiment(f"kaggleslayer/{competition}")
        # mlflow.start_run() ends the run on context exit, even if log_params raises.
        with mlflow.start_run():
            mlflow.set_tags({
                "kaggle_competition": competition,
                "problem_type": problem_type,
            })
            mlflow.log_params({
                "cv_strategy": cv_strategy,
                "metric": metric,
                "fe_version": fe_version,
                "model_version": model_version,
            })
            yield logger
    except Exception as exc:  # noqa: BLE001
        # MLflow itself blew up. Still yield a logger so caller code is identical;
        # log_result will fail silently inside.
        _log.exception("mlflow.start_run / set_experiment failed; metrics not recorded")
        _record_error(workspace, "mlflow.start_run / set_experiment failed", exc)
        yield logger
