"""MLflow logging for train_cv — one run per call.

Spec section 11.1: one experiment per competition (`kaggleslayer/<comp>`),
one run per `train_cv` invocation. Params: cv_strategy, metric, fe_version,
model_version. Metrics: cv_mean, cv_std, fold_0...fold_N.

Artifact logging (fe.py, model.py, oof_preds.npy) lands in Week 6.

`log_train_cv(...)` is a context manager. Errors from MLflow are
swallowed (the agent shouldn't crash if the tracking server is down) —
they're logged via stdlib logging so the operator can find them.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import mlflow

_log = logging.getLogger(__name__)


@dataclass
class _RunLogger:
    """Handed to the caller inside the `with` block; collects results."""

    _logged: bool = field(default=False, init=False)

    def log_result(
        self,
        *,
        cv_mean: float,
        cv_std: float,
        fold_scores: list[float],
    ) -> None:
        if self._logged:
            return  # idempotent
        try:
            metrics: dict[str, float] = {
                "cv_mean": float(cv_mean),
                "cv_std": float(cv_std),
            }
            for i, score in enumerate(fold_scores):
                metrics[f"fold_{i}"] = float(score)
            mlflow.log_metrics(metrics)
        except Exception:  # noqa: BLE001
            _log.exception("mlflow.log_metrics failed; continuing")
        self._logged = True


@contextlib.contextmanager
def log_train_cv(
    *,
    competition: str,
    cv_strategy: str,
    metric: str,
    fe_version: str,
    model_version: str,
) -> Iterator[_RunLogger]:
    """Wrap one train_cv invocation in an MLflow run."""
    logger = _RunLogger()
    try:
        mlflow.set_experiment(f"kaggleslayer/{competition}")
        with mlflow.start_run():
            mlflow.log_params({
                "cv_strategy": cv_strategy,
                "metric": metric,
                "fe_version": fe_version,
                "model_version": model_version,
            })
            yield logger
    except Exception:  # noqa: BLE001
        # MLflow itself blew up. Still yield a logger so caller code is identical;
        # log_result will fail silently inside.
        _log.exception("mlflow.start_run / set_experiment failed; metrics not recorded")
        yield logger
