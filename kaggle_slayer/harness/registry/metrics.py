"""Metric registry.

A Metric is a small wrapper around a scoring function that carries the
metadata the harness needs to call it correctly: whether higher scores
are better, and whether the predictions should be probabilities or class
labels.

Week 1 metrics: accuracy, auc, logloss, rmse, mae, r2.
Additional Kaggle-specific metrics (RMSLE, QWK, MAP@K, MCRMSE, Pinball)
are added in Week 2 as the agent first needs them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class Metric:
    """A scoring function plus the metadata the harness needs to use it."""

    name: str
    higher_is_better: bool
    needs_proba: bool
    kind: Literal["classification", "regression"]
    score_fn: Callable[[np.ndarray, np.ndarray], float]

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(self.score_fn(y_true, y_pred))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


_REGISTRY: dict[str, Metric] = {
    "accuracy": Metric(
        name="accuracy",
        higher_is_better=True,
        needs_proba=False,
        kind="classification",
        score_fn=accuracy_score,
    ),
    "auc": Metric(
        name="auc",
        higher_is_better=True,
        needs_proba=True,
        kind="classification",
        score_fn=roc_auc_score,
    ),
    "logloss": Metric(
        name="logloss",
        higher_is_better=False,
        needs_proba=True,
        kind="classification",
        score_fn=log_loss,
    ),
    "rmse": Metric(
        name="rmse",
        higher_is_better=False,
        needs_proba=False,
        kind="regression",
        score_fn=_rmse,
    ),
    "mae": Metric(
        name="mae",
        higher_is_better=False,
        needs_proba=False,
        kind="regression",
        score_fn=mean_absolute_error,
    ),
    "r2": Metric(
        name="r2",
        higher_is_better=True,
        needs_proba=False,
        kind="regression",
        score_fn=r2_score,
    ),
}


def get(name: str) -> Metric:
    """Return the registered Metric for `name`. Raises KeyError if unknown."""
    if name not in _REGISTRY:
        raise KeyError(f"metric '{name}' not in registry; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_metrics() -> list[str]:
    """Return the names of all registered metrics."""
    return sorted(_REGISTRY)
