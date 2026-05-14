"""The leak-free CV contract.

`train_cv` is the only path to a CV score. The harness loads the agent's
fe.py and model.py from disk, runs the configured CV strategy, and on each
fold calls the agent's fit_feature_transformer with ONLY that fold's
training data. The val fold is transformed but never seen at fit time.

This is the temporal version of the leak-free guarantee — see spec §6 for
context. The legacy V1 used a sklearn TransformerMixin (a structural
guarantee), which was easy to subvert by accident.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from kaggle_slayer.harness.registry.cv_strategies import CVStrategy
from kaggle_slayer.harness.registry.metrics import Metric


class CVError(Exception):
    """Raised when the leak-free CV contract is violated or its inputs are bad."""


@dataclass
class CVResult:
    """Result of a single train_cv invocation."""

    fold_scores: list[float]
    mean: float
    std: float
    oof: np.ndarray
    duration_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _load_module(path: Path, name: str) -> ModuleType:
    """Dynamically import the agent's module from disk."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CVError(f"cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _require_callable(mod: ModuleType, name: str, where: Path) -> Any:
    fn = getattr(mod, name, None)
    if not callable(fn):
        raise CVError(f"module {where} missing required callable '{name}'")
    return fn


def _infer_problem_type(metric: Metric, target: pd.Series) -> str:  # noqa: ARG001
    """Use the metric's declared kind as the primary (and for Week-1, sole) signal.

    The `target` parameter is retained for forward compatibility — Week 2 may
    introduce ambiguous metrics where the target dtype or cardinality is needed
    as a tiebreaker. For all Week-1 metrics every Metric has an explicit kind
    field, so the target is unused here.
    """
    return metric.kind


def train_cv(
    *,
    fe_path: str | Path,
    model_path: str | Path,
    train_df: pd.DataFrame,
    target_col: str,
    cv: CVStrategy,
    metric: Metric,
) -> CVResult:
    """Run leak-free K-fold CV using the agent's fe.py and model.py.

    Args:
        fe_path: Path to a Python file exposing `fit_feature_transformer(train_df, target_col)`.
        model_path: Path to a Python file exposing `fit_model(X, y, problem_type, metric_name)`.
        train_df: The training dataframe with the target column.
        target_col: Name of the target column.
        cv: CV strategy from `cv_strategies.get(...)`.
        metric: Metric from `metrics.get(...)`.

    Returns:
        CVResult with per-fold scores, mean, std, OOF predictions, and timing.

    Raises:
        CVError: On contract violations (missing fit functions, row count mismatch,
            module load failures).
    """
    fe_path = Path(fe_path)
    model_path = Path(model_path)

    fe_mod = _load_module(fe_path, "_agent_fe")
    model_mod = _load_module(model_path, "_agent_model")
    fit_fe = _require_callable(fe_mod, "fit_feature_transformer", fe_path)
    fit_model = _require_callable(model_mod, "fit_model", model_path)

    problem_type = _infer_problem_type(metric, train_df[target_col])
    n = len(train_df)
    # OOF dtype: float for proba/regression, int for class labels.
    oof = np.full(n, np.nan, dtype=float)

    fold_scores: list[float] = []
    started = time.perf_counter()

    for fold_i, (train_idx, val_idx) in enumerate(cv.split(train_df, target_col)):
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]

        fe = fit_fe(train_fold, target_col)
        if not hasattr(fe, "transform"):
            raise CVError(
                f"fit_feature_transformer in {fe_path} must return an object with "
                f".transform(df) (got {type(fe).__name__})"
            )

        X_train = fe.transform(train_fold.drop(columns=[target_col]))
        X_val = fe.transform(val_fold.drop(columns=[target_col]))

        if len(X_train) != len(train_fold):
            raise CVError(
                f"fe.transform on train fold {fold_i}: rows changed "
                f"({len(train_fold)} -> {len(X_train)})"
            )
        if len(X_val) != len(val_fold):
            raise CVError(
                f"fe.transform on val fold {fold_i}: rows changed "
                f"({len(val_fold)} -> {len(X_val)})"
            )

        y_train = train_fold[target_col].to_numpy()
        y_val = val_fold[target_col].to_numpy()

        model = fit_model(X_train, y_train, problem_type, metric.name)

        if metric.needs_proba:
            if not hasattr(model, "predict_proba"):
                raise CVError(
                    f"metric '{metric.name}' needs probabilities but model from "
                    f"{model_path} has no .predict_proba"
                )
            proba = model.predict_proba(X_val)
            if proba.ndim == 2 and proba.shape[1] == 2:
                preds = proba[:, 1]  # binary: positive-class proba
            else:
                preds = proba  # multi-class: keep matrix
        else:
            preds = model.predict(X_val)

        preds_arr = np.asarray(preds, dtype=float)

        # Lazily widen OOF on the first fold if predictions are 2-D.
        if preds_arr.ndim == 2:
            if oof.ndim == 1:
                oof = np.full((n, preds_arr.shape[1]), np.nan, dtype=float)
            elif oof.shape[1] != preds_arr.shape[1]:
                raise CVError(
                    f"fold {fold_i}: prediction width changed "
                    f"({oof.shape[1]} -> {preds_arr.shape[1]})"
                )
            oof[val_idx] = preds_arr
        elif preds_arr.ndim == 1:
            oof[val_idx] = preds_arr
        else:
            raise CVError(
                f"unsupported prediction shape {preds_arr.shape}"
            )
        fold_scores.append(metric.score(y_val, preds))

    duration_s = time.perf_counter() - started
    return CVResult(
        fold_scores=fold_scores,
        mean=float(np.mean(fold_scores)),
        std=float(np.std(fold_scores)),
        oof=oof,
        duration_s=duration_s,
        metadata={
            "cv_strategy": cv.name,
            "n_splits": cv.n_splits,
            "metric": metric.name,
            "problem_type": problem_type,
        },
    )
