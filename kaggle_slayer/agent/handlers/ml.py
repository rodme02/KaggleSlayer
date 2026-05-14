"""ML-side tool handlers — set_cv, train_cv, submit_local.

These talk to the harness's leak-free CV via train_cv() and produce
a submission CSV via submit_local(). Both read the agent's current
fe.py and model.py from the workspace.

Before each train_cv, the current fe.py and model.py are copy-archived
into agent/versions/ as fe_v{N}.py and model_v{N}.py — this gives the
agent (and the future dashboard) a paper trail of every attempt.
"""

from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from kaggle_slayer.agent.tools import ToolError
from kaggle_slayer.harness import cv as cv_mod
from kaggle_slayer.harness.registry import cv_strategies, metrics

_ALLOWED_CV_KINDS: frozenset[str] = frozenset({"kfold", "stratified_kfold", "time_series", "group_kfold"})


def _require_files(ctx: Any) -> tuple[Path, Path]:
    fe = ctx.workspace.fe_path
    model = ctx.workspace.model_path
    if not fe.exists():
        raise ToolError("agent/fe.py not found — write it first with write_file")
    if not model.exists():
        raise ToolError("agent/model.py not found — write it first with write_file")
    return fe, model


def set_cv(ctx: Any, *, kind: str, n_splits: int = 5, group_col: str | None = None) -> str:
    """Override the CV strategy for subsequent train_cv calls."""
    if kind not in _ALLOWED_CV_KINDS:
        raise ToolError(f"unknown CV kind {kind!r}; allowed: {sorted(_ALLOWED_CV_KINDS)}")
    ctx.cv_kind = kind
    params: dict[str, Any] = {"n_splits": n_splits}
    if group_col is not None:
        params["group_col"] = group_col
    ctx.cv_params = params
    return f"set cv strategy: {kind} with {params}"


def _build_cv_strategy(ctx: Any) -> Any:
    """Either honor ctx.cv_kind/cv_params or auto_select from problem_type."""
    if getattr(ctx, "cv_kind", None):
        return cv_strategies.get(ctx.cv_kind, **ctx.cv_params)
    return cv_strategies.auto_select(
        problem_type=ctx.problem_type,
        train_df=pd.read_csv(ctx.workspace.raw_dir / "train.csv", nrows=5),
        target_col=ctx.target_col,
    )


def train_cv(ctx: Any) -> str:
    """Archive fe.py + model.py into versions/, then run leak-free CV."""
    fe_path, model_path = _require_files(ctx)
    # Archive
    fe_archive = ctx.workspace.next_version_path("fe")
    model_archive = ctx.workspace.next_version_path("model")
    shutil.copyfile(fe_path, fe_archive)
    shutil.copyfile(model_path, model_archive)

    train_df = pd.read_csv(ctx.workspace.raw_dir / "train.csv")
    cv = _build_cv_strategy(ctx)
    metric = metrics.get(ctx.metric_name)

    result = cv_mod.train_cv(
        fe_path=fe_path,
        model_path=model_path,
        train_df=train_df,
        target_col=ctx.target_col,
        cv=cv,
        metric=metric,
        metadata_extra={"fe_version": fe_archive.stem, "model_version": model_archive.stem},
    )

    summary = (
        f"train_cv complete: {cv.name} ({cv.n_splits} folds), metric={metric.name}, "
        f"mean={result.mean:.4f}, std={result.std:.4f}, fold_scores={[round(s, 4) for s in result.fold_scores]}, "
        f"duration_s={result.duration_s:.2f}"
    )
    return summary


def submit_local(ctx: Any, *, label: str) -> str:
    """Fit fe.py + model.py on the full train set; predict test; write submission CSV."""
    fe_path, model_path = _require_files(ctx)

    train_df = pd.read_csv(ctx.workspace.raw_dir / "train.csv")
    test_path = ctx.workspace.raw_dir / "test.csv"
    if not test_path.exists():
        raise ToolError(f"test.csv not found at {test_path}")
    test_df = pd.read_csv(test_path)

    # Detect id column — use whichever of {id, Id, ID, PassengerId, ...} appears in test.
    id_col = _detect_id_column(test_df)

    # Load agent modules and fit on the FULL train set.
    import importlib.util

    spec = importlib.util.spec_from_file_location("_agent_fe_final", fe_path)
    if spec is None or spec.loader is None:
        raise ToolError(f"cannot load {fe_path}")
    fe_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fe_mod)

    spec = importlib.util.spec_from_file_location("_agent_model_final", model_path)
    if spec is None or spec.loader is None:
        raise ToolError(f"cannot load {model_path}")
    model_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_mod)

    fe = fe_mod.fit_feature_transformer(train_df, ctx.target_col)
    X_train = fe.transform(train_df.drop(columns=[ctx.target_col]))
    y_train = train_df[ctx.target_col].to_numpy()

    test_for_transform = test_df.drop(columns=[id_col]) if id_col else test_df
    X_test = fe.transform(test_for_transform)

    metric = metrics.get(ctx.metric_name)
    model = model_mod.fit_model(X_train, y_train, ctx.problem_type, metric.name)

    if metric.needs_proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        preds = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba
    else:
        preds = model.predict(X_test)

    # Build submission DataFrame
    out = pd.DataFrame()
    if id_col is not None:
        out[id_col] = test_df[id_col].values
    out[ctx.target_col] = preds

    stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d_%H%M%S")
    out_path = ctx.workspace.submissions_dir / f"{stamp}_{label}.csv"
    out.to_csv(out_path, index=False)
    return f"wrote submission ({len(out)} rows) to {out_path.relative_to(ctx.workspace.root)}"


def _detect_id_column(df: pd.DataFrame) -> str | None:
    for candidate in ("id", "Id", "ID", "PassengerId", "index"):
        if candidate in df.columns:
            return candidate
    # Heuristic: first column whose name ends in "id"
    first = str(df.columns[0])
    if first.lower().endswith("id"):
        return first
    return None


def done(ctx: Any, *, summary: str) -> str:
    """Signal that the agent is finished. The Solver loop exits after this returns."""
    ctx.finished = True
    ctx.final_summary = summary
    return f"acknowledged: {summary}"
