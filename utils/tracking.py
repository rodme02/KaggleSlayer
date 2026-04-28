"""MLflow tracking integration.

Centralised so the rest of the codebase doesn't import mlflow directly. Run-name and
experiment-name follow the convention `kaggle-slayer/<competition>` and `<competition>:<timestamp>`
respectively. Tracking URI defaults to `./mlruns` (local file store) and can be overridden via
the `MLFLOW_TRACKING_URI` environment variable.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import mlflow


def _configure_tracking_uri() -> None:
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return
    # Default: local file store at <repo-root>/mlruns
    repo_root = Path.cwd()
    mlflow.set_tracking_uri(f"file:{repo_root / 'mlruns'}")


@contextmanager
def start_run(competition: str, run_name: str | None = None):
    """Open an MLflow run scoped to a competition. No-ops gracefully if MLflow unreachable."""
    _configure_tracking_uri()
    experiment = f"kaggle-slayer/{competition}"
    try:
        mlflow.set_experiment(experiment)
    except Exception:
        # Tracking server unreachable; continue silently rather than break the pipeline.
        yield None
        return

    name = run_name or f"{competition}:{int(time.time())}"
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("competition", competition)
        yield run


def log_params(params: Mapping[str, Any]) -> None:
    if not mlflow.active_run():
        return
    safe = {k: _stringify(v) for k, v in params.items()}
    mlflow.log_params(safe)


def log_metrics(metrics: Mapping[str, float]) -> None:
    if not mlflow.active_run():
        return
    safe = {k: float(v) for k, v in metrics.items() if v is not None}
    if safe:
        mlflow.log_metrics(safe)


def log_artifact(path: str | Path) -> None:
    if not mlflow.active_run():
        return
    p = Path(path)
    if p.exists():
        mlflow.log_artifact(str(p))


def set_tags(tags: Mapping[str, Any]) -> None:
    if not mlflow.active_run():
        return
    mlflow.set_tags({k: _stringify(v) for k, v in tags.items()})


def _stringify(v: Any) -> str:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return str(v)
    return repr(v)
