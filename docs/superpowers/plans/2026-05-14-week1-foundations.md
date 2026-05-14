# Week 1 — Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the trusted side of the harness: the leak-free CV contract, metric and CV-strategy registries, and the AST sandbox lint. Land it as the first commit on a new `v2-rebuild` branch with the old code archived under `legacy/`.

**Architecture:** New package `kaggle_slayer/` with `harness/` subpackage. The harness owns CV correctness; it loads agent-written `fe.py`/`model.py` modules and runs CV with train-fold data only. Registries are simple `name → factory` maps. Sandbox lint is an AST walk that rejects forbidden patterns at module-load time.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn, pytest, ruff, mypy, pytest-cov. (Google GenAI SDK, OpenTelemetry, MLflow — added in later weeks but listed in `pyproject.toml` now to avoid churn.)

**Acceptance (from spec §14.3):** Unit tier green; leak-free CV runs on a stub `fe.py` + `model.py` pair via the `train_cv` contract.

---

## File map

**Files created this week:**
- `kaggle_slayer/__init__.py`
- `kaggle_slayer/harness/__init__.py`
- `kaggle_slayer/harness/cv.py` — leak-free CV contract
- `kaggle_slayer/harness/sandbox.py` — AST lint (resource limits deferred to Week 4)
- `kaggle_slayer/harness/registry/__init__.py`
- `kaggle_slayer/harness/registry/metrics.py` — Metric protocol + 6 metrics
- `kaggle_slayer/harness/registry/cv_strategies.py` — CVStrategy protocol + 2 strategies
- `tests/__init__.py`
- `tests/conftest.py` — synthetic fixtures
- `tests/unit/__init__.py`
- `tests/unit/test_metrics_registry.py`
- `tests/unit/test_cv_strategies_registry.py`
- `tests/unit/test_cv_contract.py`
- `tests/unit/test_sandbox_lint.py`
- `tests/integration/__init__.py`
- `tests/integration/test_cv_with_stubs.py` — Week 1 acceptance test
- `tests/fixtures/__init__.py`
- `tests/fixtures/fe_stub.py`
- `tests/fixtures/model_stub.py`

**Files modified:**
- `pyproject.toml` — new package layout, new deps, dev deps including mypy
- `.github/workflows/ci.yml` — extend to v2-rebuild, mypy on harness, coverage, tighten ruff

**Files moved to `legacy/` (Task 1):**
- `agents/` → `legacy/agents/`
- `core/` → `legacy/core/`
- `utils/` → `legacy/utils/`
- `dashboard/` → `legacy/dashboard/`
- `scripts/` → `legacy/scripts/`
- `tests/` → `legacy/tests/`
- `kaggle_slayer.py` → `legacy/kaggle_slayer.py`

---

## Task 1: Branch `v2-rebuild` and archive existing source under `legacy/`

**Files:**
- Move (via `git mv`): everything listed in "Files moved to `legacy/`" above
- No new files created in this task

- [ ] **Step 1: Create and switch to `v2-rebuild` branch**

```bash
git checkout -b v2-rebuild
git status
```

Expected: `On branch v2-rebuild`, working tree clean.

- [ ] **Step 2: Create the `legacy/` directory and move all existing source**

```bash
mkdir -p legacy
git mv agents legacy/
git mv core legacy/
git mv utils legacy/
git mv dashboard legacy/
git mv scripts legacy/
git mv tests legacy/
git mv kaggle_slayer.py legacy/
```

- [ ] **Step 3: Verify what moved and what remains at top level**

```bash
ls -F
git status
```

Expected at top level (besides `legacy/`): `.dockerignore`, `.git*`, `.github/`, `.gitignore`, `.pre-commit-config.yaml`, `.pytest_cache/`, `.ruff_cache/`, `CLAUDE.md`, `Dockerfile`, `LICENSE`, `PORTFOLIO_STANDARD.md`, `README.md`, `config.yaml`, `docker-compose.yml`, `docs/`, `legacy/`, `pyproject.toml`, `requirements.txt`.
`git status` should show all moves as renames.

- [ ] **Step 4: Commit the archive**

```bash
git commit -m "$(cat <<'EOF'
chore: archive V1 source under legacy/

First commit on v2-rebuild. Moves all V1 Python sources (agents, core,
utils, dashboard, scripts, tests, kaggle_slayer.py) under legacy/ so they
remain visible for reference while the V2 LLM-agent harness is built
alongside. Blame history preserved by using git mv.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git log --oneline -3
```

Expected: new commit "chore: archive V1 source under legacy/" at the top.

---

## Task 2: Rewrite `pyproject.toml` for the new package layout

**Files:**
- Modify: `pyproject.toml`

The new pyproject declares `kaggle_slayer` as a proper package (not a top-level module), adds the V2 runtime dependencies (Google GenAI SDK, OpenTelemetry, jsonschema), and adds `mypy` to dev deps.

- [ ] **Step 1: Write the failing test for the new entry point**

Create `tests/unit/test_package_layout.py`:

```python
"""Sanity checks that the new package layout is importable."""

def test_top_level_package_importable():
    import kaggle_slayer
    assert kaggle_slayer.__name__ == "kaggle_slayer"


def test_harness_subpackage_importable():
    import kaggle_slayer.harness
    assert kaggle_slayer.harness.__name__ == "kaggle_slayer.harness"


def test_harness_registry_importable():
    import kaggle_slayer.harness.registry
    assert kaggle_slayer.harness.registry.__name__ == "kaggle_slayer.harness.registry"
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
pytest tests/unit/test_package_layout.py -v
```

Expected: collection error or `ModuleNotFoundError: No module named 'kaggle_slayer'`.

- [ ] **Step 3: Replace `pyproject.toml`**

Overwrite the existing `pyproject.toml` with:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "kaggle-slayer"
version = "2.0.0a1"
description = "LLM-agent harness for tabular Kaggle competitions with leak-free CV."
readme = "README.md"
requires-python = ">=3.11"
license = { file = "LICENSE" }
authors = [{ name = "Rodrigo Medeiros" }]
keywords = ["kaggle", "llm-agent", "automl", "mlops", "tabular"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "scipy>=1.10",
    "pyarrow>=14.0",
    "pyyaml>=6.0",
    "joblib>=1.3",
    "rich>=13.0",
    "kaggle>=1.6",
    "mlflow>=2.10",
    "google-genai>=0.3",
    "opentelemetry-api>=1.24",
    "opentelemetry-sdk>=1.24",
    "jsonschema>=4.21",
    "psutil>=5.9",
    "xgboost>=2.0",
    "lightgbm>=4.0",
    "catboost>=1.2",
    "optuna>=3.5",
]

[project.optional-dependencies]
dashboard = [
    "streamlit>=1.30",
    "plotly>=5.18",
]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.1",
    "ruff>=0.4",
    "mypy>=1.8",
    "pre-commit>=3.6",
]

[project.scripts]
kaggle-slayer = "kaggle_slayer.cli:main"

[project.urls]
Homepage = "https://github.com/rodme02/KaggleSlayer"
Repository = "https://github.com/rodme02/KaggleSlayer"

[tool.setuptools.packages.find]
include = ["kaggle_slayer*"]
exclude = ["tests*", "legacy*", "competitions*", "docs*"]

[tool.ruff]
line-length = 110
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "SIM"]
ignore = ["E501", "B008"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["B", "SIM"]
"legacy/**" = ["E", "F", "W", "I", "B", "UP", "SIM"]

[tool.mypy]
python_version = "3.11"
files = ["kaggle_slayer/harness"]
strict = true
warn_unused_ignores = true
warn_redundant_casts = true
exclude = ["legacy/.*", "tests/.*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-q --strict-markers"
markers = [
    "slow: end-to-end / slower tests (run with -m slow)",
    "integration: integration tests with fake agent (faster than slow)",
]
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore::FutureWarning",
]
```

- [ ] **Step 4: Install the package in editable mode**

```bash
pip install -e ".[dev,dashboard]"
```

Expected: builds and installs without error. Any dependency resolution failure means a version bound needs to be loosened.

- [ ] **Step 5: Rerun the package-layout test (should still fail — package skeleton not created yet)**

```bash
pytest tests/unit/test_package_layout.py -v
```

Expected: still fails with `ModuleNotFoundError`. The skeleton is created in Task 3.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tests/unit/test_package_layout.py
git commit -m "$(cat <<'EOF'
build: rewrite pyproject.toml for V2 package layout

Bumps to 2.0.0a1, declares kaggle_slayer as a proper package, adds V2
runtime deps (google-genai, opentelemetry-api/sdk, jsonschema), adds
mypy to dev deps, and configures mypy strict on harness only.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Create the new package skeleton

**Files:**
- Create: `kaggle_slayer/__init__.py`
- Create: `kaggle_slayer/harness/__init__.py`
- Create: `kaggle_slayer/harness/registry/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`

- [ ] **Step 1: Create `kaggle_slayer/__init__.py`**

```python
"""KaggleSlayer V2 — LLM-agent harness for tabular Kaggle competitions."""

__version__ = "2.0.0a1"
```

- [ ] **Step 2: Create `kaggle_slayer/harness/__init__.py`**

```python
"""Trusted side of the harness: CV, registries, sandbox, telemetry.

Contains no LLM calls. Owns the parts of the pipeline whose correctness
must not be left to LLM judgment (leak-free CV, metric scoring, etc.).
"""
```

- [ ] **Step 3: Create `kaggle_slayer/harness/registry/__init__.py`**

```python
"""Registries for metrics and CV strategies."""
```

- [ ] **Step 4: Create `tests/__init__.py` and `tests/unit/__init__.py`**

Both are empty files. Create them with:

```bash
touch tests/__init__.py tests/unit/__init__.py
```

- [ ] **Step 5: Run the package-layout test, now expected to pass**

```bash
pytest tests/unit/test_package_layout.py -v
```

Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add kaggle_slayer/ tests/__init__.py tests/unit/__init__.py
git commit -m "$(cat <<'EOF'
feat: add new kaggle_slayer package skeleton

Empty __init__.py files for kaggle_slayer, kaggle_slayer.harness, and
kaggle_slayer.harness.registry. Confirms the new package layout is
importable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Extend CI to V2 branch and stack

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Read the current CI workflow**

```bash
cat .github/workflows/ci.yml
```

Note the current structure: lint-and-test job, e2e-smoke job.

- [ ] **Step 2: Replace `.github/workflows/ci.yml`**

Overwrite with:

```yaml
name: CI

on:
  push:
    branches: [main, v2-rebuild]
  pull_request:
    branches: [main, v2-rebuild]
  workflow_dispatch:

jobs:
  lint-test:
    name: Lint, type-check, unit + integration tests
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip

      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev,dashboard]"

      - name: Ruff
        run: ruff check kaggle_slayer tests

      - name: Mypy (harness only)
        run: mypy kaggle_slayer/harness

      - name: Pytest with coverage
        run: pytest --cov=kaggle_slayer/harness --cov-report=term --cov-report=xml -m "not slow"

      - name: Upload coverage
        if: matrix.python-version == '3.11'
        uses: actions/upload-artifact@v4
        with:
          name: coverage-xml
          path: coverage.xml
```

- [ ] **Step 3: Run the same commands locally to verify they pass**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness
pytest -m "not slow" -v
```

Expected: ruff passes (no harness code yet → no issues). mypy passes (no harness code yet). pytest passes the 3 package-layout tests.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "$(cat <<'EOF'
ci: extend workflow for v2-rebuild branch and V2 stack

Adds v2-rebuild to push/PR triggers. Adds mypy step (harness only).
Tightens ruff (no continue-on-error). Adds pytest-cov reporting with
target on kaggle_slayer/harness. Matrix on Python 3.11 + 3.12.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Synthetic data fixtures in `tests/conftest.py`

**Files:**
- Create: `tests/conftest.py`

Fixtures used across all Week 1 tests. Three small synthetic datasets — kept small so the test suite runs in seconds.

- [ ] **Step 1: Write a smoke test that asserts the fixtures exist**

Create `tests/unit/test_fixtures.py`:

```python
"""Smoke tests that confirm conftest fixtures are wired correctly."""

import pandas as pd


def test_binary_classification_fixture_shape(synthetic_binary):
    train, target_col = synthetic_binary
    assert isinstance(train, pd.DataFrame)
    assert target_col in train.columns
    assert len(train) == 500
    assert set(train[target_col].unique()) == {0, 1}


def test_regression_fixture_shape(synthetic_regression):
    train, target_col = synthetic_regression
    assert isinstance(train, pd.DataFrame)
    assert target_col in train.columns
    assert len(train) == 500
    assert train[target_col].dtype.kind == "f"


def test_time_series_fixture_shape(synthetic_time_series):
    train, target_col, date_col = synthetic_time_series
    assert isinstance(train, pd.DataFrame)
    assert target_col in train.columns
    assert date_col in train.columns
    assert len(train) == 300
    assert pd.api.types.is_datetime64_any_dtype(train[date_col])
```

- [ ] **Step 2: Run to confirm it fails**

```bash
pytest tests/unit/test_fixtures.py -v
```

Expected: 3 errors, all "fixture 'synthetic_binary' (etc.) not found".

- [ ] **Step 3: Create `tests/conftest.py`**

```python
"""Shared synthetic data fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_binary() -> tuple[pd.DataFrame, str]:
    """500-row binary classification: 4 numerics + 2 categoricals + target.

    Target is a noisy function of x1 and x2 — non-trivial but learnable
    by any reasonable model.
    """
    rng = np.random.default_rng(seed=42)
    n = 500
    df = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "x3": rng.normal(size=n),
        "x4": rng.normal(size=n),
        "cat_a": rng.choice(["A", "B", "C"], size=n),
        "cat_b": rng.choice(["P", "Q"], size=n),
    })
    logits = 1.5 * df["x1"] - 0.8 * df["x2"] + rng.normal(scale=0.5, size=n)
    df["target"] = (logits > 0).astype(int)
    return df, "target"


@pytest.fixture
def synthetic_regression() -> tuple[pd.DataFrame, str]:
    """500-row regression: 4 numerics + 1 categorical + continuous target."""
    rng = np.random.default_rng(seed=43)
    n = 500
    df = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "x3": rng.uniform(size=n),
        "x4": rng.normal(size=n),
        "cat_a": rng.choice(["A", "B", "C", "D"], size=n),
    })
    df["target"] = (
        2.0 * df["x1"]
        + 0.5 * df["x2"] ** 2
        - 1.0 * df["x3"]
        + rng.normal(scale=0.3, size=n)
    )
    return df, "target"


@pytest.fixture
def synthetic_time_series() -> tuple[pd.DataFrame, str, str]:
    """300-row time-indexed regression with weekly seasonality."""
    rng = np.random.default_rng(seed=44)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    t = np.arange(n)
    seasonal = np.sin(2 * np.pi * t / 7)
    trend = 0.01 * t
    noise = rng.normal(scale=0.2, size=n)
    df = pd.DataFrame({
        "date": dates,
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "target": seasonal + trend + noise,
    })
    return df, "target", "date"
```

- [ ] **Step 4: Rerun and confirm the fixture tests pass**

```bash
pytest tests/unit/test_fixtures.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/unit/test_fixtures.py
git commit -m "$(cat <<'EOF'
test: add synthetic data fixtures for V2 harness tests

Three fixtures: synthetic_binary (500 rows, binary classification),
synthetic_regression (500 rows, continuous target), and
synthetic_time_series (300 rows, time-indexed with weekly seasonality).
Deterministic via fixed seeds.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Metric registry — Protocol + 6 metrics

**Files:**
- Create: `kaggle_slayer/harness/registry/metrics.py`
- Create: `tests/unit/test_metrics_registry.py`

Six metrics for Week 1: `accuracy`, `auc`, `logloss`, `rmse`, `mae`, `r2`. The remaining tabular Kaggle metrics (RMSLE, QWK, MAP@K, MCRMSE, Pinball) are added in Week 2 when an agent first needs them.

- [ ] **Step 1: Write failing tests for the Metric protocol and registry**

Create `tests/unit/test_metrics_registry.py`:

```python
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
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
pytest tests/unit/test_metrics_registry.py -v
```

Expected: collection error or `ModuleNotFoundError: kaggle_slayer.harness.registry.metrics`.

- [ ] **Step 3: Create `kaggle_slayer/harness/registry/metrics.py`**

```python
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

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import (
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
        score_fn=accuracy_score,
    ),
    "auc": Metric(
        name="auc",
        higher_is_better=True,
        needs_proba=True,
        score_fn=roc_auc_score,
    ),
    "logloss": Metric(
        name="logloss",
        higher_is_better=False,
        needs_proba=True,
        score_fn=log_loss,
    ),
    "rmse": Metric(
        name="rmse",
        higher_is_better=False,
        needs_proba=False,
        score_fn=_rmse,
    ),
    "mae": Metric(
        name="mae",
        higher_is_better=False,
        needs_proba=False,
        score_fn=mean_absolute_error,
    ),
    "r2": Metric(
        name="r2",
        higher_is_better=True,
        needs_proba=False,
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
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
pytest tests/unit/test_metrics_registry.py -v
```

Expected: 13 tests pass.

- [ ] **Step 5: Run ruff and mypy to confirm no lint or type issues**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness
```

Expected: both pass.

- [ ] **Step 6: Commit**

```bash
git add kaggle_slayer/harness/registry/metrics.py tests/unit/test_metrics_registry.py
git commit -m "$(cat <<'EOF'
feat: add metric registry with 6 Week-1 metrics

Registers accuracy, auc, logloss, rmse, mae, r2. Each Metric carries
higher_is_better and needs_proba metadata so the harness knows how to
call .predict vs .predict_proba and how to compare scores.

Remaining tabular Kaggle metrics (RMSLE, QWK, MAP@K, MCRMSE, Pinball)
land in Week 2 when an agent first needs them.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: CV strategy registry — Protocol + StratifiedKFold + KFold

**Files:**
- Create: `kaggle_slayer/harness/registry/cv_strategies.py`
- Create: `tests/unit/test_cv_strategies_registry.py`

Week 1 strategies: `stratified_kfold` and `kfold`. TimeSeriesSplit and GroupKFold land in Week 2 when context-driven CV selection is built.

- [ ] **Step 1: Write failing tests for the CVStrategy protocol and registry**

Create `tests/unit/test_cv_strategies_registry.py`:

```python
"""Tests for kaggle_slayer.harness.registry.cv_strategies."""

from __future__ import annotations

import pandas as pd
import pytest

from kaggle_slayer.harness.registry import cv_strategies


def test_get_known_strategy_returns_instance():
    cv = cv_strategies.get("kfold", n_splits=3)
    assert cv.name == "kfold"
    assert cv.n_splits == 3


def test_get_stratified_kfold():
    cv = cv_strategies.get("stratified_kfold", n_splits=5)
    assert cv.name == "stratified_kfold"
    assert cv.n_splits == 5


def test_get_unknown_strategy_raises():
    with pytest.raises(KeyError, match="not_a_strategy"):
        cv_strategies.get("not_a_strategy")


def test_list_strategies_includes_week1_set():
    names = set(cv_strategies.list_strategies())
    assert {"kfold", "stratified_kfold"} <= names


def test_kfold_split_returns_expected_fold_count(synthetic_regression):
    train, target_col = synthetic_regression
    cv = cv_strategies.get("kfold", n_splits=5, random_state=42)
    folds = list(cv.split(train, target_col))
    assert len(folds) == 5
    for train_idx, val_idx in folds:
        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(train_idx) + len(val_idx) == len(train)


def test_stratified_kfold_preserves_class_balance(synthetic_binary):
    train, target_col = synthetic_binary
    cv = cv_strategies.get("stratified_kfold", n_splits=5, random_state=42)
    overall_rate = train[target_col].mean()
    for _, val_idx in cv.split(train, target_col):
        fold_rate = train.iloc[val_idx][target_col].mean()
        # stratification keeps the class rate close to overall (~within 5pp)
        assert abs(fold_rate - overall_rate) < 0.05


def test_auto_select_classification_returns_stratified(synthetic_binary):
    train, target_col = synthetic_binary
    cv = cv_strategies.auto_select(
        problem_type="classification", train_df=train, target_col=target_col
    )
    assert cv.name == "stratified_kfold"


def test_auto_select_regression_returns_kfold(synthetic_regression):
    train, target_col = synthetic_regression
    cv = cv_strategies.auto_select(
        problem_type="regression", train_df=train, target_col=target_col
    )
    assert cv.name == "kfold"
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
pytest tests/unit/test_cv_strategies_registry.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.harness.registry.cv_strategies`.

- [ ] **Step 3: Create `kaggle_slayer/harness/registry/cv_strategies.py`**

```python
"""Cross-validation strategy registry.

A CVStrategy wraps a sklearn-style splitter and carries a name + the
config the harness needs to log it (n_splits, random_state, etc.).
Week 1 strategies: kfold, stratified_kfold.

Time-indexed and grouped strategies land in Week 2 when context-driven
CV selection is built (the agent and the data shape decide which).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


@dataclass
class CVStrategy:
    """Wraps a sklearn splitter for the harness CV contract."""

    name: str
    n_splits: int
    random_state: int | None = 42
    extra: dict = field(default_factory=dict)
    _splitter: object = None

    def split(
        self, df: pd.DataFrame, target_col: str
    ) -> Iterator[tuple[list[int], list[int]]]:
        """Yield (train_idx, val_idx) tuples for each fold."""
        if self._splitter is None:
            raise RuntimeError("CVStrategy._splitter not initialized")
        y = df[target_col]
        # sklearn splitters all take (X, y); we pass df so the splitter can
        # access any extra columns it might need (e.g., groups in future).
        for train_idx, val_idx in self._splitter.split(df, y):
            yield list(train_idx), list(val_idx)


def _make_kfold(n_splits: int, random_state: int | None = 42, **_: object) -> CVStrategy:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return CVStrategy(
        name="kfold",
        n_splits=n_splits,
        random_state=random_state,
        _splitter=splitter,
    )


def _make_stratified_kfold(
    n_splits: int, random_state: int | None = 42, **_: object
) -> CVStrategy:
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    return CVStrategy(
        name="stratified_kfold",
        n_splits=n_splits,
        random_state=random_state,
        _splitter=splitter,
    )


_FACTORIES = {
    "kfold": _make_kfold,
    "stratified_kfold": _make_stratified_kfold,
}


def get(name: str, *, n_splits: int = 5, random_state: int | None = 42, **kwargs: object) -> CVStrategy:
    """Construct a CVStrategy by name."""
    if name not in _FACTORIES:
        raise KeyError(f"cv strategy '{name}' not in registry; known: {sorted(_FACTORIES)}")
    return _FACTORIES[name](n_splits=n_splits, random_state=random_state, **kwargs)


def list_strategies() -> list[str]:
    """Return the names of all registered strategies."""
    return sorted(_FACTORIES)


def auto_select(
    *, problem_type: str, train_df: pd.DataFrame, target_col: str, n_splits: int = 5
) -> CVStrategy:
    """Pick a sensible default CV strategy from the problem type.

    Week-1 logic is intentionally minimal: classification → stratified,
    regression → plain. Time-indexed and grouped detection is added in
    Week 2 along with the strategies themselves.
    """
    if problem_type == "classification":
        return get("stratified_kfold", n_splits=n_splits)
    if problem_type == "regression":
        return get("kfold", n_splits=n_splits)
    raise ValueError(
        f"unsupported problem_type '{problem_type}'; expected 'classification' or 'regression'"
    )
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
pytest tests/unit/test_cv_strategies_registry.py -v
```

Expected: 8 tests pass.

- [ ] **Step 5: Lint and type-check**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness
```

Expected: both pass.

- [ ] **Step 6: Commit**

```bash
git add kaggle_slayer/harness/registry/cv_strategies.py tests/unit/test_cv_strategies_registry.py
git commit -m "$(cat <<'EOF'
feat: add CV strategy registry with kfold and stratified_kfold

Wraps sklearn KFold and StratifiedKFold under a CVStrategy dataclass.
auto_select() picks stratified for classification and plain kfold for
regression. TimeSeriesSplit and GroupKFold land in Week 2 alongside
context-driven selection.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `train_cv` leak-free contract + CVResult dataclass

**Files:**
- Create: `kaggle_slayer/harness/cv.py`
- Create: `tests/unit/test_cv_contract.py`

This is the correctness heart of the harness. The function loads the agent's `fe.py` and `model.py` modules from disk, iterates folds via the chosen CV strategy, and on each fold calls `fit_feature_transformer(train_fold, target_col)` with **only the train-fold data**. Anti-cheat: rows out must equal rows in.

- [ ] **Step 1: Write failing unit tests for `train_cv`**

Create `tests/unit/test_cv_contract.py`:

```python
"""Unit tests for kaggle_slayer.harness.cv.train_cv.

These tests use minimal hand-written fe/model modules created via tmp_path
to validate the contract behaviour. The richer end-to-end test using the
proper stub modules lives in tests/integration/test_cv_with_stubs.py.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from kaggle_slayer.harness import cv as cv_mod
from kaggle_slayer.harness.registry import cv_strategies, metrics


@pytest.fixture
def fe_pass_through(tmp_path):
    """Trivial FE: drops categoricals, passes numerics through."""
    p = tmp_path / "fe.py"
    p.write_text(textwrap.dedent("""
        import pandas as pd

        class _PassThrough:
            def __init__(self, numeric_cols):
                self.numeric_cols = numeric_cols
            def transform(self, df):
                return df[self.numeric_cols].copy()

        def fit_feature_transformer(train_df, target_col):
            numeric = [c for c in train_df.columns
                       if c != target_col and train_df[c].dtype.kind in "fiub"]
            return _PassThrough(numeric)
    """))
    return p


@pytest.fixture
def model_logreg(tmp_path):
    """Trivial model: logistic regression for classification."""
    p = tmp_path / "model.py"
    p.write_text(textwrap.dedent("""
        from sklearn.linear_model import LogisticRegression
        from sklearn.linear_model import Ridge

        def fit_model(X_train, y_train, problem_type, metric_name):
            if problem_type == "classification":
                m = LogisticRegression(max_iter=500)
            else:
                m = Ridge()
            m.fit(X_train, y_train)
            return m
    """))
    return p


def test_train_cv_runs_and_returns_cvresult(
    fe_pass_through, model_logreg, synthetic_binary
):
    train, target_col = synthetic_binary
    cv = cv_strategies.get("stratified_kfold", n_splits=3, random_state=42)
    metric = metrics.get("accuracy")

    result = cv_mod.train_cv(
        fe_path=fe_pass_through,
        model_path=model_logreg,
        train_df=train,
        target_col=target_col,
        cv=cv,
        metric=metric,
    )

    assert isinstance(result, cv_mod.CVResult)
    assert len(result.fold_scores) == 3
    assert 0.0 <= result.mean <= 1.0
    # Synthetic binary should be well above chance
    assert result.mean > 0.7
    assert result.oof.shape[0] == len(train)


def test_train_cv_with_proba_metric(fe_pass_through, model_logreg, synthetic_binary):
    train, target_col = synthetic_binary
    cv = cv_strategies.get("stratified_kfold", n_splits=3)
    metric = metrics.get("auc")  # needs_proba=True

    result = cv_mod.train_cv(
        fe_path=fe_pass_through,
        model_path=model_logreg,
        train_df=train,
        target_col=target_col,
        cv=cv,
        metric=metric,
    )

    assert 0.5 < result.mean <= 1.0
    # OOF should be probabilities, not class labels
    assert ((result.oof >= 0.0) & (result.oof <= 1.0)).all()


def test_train_cv_regression(fe_pass_through, model_logreg, synthetic_regression):
    train, target_col = synthetic_regression
    cv = cv_strategies.get("kfold", n_splits=3)
    metric = metrics.get("rmse")

    result = cv_mod.train_cv(
        fe_path=fe_pass_through,
        model_path=model_logreg,
        train_df=train,
        target_col=target_col,
        cv=cv,
        metric=metric,
    )

    assert result.mean > 0.0  # rmse is non-negative
    assert result.oof.shape[0] == len(train)


def test_train_cv_rejects_row_dropping_fe(tmp_path, model_logreg, synthetic_binary):
    """Anti-cheat: FE that drops rows must be rejected."""
    fe = tmp_path / "fe.py"
    fe.write_text(textwrap.dedent("""
        import pandas as pd

        class _Dropper:
            def __init__(self, numeric_cols):
                self.numeric_cols = numeric_cols
            def transform(self, df):
                # Drops half the rows — would change the val split
                return df[self.numeric_cols].iloc[::2].copy()

        def fit_feature_transformer(train_df, target_col):
            numeric = [c for c in train_df.columns
                       if c != target_col and train_df[c].dtype.kind in "fiub"]
            return _Dropper(numeric)
    """))
    train, target_col = synthetic_binary
    cv = cv_strategies.get("stratified_kfold", n_splits=3)
    metric = metrics.get("accuracy")

    with pytest.raises(cv_mod.CVError, match="rows changed"):
        cv_mod.train_cv(
            fe_path=fe,
            model_path=model_logreg,
            train_df=train,
            target_col=target_col,
            cv=cv,
            metric=metric,
        )


def test_train_cv_rejects_fe_missing_fit_function(tmp_path, model_logreg, synthetic_binary):
    """If fe.py is missing fit_feature_transformer, raise clearly."""
    fe = tmp_path / "fe.py"
    fe.write_text("# empty\n")
    train, target_col = synthetic_binary
    cv = cv_strategies.get("stratified_kfold", n_splits=3)
    metric = metrics.get("accuracy")

    with pytest.raises(cv_mod.CVError, match="fit_feature_transformer"):
        cv_mod.train_cv(
            fe_path=fe,
            model_path=model_logreg,
            train_df=train,
            target_col=target_col,
            cv=cv,
            metric=metric,
        )


def test_train_cv_rejects_model_missing_fit_function(tmp_path, fe_pass_through, synthetic_binary):
    """If model.py is missing fit_model, raise clearly."""
    model = tmp_path / "model.py"
    model.write_text("# empty\n")
    train, target_col = synthetic_binary
    cv = cv_strategies.get("stratified_kfold", n_splits=3)
    metric = metrics.get("accuracy")

    with pytest.raises(cv_mod.CVError, match="fit_model"):
        cv_mod.train_cv(
            fe_path=fe_pass_through,
            model_path=model,
            train_df=train,
            target_col=target_col,
            cv=cv,
            metric=metric,
        )


def test_train_cv_problem_type_inference_from_metric(
    fe_pass_through, model_logreg, synthetic_binary
):
    """Probability-needing metrics imply classification; train_cv passes the
    right `problem_type` to the agent's fit_model."""
    train, target_col = synthetic_binary
    cv = cv_strategies.get("stratified_kfold", n_splits=3)
    metric = metrics.get("auc")
    result = cv_mod.train_cv(
        fe_path=fe_pass_through,
        model_path=model_logreg,
        train_df=train,
        target_col=target_col,
        cv=cv,
        metric=metric,
    )
    # If problem_type was misinferred as regression, Ridge would not have
    # predict_proba and the call would have raised.
    assert result.mean > 0.5
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
pytest tests/unit/test_cv_contract.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.harness.cv`.

- [ ] **Step 3: Create `kaggle_slayer/harness/cv.py`**

```python
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
import pandas as pd

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


def _infer_problem_type(metric: Metric) -> str:
    """Classification iff metric needs probabilities (heuristic; agent override later)."""
    return "classification" if metric.needs_proba else "regression"


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

    problem_type = _infer_problem_type(metric)
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
            # Binary case: take positive-class probability. Multi-class
            # probability outputs are deferred to a later week.
            if proba.ndim == 2 and proba.shape[1] == 2:
                preds = proba[:, 1]
            else:
                preds = proba
        else:
            preds = model.predict(X_val)

        preds_arr = np.asarray(preds, dtype=float)
        if preds_arr.ndim != 1:
            raise CVError(
                f"multi-dimensional predictions not yet supported "
                f"(got shape {preds_arr.shape}); only 1-D predictions "
                f"(binary classification or regression) are valid in V1"
            )
        oof[val_idx] = preds_arr
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
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
pytest tests/unit/test_cv_contract.py -v
```

Expected: 7 tests pass. (`test_train_cv_with_proba_metric` triggers the binary-classification branch; others cover the contract guarantees.)

- [ ] **Step 5: Lint and type-check**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness
```

Expected: both pass.

- [ ] **Step 6: Commit**

```bash
git add kaggle_slayer/harness/cv.py tests/unit/test_cv_contract.py
git commit -m "$(cat <<'EOF'
feat: add train_cv leak-free contract

The harness loads the agent's fe.py and model.py from disk, runs the
configured CV strategy, and calls fit_feature_transformer with only the
fold's training data. Anti-cheat: row count must be preserved through
.transform(); metric->problem_type inferred; CVError raised on contract
violations (missing fit fns, row count mismatch, model missing
predict_proba when metric needs it).

Returns CVResult with per-fold scores, mean, std, OOF predictions,
timing, and metadata.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Stub `fe.py` / `model.py` fixtures + integration test

**Files:**
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/fe_stub.py`
- Create: `tests/fixtures/model_stub.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_cv_with_stubs.py`

The stub modules are the canonical reference for what the agent will eventually write. They're also used by the Week 1 acceptance test.

- [ ] **Step 1: Create the fixture package**

```bash
touch tests/fixtures/__init__.py tests/integration/__init__.py
```

- [ ] **Step 2: Create `tests/fixtures/fe_stub.py`**

```python
"""Minimal valid feature-engineering module.

Demonstrates the agent's contract: fit_feature_transformer(train_df, target_col)
returns an object with .transform(df) -> df. Statistics (means for imputation)
are fit on the train fold ONLY — see spec §6.
"""

from __future__ import annotations

import pandas as pd


class _StubTransformer:
    def __init__(self, numeric_cols: list[str], cat_cols: list[str],
                 numeric_means: dict[str, float], cat_modes: dict[str, str]):
        self.numeric_cols = numeric_cols
        self.cat_cols = cat_cols
        self.numeric_means = numeric_means
        self.cat_modes = cat_modes

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.numeric_cols:
            if col in df.columns:
                out[col] = df[col].fillna(self.numeric_means.get(col, 0.0))
            else:
                out[col] = self.numeric_means.get(col, 0.0)
        for col in self.cat_cols:
            if col in df.columns:
                series = df[col].fillna(self.cat_modes.get(col, "MISSING"))
                # One-hot encode against the training categories.
                dummies = pd.get_dummies(series, prefix=col, drop_first=False)
                out = pd.concat([out, dummies], axis=1)
        return out


def fit_feature_transformer(train_df: pd.DataFrame, target_col: str) -> _StubTransformer:
    """Fit on train fold only.

    Imputes numerics with train-fold mean; one-hot encodes categoricals using
    train-fold categories. Returns an object with .transform(df) for val/test.
    """
    feature_df = train_df.drop(columns=[target_col])
    numeric_cols = [c for c in feature_df.columns if feature_df[c].dtype.kind in "fiub"]
    cat_cols = [c for c in feature_df.columns if c not in numeric_cols]
    numeric_means = {c: float(feature_df[c].mean()) for c in numeric_cols}
    cat_modes = {
        c: str(feature_df[c].mode(dropna=True).iloc[0])
        if not feature_df[c].mode(dropna=True).empty
        else "MISSING"
        for c in cat_cols
    }
    return _StubTransformer(numeric_cols, cat_cols, numeric_means, cat_modes)
```

- [ ] **Step 3: Create `tests/fixtures/model_stub.py`**

```python
"""Minimal valid model module.

Demonstrates the agent's contract: fit_model(X, y, problem_type, metric_name)
returns a model with .predict(X) (and .predict_proba for proba-metric paths).
Picks a small sklearn classifier/regressor — the goal is a sanity model, not
a winning one.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge


def fit_model(X_train, y_train: np.ndarray, problem_type: str, metric_name: str):
    """Return a fitted classifier or regressor."""
    if problem_type == "classification":
        model = LogisticRegression(max_iter=500, random_state=42)
    elif problem_type == "regression":
        model = Ridge(alpha=1.0, random_state=42)
    else:
        raise ValueError(f"unsupported problem_type '{problem_type}'")
    model.fit(X_train, y_train)
    return model
```

- [ ] **Step 4: Write the failing integration test**

Create `tests/integration/test_cv_with_stubs.py`:

```python
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
```

- [ ] **Step 5: Run the integration test**

```bash
pytest tests/integration/ -v -m integration
```

Expected: 4 tests pass. Together they prove the Week 1 acceptance: leak-free CV runs end-to-end on a stub fe.py / model.py pair.

- [ ] **Step 6: Run the full unit + integration suite**

```bash
pytest -m "not slow" --cov=kaggle_slayer/harness --cov-report=term-missing -v
```

Expected: all tests pass; coverage ≥ 85% on `kaggle_slayer/harness` (the cv.py, metrics.py, cv_strategies.py files are now exercised by both unit and integration tests).

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures tests/integration
git commit -m "$(cat <<'EOF'
test: add stub fe/model modules and Week-1 acceptance test

tests/fixtures/fe_stub.py and model_stub.py are the canonical references
for what the agent will eventually write: train-fold-only fits, transform()
preserves row count, model returns predict/predict_proba per problem type.

tests/integration/test_cv_with_stubs.py is the Week-1 acceptance test:
train_cv runs end-to-end on the stubs over synthetic binary and
regression data, beating the chance/mean baseline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Sandbox AST lint

**Files:**
- Create: `kaggle_slayer/harness/sandbox.py`
- Create: `tests/unit/test_sandbox_lint.py`

The AST lint scans a `.py` file before the harness will load it. It rejects: destructive filesystem calls, network calls, eval/exec/compile, subprocess invocations, and any literal read of `raw/...` paths. Resource limits are deferred to Week 4 (per spec §8).

- [ ] **Step 1: Write failing tests for the AST lint**

Create `tests/unit/test_sandbox_lint.py`:

```python
"""Tests for kaggle_slayer.harness.sandbox.lint_module."""

from __future__ import annotations

import textwrap

import pytest

from kaggle_slayer.harness import sandbox


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


def test_lint_passes_minimal_valid_fe(tmp_path):
    p = _write(tmp_path, "fe.py", """
        import pandas as pd

        class T:
            def transform(self, df):
                return df

        def fit_feature_transformer(train_df, target_col):
            return T()
    """)
    result = sandbox.lint_module(p)
    assert result.ok, result.violations


def test_lint_passes_stub_fe(tmp_path):
    """The shipped fe_stub.py must lint clean."""
    from pathlib import Path
    stub = Path(__file__).resolve().parents[1] / "fixtures" / "fe_stub.py"
    result = sandbox.lint_module(stub)
    assert result.ok, result.violations


def test_lint_rejects_os_remove(tmp_path):
    p = _write(tmp_path, "bad.py", """
        import os
        def fit_feature_transformer(train_df, target_col):
            os.remove("/tmp/x")
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("os.remove" in v for v in result.violations)


def test_lint_rejects_shutil_rmtree(tmp_path):
    p = _write(tmp_path, "bad.py", """
        import shutil
        def fit_feature_transformer(train_df, target_col):
            shutil.rmtree("/tmp")
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("shutil.rmtree" in v for v in result.violations)


def test_lint_rejects_subprocess(tmp_path):
    p = _write(tmp_path, "bad.py", """
        import subprocess
        def fit_feature_transformer(train_df, target_col):
            subprocess.run(["rm", "-rf", "/tmp"])
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("subprocess" in v for v in result.violations)


def test_lint_rejects_os_system(tmp_path):
    p = _write(tmp_path, "bad.py", """
        import os
        def fit_feature_transformer(train_df, target_col):
            os.system("echo bad")
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("os.system" in v for v in result.violations)


def test_lint_rejects_eval(tmp_path):
    p = _write(tmp_path, "bad.py", """
        def fit_feature_transformer(train_df, target_col):
            return eval("None")
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("eval" in v for v in result.violations)


def test_lint_rejects_exec(tmp_path):
    p = _write(tmp_path, "bad.py", """
        def fit_feature_transformer(train_df, target_col):
            exec("pass")
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("exec" in v for v in result.violations)


def test_lint_rejects_requests(tmp_path):
    p = _write(tmp_path, "bad.py", """
        import requests
        def fit_feature_transformer(train_df, target_col):
            requests.get("https://evil.com")
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("requests" in v for v in result.violations)


def test_lint_rejects_urllib(tmp_path):
    p = _write(tmp_path, "bad.py", """
        import urllib.request
        def fit_feature_transformer(train_df, target_col):
            urllib.request.urlopen("https://evil.com")
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("urllib" in v for v in result.violations)


def test_lint_rejects_read_of_raw_path(tmp_path):
    """The agent must not read competition raw data directly — only what
    the harness passes to fit_feature_transformer."""
    p = _write(tmp_path, "bad.py", """
        import pandas as pd
        def fit_feature_transformer(train_df, target_col):
            extra = pd.read_csv("raw/train.csv")
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("raw/" in v for v in result.violations)


def test_lint_rejects_open_write_outside_workspace(tmp_path):
    p = _write(tmp_path, "bad.py", """
        def fit_feature_transformer(train_df, target_col):
            open("/etc/passwd", "w")
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert any("open" in v for v in result.violations)


def test_lint_aggregates_multiple_violations(tmp_path):
    """All violations are reported, not just the first."""
    p = _write(tmp_path, "bad.py", """
        import os
        import subprocess
        def fit_feature_transformer(train_df, target_col):
            os.remove("/tmp/x")
            subprocess.run(["true"])
            return None
    """)
    result = sandbox.lint_module(p)
    assert not result.ok
    assert len(result.violations) >= 2
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
pytest tests/unit/test_sandbox_lint.py -v
```

Expected: `ModuleNotFoundError: kaggle_slayer.harness.sandbox`.

- [ ] **Step 3: Create `kaggle_slayer/harness/sandbox.py`**

```python
"""Sandbox for agent-written Python modules.

Week 1 scope: AST lint that scans a Python file before the harness loads
it and rejects forbidden patterns. The lint is the leak-prevention
mechanism for the in-process CV contract (see spec §6.5): if an agent
tries to read `raw/...` directly, the file fails lint and never executes.

Resource limits (subprocess + setrlimit) are added in Week 4 alongside the
broader sandbox hardening when the agent gets a generic `run_python` tool.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


# Module-prefix denylist: any `X.Y.Z(...)` whose dotted attribute chain
# starts with one of these tuples is rejected. We match against the
# *attribute chain* (e.g., `os.remove`), not the imported alias —
# `import os as o; o.remove(...)` is still flagged because we track aliases.
_FORBIDDEN_ATTR_CALLS: tuple[tuple[str, ...], ...] = (
    ("os", "remove"),
    ("os", "unlink"),
    ("os", "removedirs"),
    ("os", "system"),
    ("os", "popen"),
    ("shutil", "rmtree"),
    ("shutil", "move"),
    ("subprocess",),  # any subprocess.* call
    ("requests",),
    ("urllib",),
    ("socket",),
    ("http", "client"),
)

_FORBIDDEN_BUILTINS: frozenset[str] = frozenset({"eval", "exec", "compile", "__import__"})

# Path literals starting with these prefixes are forbidden as arguments to
# file-reading calls — agent code must not directly touch competition data.
_FORBIDDEN_PATH_PREFIXES: tuple[str, ...] = ("raw/", "raw\\", "/raw/", "./raw/")

# Calls whose first string-literal argument we check for forbidden path prefixes.
_PATH_OPEN_CALLS: frozenset[tuple[str, ...]] = frozenset({
    ("pd", "read_csv"),
    ("pd", "read_parquet"),
    ("pd", "read_feather"),
    ("pd", "read_json"),
    ("pd", "read_excel"),
    ("pandas", "read_csv"),
    ("pandas", "read_parquet"),
    ("pandas", "read_feather"),
    ("pandas", "read_json"),
    ("pandas", "read_excel"),
    ("np", "loadtxt"),
    ("np", "load"),
    ("numpy", "loadtxt"),
    ("numpy", "load"),
})

# Absolute filesystem paths the agent must never try to open (any mode).
_FORBIDDEN_ABS_PATHS: tuple[str, ...] = (
    "/etc/", "/var/", "/usr/", "/private/", "/Users/", "/root/", "/home/",
)


@dataclass(frozen=True)
class LintResult:
    ok: bool
    violations: list[str] = field(default_factory=list)


def lint_module(path: str | Path) -> LintResult:
    """Scan a Python file's AST for forbidden patterns.

    Returns LintResult.ok=True if no violations; otherwise a list of
    human-readable violation messages keyed by line number.
    """
    path = Path(path)
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))

    aliases = _collect_import_aliases(tree)
    visitor = _ForbidVisitor(aliases=aliases, filename=str(path))
    visitor.visit(tree)

    return LintResult(ok=not visitor.violations, violations=visitor.violations)


def _collect_import_aliases(tree: ast.AST) -> dict[str, tuple[str, ...]]:
    """Map local names to their underlying dotted module path.

    `import os` → {"os": ("os",)}
    `import os as o` → {"o": ("os",)}
    `from os import path` → {"path": ("os", "path")}
    `from os import path as p` → {"p": ("os", "path")}
    """
    aliases: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                local = n.asname or n.name.split(".")[0]
                aliases[local] = tuple(n.name.split("."))
        elif isinstance(node, ast.ImportFrom) and node.module:
            base = tuple(node.module.split("."))
            for n in node.names:
                local = n.asname or n.name
                aliases[local] = base + (n.name,)
    return aliases


class _ForbidVisitor(ast.NodeVisitor):
    def __init__(self, *, aliases: dict[str, tuple[str, ...]], filename: str) -> None:
        self.aliases = aliases
        self.filename = filename
        self.violations: list[str] = []

    def _attr_chain(self, node: ast.AST) -> tuple[str, ...]:
        """Resolve an AST attribute/name chain to a dotted tuple,
        de-aliased against imports."""
        parts: list[str] = []
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        else:
            return ()
        chain = tuple(reversed(parts))
        if chain and chain[0] in self.aliases:
            return self.aliases[chain[0]] + chain[1:]
        return chain

    def _violate(self, msg: str, node: ast.AST) -> None:
        self.violations.append(f"{self.filename}:{node.lineno}: {msg}")

    # --- Calls ---
    def visit_Call(self, node: ast.Call) -> None:
        chain = self._attr_chain(node.func)

        # Forbidden builtins (eval, exec, compile, __import__)
        if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_BUILTINS:
            self._violate(f"forbidden builtin call: {node.func.id}", node)
        elif chain and chain[-1] == "open":
            # builtin open: check path arg for absolute denylist
            self._check_open_path(node)
        else:
            # Module-prefix denylist
            for forbidden in _FORBIDDEN_ATTR_CALLS:
                if chain[: len(forbidden)] == forbidden:
                    self._violate(
                        f"forbidden call: {'.'.join(chain)}", node
                    )
                    break

            # pd.read_csv / np.load et al. against raw/ paths
            if chain in _PATH_OPEN_CALLS:
                self._check_raw_path_arg(chain, node)

        # plain `open(...)`
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            self._check_open_path(node)

        self.generic_visit(node)

    def _check_raw_path_arg(self, chain: tuple[str, ...], node: ast.Call) -> None:
        if not node.args:
            return
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            if any(arg.value.startswith(p) for p in _FORBIDDEN_PATH_PREFIXES):
                self._violate(
                    f"forbidden {'.'.join(chain)} read of competition raw/ path: {arg.value!r}",
                    node,
                )

    def _check_open_path(self, node: ast.Call) -> None:
        if not node.args:
            return
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            value = arg.value
            if any(value.startswith(p) for p in _FORBIDDEN_PATH_PREFIXES):
                self._violate(
                    f"forbidden open of competition raw/ path: {value!r}", node
                )
            if any(value.startswith(p) for p in _FORBIDDEN_ABS_PATHS):
                self._violate(
                    f"forbidden open of absolute system path: {value!r}", node
                )
```

- [ ] **Step 4: Run the lint tests to confirm they pass**

```bash
pytest tests/unit/test_sandbox_lint.py -v
```

Expected: 13 tests pass.

- [ ] **Step 5: Lint and type-check**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness
```

Expected: both pass.

- [ ] **Step 6: Commit**

```bash
git add kaggle_slayer/harness/sandbox.py tests/unit/test_sandbox_lint.py
git commit -m "$(cat <<'EOF'
feat: add AST sandbox lint for agent-written modules

lint_module(path) walks the AST and flags:
  - destructive fs: os.remove, os.unlink, shutil.rmtree, shutil.move
  - shell-out: os.system, os.popen, subprocess.*
  - network: requests, urllib, socket, http.client
  - dangerous builtins: eval, exec, compile, __import__
  - direct reads of competition raw/ paths via pd.read_csv et al.
  - opens of forbidden absolute system paths

Tracks import aliases so 'import os as o; o.remove(...)' is still caught.
Reports line numbers and aggregates all violations (not first-fail).

Resource limits (subprocess + setrlimit) defer to Week 4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Week 1 acceptance test + sanity check

**Files:**
- Create: `tests/integration/test_week1_acceptance.py`

Bundles the Week 1 acceptance criteria into one explicit test so future regressions are loud. Runs the lint over the stub modules, then exercises `train_cv` over each synthetic dataset.

- [ ] **Step 1: Write the acceptance test**

Create `tests/integration/test_week1_acceptance.py`:

```python
"""Week 1 acceptance: lint passes on stubs, leak-free CV runs end-to-end.

If this passes, Week 1 is done. Any future change that breaks this test
indicates the foundations have regressed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kaggle_slayer.harness import cv as cv_mod, sandbox
from kaggle_slayer.harness.registry import cv_strategies, metrics

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FE_STUB = FIXTURES / "fe_stub.py"
MODEL_STUB = FIXTURES / "model_stub.py"


pytestmark = pytest.mark.integration


def test_stubs_pass_ast_lint():
    fe_result = sandbox.lint_module(FE_STUB)
    model_result = sandbox.lint_module(MODEL_STUB)
    assert fe_result.ok, fe_result.violations
    assert model_result.ok, model_result.violations


def test_leak_free_cv_e2e_binary_classification(synthetic_binary):
    train, target_col = synthetic_binary
    cv = cv_strategies.auto_select(
        problem_type="classification", train_df=train, target_col=target_col
    )
    result = cv_mod.train_cv(
        fe_path=FE_STUB,
        model_path=MODEL_STUB,
        train_df=train,
        target_col=target_col,
        cv=cv,
        metric=metrics.get("accuracy"),
    )
    assert result.mean > 0.70
    assert result.metadata["cv_strategy"] == "stratified_kfold"


def test_leak_free_cv_e2e_regression(synthetic_regression):
    train, target_col = synthetic_regression
    cv = cv_strategies.auto_select(
        problem_type="regression", train_df=train, target_col=target_col
    )
    result = cv_mod.train_cv(
        fe_path=FE_STUB,
        model_path=MODEL_STUB,
        train_df=train,
        target_col=target_col,
        cv=cv,
        metric=metrics.get("rmse"),
    )
    assert result.metadata["cv_strategy"] == "kfold"
    target_std = train[target_col].std()
    assert result.mean < 0.7 * target_std


def test_leak_free_cv_records_metadata():
    """Result.metadata must surface enough for downstream MLflow logging."""
    # Build a minimal in-memory dataset
    import pandas as pd
    import numpy as np

    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.normal(size=200),
        "target": rng.integers(0, 2, size=200),
    })
    cv = cv_strategies.get("stratified_kfold", n_splits=3)
    result = cv_mod.train_cv(
        fe_path=FE_STUB,
        model_path=MODEL_STUB,
        train_df=df,
        target_col="target",
        cv=cv,
        metric=metrics.get("accuracy"),
    )
    assert "cv_strategy" in result.metadata
    assert "n_splits" in result.metadata
    assert "metric" in result.metadata
    assert "problem_type" in result.metadata
    assert result.duration_s > 0
```

- [ ] **Step 2: Run the acceptance test**

```bash
pytest tests/integration/test_week1_acceptance.py -v -m integration
```

Expected: 4 tests pass. **This is the Week 1 acceptance gate.**

- [ ] **Step 3: Run the entire suite + coverage**

```bash
pytest -m "not slow" --cov=kaggle_slayer/harness --cov-report=term-missing -v
```

Expected:
- All tests pass (count roughly: 3 layout + 3 fixtures + 13 metrics + 8 cv_strategies + 7 cv_contract + 13 sandbox_lint + 4 integration + 4 acceptance ≈ 55 tests).
- Coverage on `kaggle_slayer/harness`: ≥ 90% (small surface area; the tests hit most branches).

- [ ] **Step 4: Run ruff and mypy across the whole new package**

```bash
ruff check kaggle_slayer tests
mypy kaggle_slayer/harness
```

Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_week1_acceptance.py
git commit -m "$(cat <<'EOF'
test: add Week 1 acceptance suite

Bundles the Week 1 acceptance criteria into explicit tests:
  - shipped stub modules pass the AST sandbox lint
  - leak-free CV runs end-to-end on synthetic binary and regression
  - CVResult.metadata exposes the fields downstream MLflow logging needs

Future regressions in any of the Week 1 foundations will fail this test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Push the branch**

```bash
git push -u origin v2-rebuild
```

Expected: branch pushed. CI runs on it and goes green on all jobs (ruff, mypy, pytest with coverage on Python 3.11 + 3.12).

---

## Summary of Week 1 acceptance

When all tasks are complete, the following must hold:

- ✅ Branch `v2-rebuild` exists with `legacy/` containing all V1 source.
- ✅ `pyproject.toml` declares the new `kaggle_slayer` package; `pip install -e ".[dev]"` succeeds.
- ✅ Package skeleton importable: `kaggle_slayer`, `.harness`, `.harness.registry`.
- ✅ CI workflow runs ruff, mypy (harness only), pytest with coverage on `v2-rebuild`.
- ✅ `kaggle_slayer.harness.registry.metrics` ships 6 metrics: accuracy, auc, logloss, rmse, mae, r2.
- ✅ `kaggle_slayer.harness.registry.cv_strategies` ships 2 strategies: kfold, stratified_kfold, and an `auto_select` for problem_type → strategy.
- ✅ `kaggle_slayer.harness.cv.train_cv` runs leak-free CV on agent-written `fe.py` + `model.py` and returns a `CVResult` with fold scores, OOF, timing, metadata.
- ✅ `kaggle_slayer.harness.sandbox.lint_module` rejects destructive fs, shell-out, network, eval/exec/compile, direct raw/ reads, absolute-system-path opens — with line numbers and aggregated violations.
- ✅ Stub `fe_stub.py` + `model_stub.py` pass the lint and drive `train_cv` end-to-end on synthetic binary and regression data.
- ✅ Coverage on `kaggle_slayer/harness` ≥ 90%.
- ✅ All ruff and mypy checks green.

Week 2 starts with: workspace journalling + extended Kaggle API client + context-builder + `LLMClient` Gemini wrapper, followed by the integration tier with a **fake agent** (canned tool-call sequences).
