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
