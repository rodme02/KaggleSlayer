"""Shared pytest fixtures: synthetic Titanic-shaped data, on-disk competition layout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_classification(rng: np.random.Generator) -> pd.DataFrame:
    """Tiny Titanic-shaped binary classification frame."""
    n = 200
    age = rng.normal(35, 12, n).clip(1, 90)
    fare = rng.exponential(30, n)
    sex = rng.choice(["male", "female"], size=n)
    pclass = rng.choice([1, 2, 3], size=n, p=[0.2, 0.3, 0.5])
    # Survival depends on sex + class + age — gives a learnable signal
    logit = (sex == "female") * 1.5 - (pclass - 1) * 0.6 - (age > 60) * 0.4
    survived = (rng.uniform(0, 1, n) < (1 / (1 + np.exp(-logit)))).astype(int)
    return pd.DataFrame(
        {
            "PassengerId": np.arange(1, n + 1),
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "Fare": fare,
            "Survived": survived,
        }
    )


@pytest.fixture
def synthetic_regression(rng: np.random.Generator) -> pd.DataFrame:
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    cat = rng.choice(["a", "b", "c"], size=n)
    y = 2 * x1 - x2 + (cat == "a") * 0.5 + rng.normal(0, 0.1, n)
    return pd.DataFrame({"id": np.arange(n), "x1": x1, "x2": x2, "cat": cat, "y": y})


@pytest.fixture
def competition_dir(tmp_path, synthetic_classification):
    """Materialise a synthetic competition on disk in the layout the pipeline expects."""
    comp = tmp_path / "synthetic_titanic"
    raw = comp / "raw"
    raw.mkdir(parents=True)

    df = synthetic_classification
    train = df.iloc[:160].copy()
    test = df.iloc[160:].drop(columns=["Survived"]).copy()
    sample = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": 0})

    train.to_csv(raw / "train.csv", index=False)
    test.to_csv(raw / "test.csv", index=False)
    sample.to_csv(raw / "sample_submission.csv", index=False)
    return comp
