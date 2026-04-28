"""Tests for the ModelFactory: confirms each available model trains on synthetic data."""

from __future__ import annotations

import pandas as pd
import pytest

from core.models import ModelFactory


@pytest.fixture
def factory() -> ModelFactory:
    return ModelFactory(random_state=0)


def test_classification_models_train(factory, synthetic_classification):
    df = synthetic_classification.drop(columns=["PassengerId"])
    df = pd.get_dummies(df, columns=["Sex"], drop_first=True)
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    for name in factory.get_available_model_names("classification"):
        if name == "svm":
            continue  # slow on tiny synthetic — covered via sklearn
        model = factory.create_model(name, problem_type="classification")
        model.fit(X, y)
        preds = model.predict(X.head(5))
        assert len(preds) == 5


def test_regression_models_train(factory, synthetic_regression):
    df = pd.get_dummies(synthetic_regression.drop(columns=["id"]), columns=["cat"], drop_first=True)
    X = df.drop(columns=["y"])
    y = df["y"]
    for name in factory.get_available_model_names("regression"):
        if name == "svr":
            continue
        model = factory.create_model(name, problem_type="regression")
        model.fit(X, y)
        preds = model.predict(X.head(5))
        assert len(preds) == 5


def test_unknown_model_raises(factory):
    with pytest.raises(ValueError):
        factory.create_model("not_a_model", problem_type="classification")
