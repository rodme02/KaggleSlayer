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
