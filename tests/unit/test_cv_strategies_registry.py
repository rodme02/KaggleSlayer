"""Tests for kaggle_slayer.harness.registry.cv_strategies."""

from __future__ import annotations

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


def test_kfold_forwards_shuffle_kwarg():
    cv = cv_strategies.get("kfold", n_splits=3, shuffle=False)
    # sklearn KFold(shuffle=False) does NOT shuffle indices, so the first
    # fold's val set should be the first ⅓ of the data in order.
    import pandas as pd
    df = pd.DataFrame({"x": range(30), "target": [0]*15 + [1]*15})
    folds = list(cv.split(df, "target"))
    val_idx = folds[0][1]
    # Without shuffle, val indices are contiguous [0..9]
    assert val_idx == list(range(10))


def test_get_time_series_split():
    cv = cv_strategies.get("time_series", n_splits=5)
    assert cv.name == "time_series"
    assert cv.n_splits == 5


def test_time_series_split_yields_forward_only_folds(synthetic_time_series):
    train, target_col, _ = synthetic_time_series
    cv = cv_strategies.get("time_series", n_splits=4)
    for train_idx, val_idx in cv.split(train, target_col):
        # Every train index must be strictly less than every val index
        assert max(train_idx) < min(val_idx)


def test_get_group_kfold():
    cv = cv_strategies.get("group_kfold", n_splits=3, group_col="cat_a")
    assert cv.name == "group_kfold"
    assert cv.n_splits == 3
    assert cv.extra.get("group_col") == "cat_a"


def test_group_kfold_respects_group_boundaries(synthetic_binary):
    train, target_col = synthetic_binary
    cv = cv_strategies.get("group_kfold", n_splits=3, group_col="cat_a")
    for train_idx, val_idx in cv.split(train, target_col):
        train_groups = set(train.iloc[train_idx]["cat_a"])
        val_groups = set(train.iloc[val_idx]["cat_a"])
        # No group appears in both train and val of the same fold
        assert train_groups.isdisjoint(val_groups)


def test_group_kfold_requires_group_col():
    with pytest.raises(ValueError, match="group_col"):
        cv_strategies.get("group_kfold", n_splits=3)  # no group_col


def test_auto_select_picks_time_series_when_date_col_given(synthetic_time_series):
    train, target_col, date_col = synthetic_time_series
    cv = cv_strategies.auto_select(
        problem_type="regression",
        train_df=train,
        target_col=target_col,
        date_col=date_col,
    )
    assert cv.name == "time_series"


def test_auto_select_picks_group_kfold_when_group_col_given(synthetic_binary):
    train, target_col = synthetic_binary
    cv = cv_strategies.auto_select(
        problem_type="classification",
        train_df=train,
        target_col=target_col,
        group_col="cat_a",
    )
    assert cv.name == "group_kfold"
    assert cv.extra.get("group_col") == "cat_a"
