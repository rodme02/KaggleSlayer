"""Tests for core/data: loaders, preprocessors, validators."""

from __future__ import annotations

import pandas as pd

from core.data import CompetitionDataLoader, DataPreprocessor, DataValidator


def test_competition_loader_finds_train_test(competition_dir):
    loader = CompetitionDataLoader(competition_dir)
    train, test = loader.load_competition_data()
    assert not train.empty
    assert test is not None and not test.empty
    assert "Survived" in train.columns and "Survived" not in test.columns


def test_target_detection(competition_dir):
    loader = CompetitionDataLoader(competition_dir)
    train, test = loader.load_competition_data()
    assert loader.detect_target_column(train, test) == "Survived"


def test_feature_type_detection(synthetic_classification):
    loader = CompetitionDataLoader.__new__(CompetitionDataLoader)
    loader.data_path = None  # not used
    loader.competition_name = "x"
    types = loader.analyze_feature_types(synthetic_classification)
    assert types["PassengerId"] == "identifier"
    assert types["Sex"] in {"binary", "categorical"}
    assert types["Age"] == "numerical"
    assert types["Pclass"] in {"ordinal", "categorical_numeric"}


def test_preprocessor_handles_missing(synthetic_classification):
    df = synthetic_classification.copy()
    df.loc[df.index[:20], "Age"] = None
    pre = DataPreprocessor(missing_threshold=0.9)
    cleaned = pre.handle_missing_values(df, target_col="Survived", fit=True)
    assert cleaned["Age"].isnull().sum() == 0
    assert len(cleaned) == len(df)


def test_validator_flags_empty():
    v = DataValidator()
    result = v.validate_dataset(pd.DataFrame())
    assert not result.is_valid


def test_validator_passes_clean_data(synthetic_classification):
    v = DataValidator()
    result = v.validate_dataset(synthetic_classification)
    assert result.is_valid, result.issues
